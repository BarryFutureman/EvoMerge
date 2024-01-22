import random
import time
import numpy as np
import torch
import shutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import DPOTrainer

DISTILABEL_ORCA = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")
DISTILABEL_ORCA = DISTILABEL_ORCA.filter(
    lambda r:
        r["status"] != "tie" and
        r["chosen_score"] >= 8 and
        len(r["input"])+len(r["chosen"]) <= 1024
)
ALL_LORA_TARGETS = ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
LEARNING_RATES = [1e-5, 5e-5, 6e-6, 1e-6, 2e-6, 5e-7, 2e-7]


def merge_lora(base_model_path, tokenizer, lora_adapter_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=False,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir="cache/models",
    )

    peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path,
                                           local_files_only=True)
    model = peft_model.merge_and_unload()

    # Delete original model and save the merged model in the same folder
    shutil.rmtree(base_model_path)
    shutil.rmtree(lora_adapter_path)
    model.save_pretrained(base_model_path)
    tokenizer.save_pretrained(base_model_path)

    torch.cuda.empty_cache()
    return


def get_mutation_dataset(num_datasets, mutation_strength):
    dataset_names = ["distilabel-intel-orca-dpo-pairs"]
    max_sample_size = num_datasets * 32 * mutation_strength
    sampled_dataset = DISTILABEL_ORCA.shuffle().select(i for i in range(max_sample_size))

    return sampled_dataset, dataset_names


def run_dpo_mutation(model_path, num_mutations, mutation_config):
    # Define Mutation Parameters
    mutation_strength = mutation_config.mutation_strength
    lora_r = random.choice(mutation_config.possible_lora_r)
    lora_alpha = random.choice(mutation_config.possible_lora_alpha)
    # TODO: For now we don't randomize lora targets,
    #  this would be something to experiment with in the future
    # lora_targets = random.sample(ALL_LORA_TARGETS, random.randint(2, len(ALL_LORA_TARGETS)))
    num_epochs = num_mutations
    warmup_ratio = 0.1  # round(random.uniform(0.05, 0.2), 2)
    lr = random.choice(LEARNING_RATES)
    beta = random.choice(mutation_config.possible_beta)

    # Load dataset
    train_dataset, selected_dataset_names = get_mutation_dataset(num_mutations, mutation_strength)

    # Info
    mutation_info = {"datasets": selected_dataset_names, "strength": mutation_strength,
                     "lora_r": lora_r, "lora_alpha": lora_alpha, "num_mutation": num_mutations,
                     "beta": beta, "learning_rate": lr}

    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.padding_side = "right"

    def split_prompt_and_responses(samples) -> dict[str, str]:
        msgs = [
            {"role": "system", "content": samples["system"]},
            {"role": "user", "content": samples["input"]},
        ]

        return {
            "prompt": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
            "chosen": samples["chosen"] + "</s>",
            "rejected": samples["rejected"] + "</s>",
        }

    original_columns = train_dataset.column_names
    dataset = train_dataset.map(
        split_prompt_and_responses,
        remove_columns=original_columns,
    )

    # Load Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )
    # model.required_grad = True
    model.enable_input_require_grads()

    # LoRA configuration
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )

    # Training arguments
    adapter_path = model_path + "-lora-adapter"
    training_args = TrainingArguments(
        per_device_train_batch_size=mutation_config.per_device_train_batch_size,
        gradient_accumulation_steps=mutation_config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=lr,
        lr_scheduler_type="cosine",
        num_train_epochs=num_epochs,
        save_strategy="no",
        logging_steps=1,
        output_dir=adapter_path,
        optim="adamw_torch",
        warmup_ratio=warmup_ratio,
        bf16=True,
        report_to=["none"],
        remove_unused_columns=False,
    )

    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=0.01,
        max_prompt_length=512,
        max_length=1024,
        loss_type="kto_pair",  # ['sigmoid', 'hinge', 'ipo', 'kto_pair']
    )

    dpo_trainer.train()

    dpo_trainer.save_model(adapter_path)

    merge_lora(model_path, tokenizer, adapter_path)

    # The random seeds are changed here,
    # reseed everything as a sanity check
    time_based_seed = int(time.time())
    random.seed(time_based_seed)
    np.random.seed(time_based_seed)
    torch.manual_seed(
        time_based_seed
    )

    return mutation_info


if __name__ == '__main__':
    run_dpo_mutation("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2, 1)
