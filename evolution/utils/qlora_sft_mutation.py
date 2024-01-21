import random
import time
import numpy as np
import torch
import shutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


DATASETS = ["arc_easy_intuition.json", "camel_math_qa.json"]
# TODO: the correct dataset folder is actually within the simulation folder
ALL_LORA_TARGETS = ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
LEARNING_RATES = [1e-4, 2e-4, 6e-4, 1e-5, 2e-5, 5e-5]


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


def get_mutation_dataset(num_datasets):
    dataset_lst = []
    dataset_names = []
    for n in range(num_datasets):
        dataset_name = DATASETS[random.randint(0, len(DATASETS)-1)]
        selected_dataset_path = os.path.dirname(os.path.abspath(__file__)) + "/data/" + dataset_name
        selected_dataset = load_dataset("json", data_files=[selected_dataset_path])["train"]

        max_sample_size = 32  # TODO: Maybe this could be random?
        if len(selected_dataset) > max_sample_size:
            selected_dataset = selected_dataset.shuffle().select([i for i in range(max_sample_size)])

        dataset_lst.append(selected_dataset)
        dataset_names.append(dataset_name)
    joined_dataset = concatenate_datasets(dataset_lst)

    return joined_dataset, dataset_names


def run_sft_mutation(model_path, num_mutations):
    # TODO: Save lora adapters, merge lora as model-mutated,
    #  delete lora and original model, update EvoLM model file path

    # Define Mutation Parameters
    lora_r = random.choice([8, 16, 32])
    lora_alpha = random.choice([8, 16, 32])
    # TODO: For now we don't randomize lora targets,
    #  this would be something to experiment with in the future
    # lora_targets = random.sample(ALL_LORA_TARGETS, random.randint(2, len(ALL_LORA_TARGETS)))
    num_epochs = num_mutations * 2
    warmup_ratio = 0.1  # round(random.uniform(0.05, 0.2), 2)
    lr = random.choice(LEARNING_RATES)

    # Load dataset
    train_dataset, selected_dataset_names = get_mutation_dataset(num_mutations)

    # Info
    mutation_info = {"datasets": selected_dataset_names, "lora_r": lora_r, "lora_alpha": lora_alpha, "num_mutation": num_mutations,
                     "warmup_ratio": warmup_ratio, "learning_rate": lr}

    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% " \
                              "endif %}{% for message in messages %}{{'<s>' + message['role'] + '\n' + message[" \
                              "'content'] + '</s>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ " \
                              "'<s>assistant\n' }}{% endif %} "
    tokenizer.padding_side = "right"

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
    model.required_grad = True

    # LoRA configuration
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )

    # Format data
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['question'])):
            messages = [
                {"role": "user", "content": example["question"][i]},
                {"role": "assistant", "content": example["answer"][i]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    response_template = "assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # Training arguments
    adapter_path = model_path + "-lora-adapter"

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        num_train_epochs=num_epochs,
        save_strategy="no",
        logging_steps=1,
        output_dir=adapter_path,
        optim="adamw_bnb_8bit",
        warmup_ratio=warmup_ratio,
        bf16=True,
        report_to=["none"],
    )

    sft_trainer = SFTTrainer(
        model,
        args=training_args,
        max_seq_length=2048,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    sft_trainer.train()
    del model
    torch.cuda.empty_cache()

    sft_trainer.save_model(adapter_path)

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
    run_sft_mutation("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2)
