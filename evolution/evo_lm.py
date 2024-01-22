import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig, LlamaConfig
import shutil
import re
import json
import random
import uuid
from transformers import AutoTokenizer


GPT2_TOKENIZER = AutoTokenizer.from_pretrained("gpt2", use_fast=True)


def generate_random_name(tokenizer):
    name = None
    while not name:
        random_word = ""
        for w in random.sample(range(100, 50000), 2):
            random_word += str(tokenizer.decode(w))
        name = re.sub(r'[^a-zA-Z]', ' ', random_word).strip()
    return name


class EvoDNA:
    def __init__(self, model_name, dna_id):
        self.id = dna_id
        self.model_name = model_name
        self.fitness = None
        self.parents = None
        self.crossover_info = None
        self.mutation_info = None

    def __repr__(self):
        return str({
            'id': self.id,
            'model_name': self.model_name,
            'fitness': self.fitness,
            'parents': self.parents,
            'mutation_info': self.mutation_info
        })

    def save_as_json(self, save_directory):
        # Create a dictionary with the attributes you want to save
        data = {
            'id': self.id,
            'model_name': self.model_name,
            'fitness': self.fitness,
            'parents': self.parents,
            'crossover_info': self.crossover_info,
            'mutation_info': self.mutation_info,
        }

        # Define the file path for saving
        file_path = f'{save_directory}/DNA-{self.id}.json'

        # Save the data as JSON
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)


class EvoLM:
    def __init__(self, model_folder, dna_id: str, model_name=None, parents=None):
        self.model_folder = model_folder
        if model_name is None:
            self.model_name = generate_random_name(tokenizer=GPT2_TOKENIZER)
        else:
            self.model_name = model_name

        self.DNA = EvoDNA(self.model_name, str(dna_id))
        if parents is not None:
            self.DNA.parents = [parents[0].DNA.id, parents[1].DNA.id]
        self.force_mutate = False

        self.config = AutoConfig.from_pretrained(self.model_folder)

    def __repr__(self):
        return f"EvoLM <{self.model_name} - {self.DNA.id}>"

    def set_fitness(self, value):
        self.DNA.fitness = value

    def get_fitness(self):
        if not self.DNA.fitness:
            return 0
        else:
            return self.DNA.fitness

    def set_mutation_info(self, info):
        self.DNA.mutation_info = info

    def get_4bit_hf_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_folder,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
        )
        return hf_model

    def get_hf_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_folder, use_fast=True)

    def delete_files(self):
        try:
            shutil.rmtree(self.model_folder)
        except OSError as e:
            raise OSError(f"Trying to delete {self.model_folder}, got error: {e}")

    def get_copy(self, population_folder):
        random_uuid = uuid.uuid4()
        output_path = f"{population_folder}/lm-{random_uuid}"
        source_path = self.model_folder
        # os.makedirs(output_path, exist_ok=True)

        shutil.copytree(source_path, output_path)
        new_lm = EvoLM(model_folder=output_path, dna_id=str(uuid.uuid4()))
        new_lm.DNA.parents = [self.DNA.id, self.DNA.id]
        return new_lm
