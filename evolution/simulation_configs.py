import yaml


class SimulationConfig:
    def __init__(self, target_iterations, keep_top_k, mutate_count, max_population_size, expand_steps,
                 evaluation, selection, crossover, mutation):
        eval_config = EvaluationConfig(**evaluation)
        selection_config = SelectionConfig(**selection)
        crossover_config = CrossoverConfig(**crossover)
        mutation_config = MutationConfig(**mutation)
        self.target_iterations = target_iterations
        self.keep_top_k = keep_top_k
        self.mutate_count = mutate_count
        self.max_population_size = max_population_size
        self.expand_steps = expand_steps
        self.eval_config = eval_config
        self.selection_config = selection_config
        self.crossover_config = crossover_config
        self.mutation_config = mutation_config

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = '\n    '.join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{class_name}(\n    {attributes}\n)"


class EvaluationConfig:
    def __init__(self, eval_size, batch_size):
        self.eval_size = eval_size
        self.batch_size = batch_size

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ', '.join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{class_name}({attributes})"


class SelectionConfig:
    def __init__(self, nothing_here):
        self.nothing_here = nothing_here

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ', '.join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{class_name}({attributes})"


class CrossoverConfig:
    def __init__(self, possible_shifts, possible_cycles, curve_type):
        self.curve_type = curve_type
        self.possible_shifts = possible_shifts
        self.possible_cycles = possible_cycles

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ', '.join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{class_name}({attributes})"


class MutationConfig:
    def __init__(self, mutation_strength, possible_lora_r, possible_lora_alpha, possible_beta,
                 per_device_train_batch_size, gradient_accumulation_steps, possible_lr):
        self.possible_lr = possible_lr
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mutation_strength = mutation_strength
        self.possible_lora_r = possible_lora_r
        self.possible_lora_alpha = possible_lora_alpha
        self.possible_beta = possible_beta

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ', '.join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{class_name}({attributes})"


def load_config(file_path):
    with open(file_path, 'r') as stream:
        data = yaml.safe_load(stream)
        return SimulationConfig(**data)


if __name__ == '__main__':
    config = load_config('configs/example_run.yaml')
    print(config)
