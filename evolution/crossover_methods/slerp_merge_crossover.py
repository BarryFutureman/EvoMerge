import random
from evolution.crossover import *
from evolution.utils import slerp_merge
from evolution.evo_lm import EvoLM
import numpy as np
import uuid


class SlerpMergeCrossover(CrossoverMethod):
    def __init__(self, population_folder):
        super(SlerpMergeCrossover, self).__init__()
        self.population_folder = population_folder

    def run_crossover(self, old_population: list, new_population: list, pairs):
        for pair in pairs:
            p1, p2 = pair
            new_lm = None
            if p1 == p2:
                # Force mutate if a copy already exist in population
                if p1 in new_population:
                    # Force mutate
                    print("Force Mutate: ", p1)
                    new_lm = p1.get_copy(self.population_folder)
                    new_lm.force_mutate = True
                # Else we just put it in the new population
                else:
                    new_lm = p1
            else:
                new_lm = self.combine(p1, p2)
            new_population.append(new_lm)

        return new_population

    def combine(self, p1, p2):
        num_layers = p1.config.num_hidden_layers

        shift = random.choice([0, 0.5, 1])  # random.randint(0, 10) / 10
        cycles = random.choice([0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 6])
        self_attn_t_curve = self.generate_normalized_cosine_list(num_points=num_layers,
                                                                 num_cycles=cycles, phase_shift=np.pi * shift)

        shift = random.choice([0, 0.5, 1])  # random.randint(0, 10) / 10
        cycles = random.choice([0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 6])
        mlp_t_curve = self.generate_normalized_cosine_list(num_points=num_layers,
                                                           num_cycles=cycles, phase_shift=np.pi * shift)

        merge_config_dict = {'slices': [
            {'sources': [{'model': p1.model_folder, 'layer_range': [0, num_layers]},
                         {'model': p2.model_folder, 'layer_range': [0, num_layers]}]}],
            'merge_method': 'slerp',
            'base_model': p1.model_folder,
            'parameters': {'t': [
                {'filter': 'self_attn', 'value': self_attn_t_curve},
                {'filter': 'mlp', 'value': mlp_t_curve},
                {'value': 0.5}
            ]},
            'dtype': 'float16',
            'tokenizer_source': 'union'}

        random_uuid = uuid.uuid4()
        merge_output_path = f"{self.population_folder}/lm-{random_uuid}"
        slerp_merge.run_slerp_merge(merge_config_dict, output_path=merge_output_path)
        new_lm = EvoLM(merge_output_path, dna_id=str(random_uuid), parents=[p1, p2])
        return new_lm

    @staticmethod
    def generate_normalized_cosine_list(num_points, num_cycles, phase_shift: float = 0):
        # Generate an array of equally spaced points from 0 to 2*pi*num_cycles
        x_values = np.linspace(0, 2 * np.pi * num_cycles, num_points)

        # Compute the cosine values for each point with the specified phase shift
        cosine_values = np.cos(x_values + phase_shift)

        # Normalize the values between 0 and 1
        normalized_cosine_values = (cosine_values - np.min(cosine_values)) / (
                np.max(cosine_values) - np.min(cosine_values))

        # Round the values to two decimal places and convert to Python list
        rounded_values_list = normalized_cosine_values.round(2).tolist()

        return rounded_values_list
