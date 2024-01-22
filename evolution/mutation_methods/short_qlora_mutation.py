from evolution.mutation_methods.mutation import *
import random
from evolution.utils.qlora_dpo_mutation import run_dpo_mutation


class ShortQloraMutation(MutationMethod):
    def __init__(self, config):
        super(ShortQloraMutation, self).__init__()
        self.config = config
        self.keep_top_k = config.keep_top_k
        """self.target_mutation_count = target_mutation_count
        self.mutation_strength = mutation_strength"""

    def run_mutation(self, population):
        mutate_count = self.config.mutate_count
        mutate_dict = {}
        for lm in population:
            if lm.force_mutate:
                if not mutate_dict.get(lm):
                    mutate_dict[lm] = 0
                mutate_dict[lm] += 1
                lm.force_mutate = False
                mutate_count -= 1
        if mutate_count > 0:
            mutate_count = min((len(population) - self.keep_top_k), mutate_count)
            random_indexes = random.sample(range(self.keep_top_k, len(population)), mutate_count)
            for i in random_indexes:
                lm = population[i]
                if not mutate_dict.get(lm):
                    mutate_dict[lm] = 0
                mutate_dict[lm] += 1

        for k in mutate_dict.keys():
            self.mutate(k, mutate_dict[k])

        return population

    def mutate(self, target, num_mutation):
        if num_mutation == 0:
            return
        mutation_info = run_dpo_mutation(target.model_folder, num_mutation,
                                         self.config.mutation_config)
        print(mutation_info)
        target.set_mutation_info(mutation_info)
