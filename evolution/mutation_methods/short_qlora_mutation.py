from evolution.mutation import *
import random
from evolution.utils.qlora_dpo_mutation import run_dpo_mutation


class ShortQloraMutation(MutationMethod):
    def __init__(self, target_mutation_count, keep_top_k, mutation_strength=1):
        super(ShortQloraMutation, self).__init__()
        self.target_mutation_count = target_mutation_count
        self.keep_top_k = keep_top_k
        self.mutation_strength = mutation_strength

    def run_mutation(self, population):
        mutate_count = self.target_mutation_count
        mutate_dict = {}
        for lm in population:
            if lm.force_mutate:
                if not mutate_dict.get(lm):
                    mutate_dict[lm] = 0
                mutate_dict[lm] += 1
                lm.force_mutate = False
                mutate_count -= 1
        if mutate_count > 0:
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
        mutation_info = run_dpo_mutation(target.model_folder, num_mutation, self.mutation_strength)
        print(mutation_info)
        target.set_mutation_info(mutation_info)
