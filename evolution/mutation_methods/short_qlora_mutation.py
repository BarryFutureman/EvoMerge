from evolution.mutation import *
import random
from evolution.utils.qlora_sft_mutation import run_sft_mutation


class ShortQloraMutation(MutationMethod):
    def __init__(self, target_mutation_count, keep_top_k):
        super(ShortQloraMutation, self).__init__()
        self.target_mutation_count = target_mutation_count
        self.keep_top_k = keep_top_k

    def run_mutation(self, population):
        mutate_count = self.target_mutation_count
        mutate_dict = {}
        for lm in population:
            if lm.force_mutate:
                if not mutate_dict.get(lm):
                    mutate_dict[lm] = 0
                mutate_dict[lm] += 1
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
        mutation_info = run_sft_mutation(target.model_folder, num_mutation)
        print(mutation_info)
        target.set_mutation_info(mutation_info)
