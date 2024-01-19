from evolution import selection
import numpy as np
import random


class RankBasedSelection(selection.SelectionMethod):
    def __init__(self, keep_top_k):
        super(RankBasedSelection, self).__init__()
        self.keep_top_k = keep_top_k

    def run_selection(self, population):
        fitness_scores = [p.get_fitness() for p in population]
        sorted_population = sorted(population, key=lambda x: x.get_fitness(), reverse=True)
        keep_population = sorted_population[:self.keep_top_k]

        num_parents_to_select = 2
        pairs = []
        for p in range(len(population) - self.keep_top_k):
            selected = self.roulette_wheel_selection(population, fitness_scores, num_parents_to_select)
            pairs.append(selected)

        new_population = keep_population

        return pairs, new_population

    @staticmethod
    def roulette_wheel_selection(population, fitness_scores, num_parents=2):
        total_fitness = sum(fitness_scores)
        probabilities = [fitness / total_fitness for fitness in fitness_scores]

        selected_pair = []
        for _ in range(num_parents):
            spin = random.uniform(0, 1)
            cumulative_probability = 0

            for index, prob in enumerate(probabilities):
                cumulative_probability += prob
                if spin <= cumulative_probability:
                    selected_pair.append(population[index])
                    break

        return selected_pair


if __name__ == '__main__':
    s = RankBasedSelection(2)
    for i in range(6):
        result = s.roulette_wheel_selection(["A", "B", "C", "D", "E", "F"], [0.52, 0.45, 0.5, 0.2, 0.85, 0.51])
        print(result)
