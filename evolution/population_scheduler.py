import numpy as np


class PopulationScheduler:
    def __init__(self, initial_population_size, max_population_size, min_population_size, total_generations,
                 expand_steps):
        self.initial_population_size = initial_population_size
        self.max_population_size = max_population_size
        self.min_population_size = min_population_size
        self.total_generations = total_generations
        self.expand_steps = expand_steps

    def cosine_annealing(self, current_generation):
        if current_generation <= self.expand_steps-1:
            # Increase population for the first half
            return int(self.initial_population_size + (self.max_population_size - self.initial_population_size) *
                       (1 - np.cos(np.pi * (current_generation+1) / self.expand_steps)) / 2)
        else:
            # Decrease population for the second half
            return self.min_population_size + \
                   int((self.max_population_size - self.min_population_size) / 2
                       * (1 + np.cos(np.pi * (current_generation - self.expand_steps) /
                                     (self.total_generations - self.expand_steps))))

    def get_population_size(self, current_generation):
        # Get the current population size based on the cosine annealing schedule
        population_size = self.cosine_annealing(current_generation)
        return min(population_size, self.max_population_size)


if __name__ == '__main__':
    # Example usage:
    initial_population_size = 4
    max_population_size = 32
    total_generations = 16
    population_expand_steps = 2

    scheduler = PopulationScheduler(initial_population_size, max_population_size, initial_population_size,
                                    total_generations, population_expand_steps)

    for generation in range(total_generations):
        current_population_size = scheduler.get_population_size(generation)
        print(f"Generation {generation + 1}: Population Size = {current_population_size}")
