class CrossoverMethod:
    def __init__(self):
        pass

    def run_crossover(self, old_population, new_population, pairs):
        raise NotImplementedError()

    def combine(self, p1, p2):
        raise NotImplementedError()
