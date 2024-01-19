class MutationMethod:
    def __init__(self):
        pass

    def run_mutation(self, population):
        raise NotImplementedError()

    def mutate(self, target, num_mutation):
        raise NotImplementedError()
