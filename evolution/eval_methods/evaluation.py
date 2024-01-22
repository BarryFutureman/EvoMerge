class EvaluationMethod:
    def __init__(self):
        pass

    def run_evaluation(self, population):
        raise NotImplementedError()

    def do_eval(self, target, task_file):
        raise NotImplementedError()
