import gc
import torch.cuda

from evolution.eval_methods.evaluation import *
from evolution.utils.simple_lm_eval_run import simple_evaluate
from evolution.utils.random_hellaswag import randomly_sample_hellaswag
# from evolution.utils.random_gsm8k import randomly_sample_gsm8k
from evolution.utils.random_winogrande import randomly_sample_winogrande


class RandomEval(EvaluationMethod):
    def __init__(self, dataset_dir, config):
        super(RandomEval, self).__init__()
        self.dataset_dir = dataset_dir
        self.config = config
        
    def run_evaluation(self, population):
        tasks_file = self.generate_eval_data()

        for lm in population:
            fitness = self.do_eval(lm, tasks_file)
            # Something is probably wrong of the fitness is less than 0.4
            if fitness < 0.4:
                fitness = 0.000001
            lm.set_fitness(fitness)

        return population

    def generate_eval_data(self):
        """state = random.getstate()
        old_seed = state[1][0]
        print(old_seed)"""
        # random_size = random.randint(10, 20)
        hellaswag_json_file = randomly_sample_hellaswag(dataset_dir=self.dataset_dir,
                                                        sample_size=64 * self.config.eval_config.eval_size)
        # gsm8k_json_file = randomly_sample_gsm8k(dataset_dir=self.dataset_dir, sample_size=10)
        winogrande_json_file = randomly_sample_winogrande(dataset_dir=self.dataset_dir,
                                                          sample_size=128 * self.config.eval_config.eval_size)

        return [hellaswag_json_file, winogrande_json_file]

    def do_eval(self, target, task_files):
        print(f"Evaluating {target}")
        model = target.get_4bit_hf_model()
        tokenizer = target.get_hf_tokenizer()

        hellaswag_result = simple_evaluate(model=model, tokenizer=tokenizer, task="hellaswag_small_sample",
                                           task_json_file=task_files[0], device="cuda", normalized_acc=True,
                                           limit=None, batch_size=self.config.eval_config.batch_size)
        """gsm8k_result = simple_evaluate(model=model, tokenizer=tokenizer, task="gsm8k_small_sample",
                                       task_json_file=task_files[1], device="cuda",
                                       limit=None, batch_size=self.batch_size)"""
        winogrande_result = simple_evaluate(model=model, tokenizer=tokenizer, task="winogrande_small_sample",
                                            task_json_file=task_files[1], device="cuda",
                                            limit=None, batch_size=self.config.eval_config.batch_size)

        fitness = hellaswag_result * 0.6 + winogrande_result * 0.4  # + gsm8k_result * 0.2

        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("Currently cached:", torch.cuda.memory_cached())
        return fitness
    