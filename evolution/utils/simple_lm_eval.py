import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Union

import os
from lm_eval.utils import (
    positional_deprecated,
    run_task_tests,
    get_git_commit_hash,
    simple_parse_args_string,
    eval_logger,
)
from lm_eval.tasks import include_path
from lm_eval.api.registry import ALL_TASKS
from lm_eval.evaluator import *
from lm_eval.models.huggingface import HFLM
from lm_eval import utils


def initialize_tasks(verbosity="INFO"):
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))

    task_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    include_path(task_dir)


initialize_tasks()


model_name = "stella-sft-1b-lora-merge"
model_path = f"cache/models/{model_name}"
TASKs = ["hellaswag_small_sample"]
OUTPUT_PATH = f"results_output/{model_name}/{TASKs}"
SHOTs = 10
DEVICE = "cuda"
BATCH = "8"
LIMIT = None


@positional_deprecated
def simple_evaluate(
    model,
    tokenizer,
    tasks=[],
    num_fewshot=None,
    batch_size=None,
    max_batch_size=None,
    device=None,
    use_cache=None,
    limit=None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    decontamination_ngrams_path=None,
    write_out: bool = False,
    log_samples: bool = True,
    gen_kwargs: str = None,
):
    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(
        1234
    )  # TODO: this may affect training runs that are run with evaluation mid-run.

    assert (
        tasks != []
    ), "No tasks specified, or no tasks found. Please verify the task names."

    # Load tasks ============================

    tasks_list = tasks
    task_names = utils.pattern_match(tasks_list, ALL_TASKS)
    for task in [task for task in tasks_list if task not in task_names]:
        if os.path.isfile(task):
            config = utils.load_yaml_config(task)
            task_names.append(config)
    task_missing = [
        task
        for task in tasks_list
        if task not in task_names and "*" not in task
    ]  # we don't want errors if a wildcard ("*") task name was used

    if task_missing:
        missing = ", ".join(task_missing)
        eval_logger.error(
            f"Tasks were not found: {missing}\n"
            f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
        )
        raise ValueError(
            f"Can't find this task."
        )
    tasks = task_names
    # ======================================

    lm = HFLM(pretrained=model, tokenizer=tokenizer,

              batch_size=batch_size,
              max_batch_size=max_batch_size,
              device=device,
    )

    if gen_kwargs is not None:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(
            "generation_kwargs specified through cli, these settings will be used over set parameters in yaml tasks."
        )
        if gen_kwargs == "":
            gen_kwargs = None

    if use_cache is not None:
        print(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank"
            + str(lm.rank)
            + ".db",
        )

    task_dict = lm_eval.tasks.get_task_dict(tasks)
    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if type(task_obj) == tuple:
            group, task_obj = task_obj
            if task_obj is None:
                continue

        config = task_obj._config
        if config["output_type"] == "generate_until" and gen_kwargs is not None:
            config["generation_kwargs"].update(gen_kwargs)

        if num_fewshot is not None:
            if config["num_fewshot"] == 0:
                eval_logger.info(
                    f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                )
            else:
                default_num_fewshot = config["num_fewshot"]
                eval_logger.warning(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                )

                task_obj._config["num_fewshot"] = num_fewshot

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        log_samples=log_samples,
    )

    if lm.rank == 0:
        # add info about the model and few shot config
        results["config"] = {
            "model": model
            if isinstance(model, str)
            else model.model.config._name_or_path,
            "batch_size": batch_size,
            "batch_sizes": list(lm.batch_sizes.values())
            if hasattr(lm, "batch_sizes")
            else [],
            "device": device,
            "use_cache": use_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
            "gen_kwargs": gen_kwargs,
        }
        if results is not None:
            for k, dic in results["results"].items():
                n = str(results["n-shot"][k])

                for (mf), v in dic.items():
                    m, _, f = mf.partition(",")
                    if m.endswith("_stderr"):
                        continue
                    if m == "acc_norm":
                        return float("%.4f" % v)
        return results
    else:
        return None


if __name__ == "__main__":
    simple_evaluate(model=model_name, base_model_path=model_path,
                    tasks=TASKs, num_fewshot=SHOTs, device=DEVICE,
                    limit=LIMIT, batch_size=BATCH)
