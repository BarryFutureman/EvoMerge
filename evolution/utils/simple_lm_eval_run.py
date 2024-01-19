import gc
import time
from lm_eval.tasks import ConfigurableTask
from lm_eval.evaluator import *
from lm_eval.models.huggingface import HFLM
from evolution.utils.tasks.hellaswag_small_sample.utils import process_docs
from evolution.utils.tasks.winogrande_small_sample\
    .preprocess_winogrande import doc_to_text, doc_to_target, doc_to_choice


hellaswag_config = dict(task='hellaswag_small_sample', group=['multiple_choice'],
                        dataset_path='json', dataset_name="json",
                        dataset_kwargs=None,  # We need to fill this
                        training_split='train', validation_split='train', test_split=None,
                        fewshot_split='train', process_docs=process_docs,
                        doc_to_text='{{query}}', doc_to_target='{{label}}', doc_to_choice='choices',
                        process_results=None, use_prompt=None, description='', target_delimiter=' ',
                        fewshot_delimiter='\n\n', num_fewshot=5,
                        metric_list=[{'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True},
                                    {'metric': 'acc_norm', 'aggregation': 'mean', 'higher_is_better': True}],
                        output_type='multiple_choice', generation_kwargs=None, repeats=1,
                        should_decontaminate=False,
                        metadata={'version': 1.0})

gsm8k_config = dict(task='gsm8k_small_sample', task_alias=None, group=['math_word_problems'], group_alias=None,
                    dataset_path='json', dataset_name="json", dataset_kwargs=None,
                    training_split='train', validation_split=None,
                    test_split='train', fewshot_split='train', process_docs=None,
                    doc_to_text='Question: {{question}}\nAnswer:',
                    doc_to_target='{{answer}}', doc_to_choice=None, process_results=None, use_prompt=None,
                    description='', target_delimiter=' ', fewshot_delimiter='\n\n', fewshot_config=None, num_fewshot=5,
                    metric_list=[{'metric': 'exact_match', 'aggregation': 'mean', 'higher_is_better': True,
                                  'ignore_case': True, 'ignore_punctuation': False,
                                  'regexes_to_ignore': [',', '\\$', '(?s).*#### ']}],
                    output_type='generate_until',
                    generation_kwargs={'until': ['\n\n', 'Question:'], 'do_sample': False,
                                       'temperature': 0.0}, repeats=1,
                    filter_list=[{'name': 'get-answer',
                                  'filter': [{'function': 'regex', 'regex_pattern': '#### (\\-?[0-9\\.\\,]+)'},
                                             {'function': 'take_first'}]}],
                    should_decontaminate=False, doc_to_decontamination_query=None, metadata={'version': 2.0})

winogrande_config = dict(task='winogrande_small_sample', task_alias=None, group=None, group_alias=None,
                         dataset_path='json', dataset_name='json', dataset_kwargs=None,
                         training_split='train', validation_split='train', test_split=None, fewshot_split='train',
                         process_docs=None, doc_to_text=doc_to_text, doc_to_target=doc_to_target,
                         doc_to_choice=doc_to_choice, process_results=None, use_prompt=None, description='',
                         target_delimiter=' ', fewshot_delimiter='\n\n', fewshot_config=None, num_fewshot=2,
                         metric_list=[{'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True}],
                         output_type='multiple_choice', generation_kwargs=None, repeats=1, filter_list=None,
                         should_decontaminate=True, doc_to_decontamination_query='sentence', metadata={'version': 1.0})

# TODO: Add ARC too?

EVAL_CONFIGS = {"hellaswag_small_sample": hellaswag_config, "gsm8k_small_sample": gsm8k_config,
                "winogrande_small_sample": winogrande_config}


@positional_deprecated
def simple_evaluate(
        model,
        tokenizer,
        task,
        task_json_file,
        normalized_acc=False,
        # num_fewshot=None,
        batch_size=None,
        max_batch_size=None,
        device=None,
        use_cache=None,
        limit=None,
        bootstrap_iters: int = 100000,
        # check_integrity: bool = False,
        decontamination_ngrams_path=None,
        write_out: bool = False,
        log_samples: bool = True,
        gen_kwargs: str = None,
):
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

    task_config = EVAL_CONFIGS[task]
    task_config["dataset_kwargs"] = {'data_files': task_json_file}
    the_task = ConfigurableTask(config=task_config)
    task_dict = {"Awesome": the_task}

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        log_samples=log_samples,
    )

    # eval changes the seed, we reseed here just in case
    time_based_seed = int(time.time())
    random.seed(time_based_seed)
    np.random.seed(time_based_seed)
    torch.manual_seed(
        time_based_seed
    )

    # add info about the model and few shot config
    """results["config"] = {
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
    }"""
    score = 0
    if results is not None:
        for k, dic in results["results"].items():
            n = str(results["n-shot"][k])

            for (mf), v in dic.items():
                m, _, f = mf.partition(",")
                if m.endswith("_stderr"):
                    continue
                if normalized_acc and m == "acc_norm":
                    score = float("%.4f" % v)
                elif m == "acc":
                    score = float("%.4f" % v)

    """del lm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    print("lm eval cached:", torch.cuda.memory_cached())
    time.sleep(1)"""
    return score
