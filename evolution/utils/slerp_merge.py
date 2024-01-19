import torch
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

LORA_MERGE_CACHE = "/tmp"


def run_slerp_merge(config: dict, output_path):
    merge_config = MergeConfiguration.model_validate(config)

    run_merge(
        merge_config,
        out_path=output_path,
        options=MergeOptions(
            lora_merge_cache=LORA_MERGE_CACHE,
            cuda=True,
            copy_tokenizer=True,
            lazy_unpickle=False,
            low_cpu_memory=True,
        ),
    )

    torch.cuda.empty_cache()
    print("Done!")
