from huggingface_hub import snapshot_download

model_repos = [
    # "mlabonne/NeuralBeagle14-7B",
    "flemmingmiguel/MDBX-7B",
    "BarryFutureman/Mistral-7b-instruct-v0.2-summ-dpo-ed2-Merged",
    "BarryFutureman/WildMarcoroni-Variant3-7B",
    # "Weyaxi/MetaMath-Chupacabra-7B-v2.01-Slerp",
    "alnrg2arg/blockchainlabs_7B_merged_test2_4",
]

for repo in model_repos:
    folder_name = repo.split("/", 1)[-1]
    snapshot_download(repo_id=repo, local_dir=f"./{folder_name}", local_dir_use_symlinks=False)
