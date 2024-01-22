from huggingface_hub import snapshot_download

model_repos = [
    # "mlabonne/NeuralBeagle14-7B",
    # "udkai/Turdus",
    "mlabonne/NeuralDaredevil-7B",
    "PetroGPT/Severus-7B-DPO",
    # "Weyaxi/MetaMath-Chupacabra-7B-v2.01-Slerp",
    "senseable/Westlake-7B",
]

for repo in model_repos:
    folder_name = repo.split("/", 1)[-1]
    snapshot_download(repo_id=repo, local_dir=f"./{folder_name}", local_dir_use_symlinks=False)
