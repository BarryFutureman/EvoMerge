from huggingface_hub import HfApi
from huggingface_hub import login

login(token="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")
api = HfApi()

repo_id = "NeuralBeagleTurtus"
api.create_repo(repo_id=repo_id, private=False, repo_type="model")

file_folder = "lm-03608072-2a50-4ebb-ab74-30b6a50c2599"
api.upload_folder(
    folder_path=file_folder,
    repo_type="model",
    repo_id="BarryFutureman/"+repo_id,
)