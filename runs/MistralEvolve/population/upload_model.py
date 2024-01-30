from huggingface_hub import HfApi
from huggingface_hub import login

login(token="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")
api = HfApi()

repo_id = "Mistral-7b-instruct-v0.2-summ-dpo-ed3-Merged"
api.create_repo(repo_id=repo_id, private=False, repo_type="model")

file_folder = "C:\Files\TextGeneration\\text-generation-webui\models\Mistral-7b-instruct-v0.2-summ-dpo-ed3-Merged"
api.upload_folder(
    folder_path=file_folder,
    repo_type="model",
    repo_id="BarryFutureman/"+repo_id,
)