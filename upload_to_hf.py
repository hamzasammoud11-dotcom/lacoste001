"""Quick upload script for HuggingFace Spaces deployment."""
import os
from huggingface_hub import HfApi

# Read token
token_path = os.path.expanduser("~/.cache/huggingface/token")
with open(token_path) as f:
    token = f.read().strip()

api = HfApi()

# Upload the clean deployment folder
api.upload_folder(
    folder_path="C:/temp/bioflow_deploy",
    repo_id="vignt97867896/bioflow",
    repo_type="space",
    token=token,
)

print("Upload complete! Check https://huggingface.co/spaces/vignt97867896/bioflow")
