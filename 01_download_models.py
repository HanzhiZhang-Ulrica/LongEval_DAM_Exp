import argparse
import os
from huggingface_hub import snapshot_download

# Define your Hugging Face token (make sure to keep it secure)
hf_token = "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"

def download_full_repo(repo_id, base_save_dir):
    """
    Downloads the complete repository (all files) from Hugging Face Hub.

    Args:
        repo_id (str): The repository id (e.g., "HanzhiZhang/DAM_LLaMA_1B").
        base_save_dir (str): Base directory where repos should be saved.
    """
    # Create a folder name by replacing "/" with "_" to avoid file system issues.
    save_dir = os.path.join(base_save_dir, repo_id.replace('/', '_'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Downloading full repository for: {repo_id}")
    # snapshot_download fetches all files (including extra data and custom code)
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=save_dir,
        repo_type="model",
        token=hf_token
    )
    print(f"Repository '{repo_id}' downloaded to: {local_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download full Hugging Face model repositories (all files) for a list of models."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Base directory to save the downloaded model repositories."
    )
    args = parser.parse_args()

    models = [
        # "HanzhiZhang/H2O_LLaMA_1B",
        # "HanzhiZhang/StreamingLLM_LLaMA_1B",
        # "HanzhiZhang/MoA_LLaMA_1B",
        # "HanzhiZhang/DAM_LLaMA_1B",
        "HanzhiZhang/DAM_LLaMA_3B",
        # "meta-llama/Llama-3.2-1B-Instruct",
    ]

    for model in models:
        download_full_repo(model, args.model_path)
