import argparse
import os
import shutil
from huggingface_hub import snapshot_download

def download_dataset(repo_id, dest_dir, token=None):
    """
    Downloads the MVTec LOCO AD dataset from a Hugging Face Dataset repository.
    """
    # Fallback/convenience check: if local mvtec_loco exists in workspace, copy it
    local_source = "mvtec_loco"
    if os.path.exists(local_source) and not os.path.exists(dest_dir):
        print(f"Found local '{local_source}' folder. Copying to '{dest_dir}' for local training compatibility...")
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        shutil.copytree(local_source, dest_dir)
        print("Local folder copied successfully.")
        return
        
    print(f"Downloading dataset from Hugging Face repository '{repo_id}' to '{dest_dir}'...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
            token=token
        )
        print(f"Dataset downloaded successfully to: {dest_dir}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your repository ID and token.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MVTec LOCO AD dataset from Hugging Face.")
    parser.add_argument("--repo_id", type=str, required=True, help="HF dataset repository ID (e.g., username/mvtec-loco).")
    parser.add_argument("--dest_dir", type=str, default="data/mvtec_loco", help="Target local folder to save dataset.")
    parser.add_argument("--token", type=str, default=None, help="Optional HF token for private dataset repositories.")
    
    args = parser.parse_args()
    
    download_dataset(args.repo_id, args.dest_dir, args.token)
