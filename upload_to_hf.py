import argparse
import os
from huggingface_hub import HfApi

def upload_dataset(local_dir, repo_id, token):
    """
    Uploads the local dataset folder to a Hugging Face Dataset repository.
    """
    print(f"Initializing Hugging Face API...")
    api = HfApi(token=token)
    
    # Create dataset repo if it doesn't exist
    print(f"Creating/verifying dataset repository: {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repository {repo_id} is ready.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
        
    print(f"Uploading files from '{local_dir}' to '{repo_id}' on Hugging Face...")
    print("This might take a few minutes depending on your internet connection...")
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload MVTec LOCO AD dataset"
        )
        print("Dataset uploaded successfully!")
    except Exception as e:
        print(f"Error uploading files: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload local MVTec LOCO AD dataset to Hugging Face.")
    parser.add_argument("--local_dir", type=str, default="mvtec_loco", help="Path to local mvtec_loco folder.")
    parser.add_argument("--repo_id", type=str, required=True, help="HF dataset repository ID (e.g., username/mvtec-loco).")
    parser.add_argument("--token", type=str, required=True, help="Your Hugging Face Write Token.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.local_dir):
        print(f"Error: Local directory '{args.local_dir}' not found.")
    else:
        upload_dataset(args.local_dir, args.repo_id, args.token)
