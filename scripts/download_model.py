import os
import sys

def download_qwen():
    print("Checking dependencies...")
    try:
        import torch
        import transformers
        import huggingface_hub
        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers version: {transformers.__version__}")
        print(f"Hugging Face Hub version: {huggingface_hub.__version__}")
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please make sure you have installed the requirements using pip.")
        sys.exit(1)
        
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
    local_dir = os.path.abspath("./models/qwen")
    
    print(f"Starting download of '{repo_id}' directly to '{local_dir}'...")
    
    # We will use huggingface_hub's snapshot_download to fetch the complete repository
    # and save it directly to the local directory without using symlinks.
    try:
        from huggingface_hub import snapshot_download
        
        # snapshot_download downloads all files from the model repository
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Writes physical files, not symlinks, making it portable
            repo_type="model"
        )
        print("\nDownload completed successfully!")
        
        # Verification: list downloaded files and calculate total size
        print("\nDownloaded files verification list:")
        total_size_bytes = 0
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, local_dir)
                size_bytes = os.path.getsize(file_path)
                total_size_bytes += size_bytes
                size_mb = size_bytes / (1024 * 1024)
                print(f" - {rel_path} ({size_mb:.2f} MB)")
        
        total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
        print(f"\nTotal Model Size: {total_size_gb:.2f} GB")
                
    except Exception as e:
        print(f"An error occurred during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_qwen()
