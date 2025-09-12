from huggingface_hub import HfApi, login, whoami

# --- CONFIG ---
REPO_ID = "bowen-upenn/ImplicitPersona"
REPO_TYPE = "dataset"
# ---------------

# Check login state
try:
    whoami()
    print("🔑 Hugging Face token already available, skipping login.")
except Exception:
    print("⚠️ No token found, please log in...")
    login()

api = HfApi()

# Upload large folders using upload_large_folder (uploads to root, no custom path_in_repo)
# Note: upload_large_folder uploads directly to repo root, so folder structure must match desired repo structure
large_folders = [
    "data/raw_data",
    "data/chat_history_32k", 
    "data/chat_history_128k",
    "data/chat_history_multimodal_32k",
    "data/chat_history_multimodal_128k",
]

for folder_path in large_folders:
    folder_name = folder_path.split('/')[-1]  # Get just the folder name
    print(f"⬆️ Uploading {folder_path}/ (large folder mode)")
    print(f"   Note: Files will be uploaded to repo root. Folder structure: {folder_name}/...")
    api.upload_large_folder(
        folder_path=folder_path,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
# Upload small individual files
files = [
    "data/irrelevant/combined_irrelevant_data.json",
    "data/benchmark.csv",
    "data/benchmark_multimodal.csv",
]

for f in files:
    path_in_repo = f.replace("data/", "")
    print(f"⬆️ Uploading {f} -> {path_in_repo}")
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message=f"Add {path_in_repo}"
    )

print("✅ Upload complete!")
