import os
import shutil
import tempfile
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, whoami
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# --- CONFIG ---
REPO_ID = "bowen-upenn/ImplicitPersona"
REPO_TYPE = "dataset"
# ---------------

# Check login state
try:
    whoami()
    print("🔑 Hugging Face token already available, skipping login.")
except Exception:
    print("⚠️ No token found, loading from .env file...")
    hf_token = os.getenv('HF_ACCESS_TOKEN')
    if hf_token:
        login(token=hf_token)
        print("🔑 Successfully logged in using token from .env file.")
    else:
        print("❌ HF_ACCESS_TOKEN not found in .env file. Please add it or log in manually.")
        login()

api = HfApi()

# # First, clean up any misplaced JSON files in the root directory
# print("🧹 Cleaning up misplaced JSON files in root directory...")
# try:
#     repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
#     files_to_delete = []
    
#     for file_path in repo_files:
#         # Check if file is in root (no '/' in path) and matches our patterns
#         if '/' not in file_path and file_path.endswith('.json'):
#             if file_path.startswith('chat_history') or file_path.startswith('raw_data'):
#                 files_to_delete.append(file_path)
    
#     if files_to_delete:
#         print(f"   Found {len(files_to_delete)} misplaced JSON files to delete:")
#         print(f"   Deleting all {len(files_to_delete)} files in a single batch operation...")
        
#         # Delete all files in a single batch operation to avoid rate limiting
#         api.delete_files(
#             delete_patterns=files_to_delete,
#             repo_id=REPO_ID,
#             repo_type=REPO_TYPE,
#             commit_message=f"Remove {len(files_to_delete)} misplaced JSON files from root directory"
#         )
#         print("   ✅ Cleanup complete!")
#     else:
#         print("   No misplaced files found.")
# except Exception as e:
#     print(f"   ⚠️ Could not clean up files: {e}")

# # Upload specific folders with proper folder structure preservation using upload_large_folder
# folders_to_upload = [
#     "data/raw_data",
#     "data/chat_history_32k", 
#     "data/chat_history_128k",
#     "data/chat_history_multimodal_32k",
#     "data/chat_history_multimodal_128k",
# ]

# # Create a temporary directory to stage the upload with proper structure
# with tempfile.TemporaryDirectory() as temp_dir:
#     print(f"📁 Creating temporary staging directory: {temp_dir}")
    
#     # Copy each folder to the temp directory, preserving the full path structure
#     for folder_path in folders_to_upload:
#         if os.path.exists(folder_path):
#             temp_folder_path = os.path.join(temp_dir, folder_path)
#             print(f"📋 Staging {folder_path}/ -> {temp_folder_path}/")
            
#             # Create the parent directories in temp
#             os.makedirs(os.path.dirname(temp_folder_path), exist_ok=True)
            
#             # Copy the entire folder structure
#             shutil.copytree(folder_path, temp_folder_path)
#         else:
#             print(f"⚠️ Skipping {folder_path} - folder does not exist")
    
#     # Upload the entire staged structure using upload_large_folder
#     print(f"⬆️ Uploading staged directory with preserved folder structure...")
#     api.upload_large_folder(
#         folder_path=temp_dir,
#         repo_id=REPO_ID,
#         repo_type=REPO_TYPE,
#     )
    
#     print("🗑️ Cleaning up temporary directory...")
    
# Upload benchmark folder structure
benchmark_folders = [
    "benchmark/text",
    "benchmark/multimodal"
]

print("⬆️ Uploading benchmark folder structure...")
for folder_path in benchmark_folders:
    if os.path.exists(folder_path):
        print(f"📁 Uploading folder: {folder_path}/")
        api.upload_folder(
            folder_path=folder_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            path_in_repo=folder_path,
            commit_message=f"Add {folder_path} benchmark files"
        )
    else:
        print(f"⚠️ Skipping {folder_path} - folder does not exist")

print("✅ Upload complete!")
