#!/bin/bash

# Script to upload CSV dataset files to Hugging Face
# Usage: ./upload_csv_to_hf.sh [repository_name]
# Example: ./upload_csv_to_hf.sh bowen-upenn/ImplicitPersona

set -e  # Exit on any error

# Configuration
DEFAULT_REPO_NAME="bowen-upenn/ImplicitPersona"

# Get repository name from command line argument or use default
REPO_NAME=${1:-$DEFAULT_REPO_NAME}

echo "========================================="
echo "Hugging Face Dataset Upload Script (CSV)"
echo "========================================="
echo "Repository name: $REPO_NAME"
echo ""

# Define source files and their target paths in the repository
declare -A FILE_MAP=(
    ["data/benchmark/multimodal/train.csv"]="benchmark/multimodal/train.csv"
    ["data/benchmark/text/train.csv"]="benchmark/text/train.csv"
    ["data/benchmark/multimodal/benchmark.csv"]="benchmark/multimodal/benchmark.csv"
    ["data/benchmark/text/benchmark.csv"]="benchmark/text/benchmark.csv"
    ["data/benchmark/multimodal/val.csv"]="benchmark/multimodal/val.csv"
    ["data/benchmark/text/val.csv"]="benchmark/text/val.csv"
)

# Check if all files exist
echo "Step 0: Checking if all files exist"
echo "-----------------------------------"
MISSING_FILES=0
for SOURCE_FILE in "${!FILE_MAP[@]}"; do
    if [ ! -f "$SOURCE_FILE" ]; then
        echo "❌ Missing: $SOURCE_FILE"
        MISSING_FILES=$((MISSING_FILES + 1))
    else
        echo "✅ Found: $SOURCE_FILE"
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "Error: $MISSING_FILES file(s) not found. Please ensure all files exist before uploading."
    exit 1
fi

echo ""
echo "Step 1: Hugging Face Authentication"
echo "-----------------------------------"

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable not set. You may need to authenticate manually."
    echo "Please set your Hugging Face token: export HF_TOKEN=your_token_here"
    echo "Or run: huggingface-cli login"
    echo ""
    echo "Attempting to use existing authentication..."
    # Try to use existing token if available
    python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(f'Authenticated as: {api.whoami()[\"name\"]}')" 2>/dev/null || {
        echo "No valid authentication found. Please authenticate first:"
        echo "Run: huggingface-cli login"
        exit 1
    }
else
    echo "Logging in to Hugging Face with provided token..."
    python3 -c "from huggingface_hub import login; login('$HF_TOKEN')"
    echo "Successfully logged in to Hugging Face!"
fi

echo ""
echo "Step 2: Creating temporary directory structure"
echo "----------------------------------------------"

# Create a temporary directory to organize files
TEMP_DIR=$(mktemp -d)
echo "Temporary directory: $TEMP_DIR"

# Copy files to temp directory with proper structure
for SOURCE_FILE in "${!FILE_MAP[@]}"; do
    TARGET_PATH="${FILE_MAP[$SOURCE_FILE]}"
    TARGET_DIR="$TEMP_DIR/$(dirname "$TARGET_PATH")"
    
    echo "Copying $SOURCE_FILE -> $TARGET_PATH"
    mkdir -p "$TARGET_DIR"
    cp "$SOURCE_FILE" "$TEMP_DIR/$TARGET_PATH"
done

echo ""
echo "Step 3: Uploading Dataset to Hugging Face"
echo "----------------------------------------"

# Create upload script
cat > upload_temp.py << 'EOF'
import os
import sys
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_dataset(temp_dir, repo_name):
    """Upload dataset to Hugging Face Hub"""
    try:
        api = HfApi()
        
        # Get current user
        user_info = api.whoami()
        username = user_info["name"]
        full_repo_name = f"{username}/{repo_name}"
        
        print(f"Uploading to repository: {full_repo_name}")
        
        # Create repository (will not fail if it already exists)
        try:
            create_repo(
                repo_id=repo_name,
                repo_type="dataset",
                exist_ok=True,
                private=False  # Set to True if you want a private repository
            )
            print(f"Repository {full_repo_name} created/verified successfully")
        except Exception as e:
            print(f"Repository creation info: {e}")
        
        # Upload all files in the temp directory
        print("Uploading dataset files...")
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Upload ImplicitPersona CSV benchmark files"
        )
        
        print(f"✅ Dataset uploaded successfully!")
        print(f"🌐 View your dataset at: https://huggingface.co/datasets/{full_repo_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        return False

if __name__ == "__main__":
    temp_dir = sys.argv[1]
    repo_name = sys.argv[2]
    
    if not os.path.exists(temp_dir):
        print(f"Error: Temp directory {temp_dir} does not exist")
        sys.exit(1)
    
    success = upload_dataset(temp_dir, repo_name)
    sys.exit(0 if success else 1)
EOF

# Run the upload
python3 upload_temp.py "$TEMP_DIR" "$REPO_NAME"
UPLOAD_STATUS=$?

# Clean up temporary files
echo ""
echo "Step 4: Cleaning up"
echo "------------------"
rm upload_temp.py
rm -rf "$TEMP_DIR"
echo "Temporary files cleaned up"

if [ $UPLOAD_STATUS -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Upload Complete!"
    echo "========================================="
    echo "Your dataset has been uploaded to Hugging Face."
    echo "Repository: https://huggingface.co/datasets/$REPO_NAME"
    echo ""
    echo "Dataset structure:"
    echo "  benchmark/"
    echo "    ├── multimodal/"
    echo "    │   └── train.csv"
    echo "    └── text/"
    echo "        └── train.csv"
    echo ""
    echo "To use your dataset:"
    echo "from datasets import load_dataset"
    echo "dataset = load_dataset('$REPO_NAME', data_files='benchmark/multimodal/train.csv')"
else
    echo ""
    echo "========================================="
    echo "Upload Failed!"
    echo "========================================="
    echo "There was an error uploading the dataset."
    exit 1
fi
