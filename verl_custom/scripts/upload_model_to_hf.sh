#!/bin/bash

# Script to upload model weights to Hugging Face
# Usage: ./upload_model_to_hf.sh [repository_name]
# Example: ./upload_model_to_hf.sh bowen-upenn/Qwen3-4B-Personalization

set -e  # Exit on any error

# Configuration
# MODEL_PATH="checkpoints/implicit_persona_verl/verl_qwen3_4b_grpo_20251001_150607/merged"
# DEFAULT_REPO_NAME="bowen-upenn/Qwen3-4B-Personailization-GRPO"
# MODEL_PATH="checkpoints/implicit_persona_verl/verl_qwen3_8b_grpo_20250928_213541/merged"
# DEFAULT_REPO_NAME="bowen-upenn/Qwen3-4B-Personalization"
MODEL_PATH="verl_custom/ckpt_sft/global_step_400"
DEFAULT_REPO_NAME="bowen-upenn/Qwen3-4B-PersonaMem-SFT"

# Get repository name from command line argument or use default
REPO_NAME=${1:-$DEFAULT_REPO_NAME}

echo "========================================="
echo "Hugging Face Model Upload Script"
echo "========================================="
echo "Model path: $MODEL_PATH"
echo "Repository name: $REPO_NAME"
echo ""

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found at $MODEL_PATH"
    exit 1
fi

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
echo "Step 2: Uploading Model to Hugging Face"
echo "--------------------------------------"

# Create upload script
cat > upload_temp.py << 'EOF'
import os
import sys
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_model(model_path, repo_name):
    """Upload model to Hugging Face Hub"""
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
                exist_ok=True,
                private=False  # Set to True if you want a private repository
            )
            print(f"Repository {full_repo_name} created/verified successfully")
        except Exception as e:
            print(f"Repository creation info: {e}")
        
        # Upload all files in the model directory
        print("Uploading model files...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type="model",
            commit_message="Upload implicit persona model weights"
        )
        
        print(f"✅ Model uploaded successfully!")
        print(f"🌐 View your model at: https://huggingface.co/{full_repo_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        return False

if __name__ == "__main__":
    model_path = sys.argv[1]
    repo_name = sys.argv[2]
    
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist")
        sys.exit(1)
    
    success = upload_model(model_path, repo_name)
    sys.exit(0 if success else 1)
EOF

# Run the upload
python3 upload_temp.py "$MODEL_PATH" "$REPO_NAME"

# Clean up temporary script
rm upload_temp.py

echo ""
echo "========================================="
echo "Upload Complete!"
echo "========================================="
echo "Your model has been uploaded to Hugging Face."
echo "Repository: https://huggingface.co/$REPO_NAME"
echo ""
echo "To use your model:"
echo "from transformers import AutoModel, AutoTokenizer"
echo "model = AutoModel.from_pretrained('$REPO_NAME')"
echo "tokenizer = AutoTokenizer.from_pretrained('$REPO_NAME')"