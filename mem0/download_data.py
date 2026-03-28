"""Download PersonaMem-v2 benchmark data from HuggingFace."""
import os
import sys
from huggingface_hub import snapshot_download

dest = os.path.join(os.path.dirname(__file__), "..", "data")
dest = os.path.abspath(dest)
os.makedirs(dest, exist_ok=True)

print(f"Downloading PersonaMem-v2 dataset to {dest} ...")
snapshot_download(
    repo_id="bowen-upenn/PersonaMem-v2",
    repo_type="dataset",
    local_dir=dest,
    ignore_patterns=["*.git*"],
)
print("Download complete.")
benchmark = os.path.join(dest, "benchmark", "multimodal", "benchmark.csv")
print(f"Benchmark file exists: {os.path.exists(benchmark)} → {benchmark}")
