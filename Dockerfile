# Option 1: By default, you can use this Dockerfile to build the container.

# Use vLLM latest as base
FROM vllm/vllm-openai:latest
# # Use vLLM 0.13.0 as base — only version with V100 (sm_70) support
# FROM vllm/vllm-openai:v0.13.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    tmux \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install verl from source (pinned to our version)
RUN git clone https://github.com/volcengine/verl.git /workspace/verl \
    && cd /workspace/verl \
    && git checkout bc2cc6b3 \
    && pip install --no-cache-dir -e .

# Install project-specific dependencies
RUN pip install --no-cache-dir \
    liger-kernel \
    tqdm \
    json_repair \
    timeout-decorator \
    pytz \
    scikit-learn \
    sentence-transformers \
    wandb \
    aiolimiter \
    jupyterlab \
    google-generativeai \
    anthropic

# # The following codes are required only for running verl on V100 GPUs

# # Install flash_attn stub (V100 can't compile real flash-attn CUDA kernels,
# # but verl needs flash_attn.bert_padding which is pure PyTorch)
# COPY docker/flash_attn_stub/flash_attn /usr/local/lib/python3.12/dist-packages/flash_attn
# COPY docker/flash_attn_stub/flash_attn-2.7.0.dist-info /usr/local/lib/python3.12/dist-packages/flash_attn-2.7.0.dist-info

# # Install vLLM API compatibility shims (verl bc2cc6b imports vLLM 0.8.x APIs
# # that were moved in vLLM 0.13.0)
# COPY docker/vllm_shims/model_executor/sampling_metadata.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/sampling_metadata.py
# COPY docker/vllm_shims/worker/__init__.py /usr/local/lib/python3.12/dist-packages/vllm/worker/__init__.py
# COPY docker/vllm_shims/worker/worker_base.py /usr/local/lib/python3.12/dist-packages/vllm/worker/worker_base.py
# COPY docker/vllm_shims/lora/models.py /usr/local/lib/python3.12/dist-packages/vllm/lora/models.py
# COPY docker/vllm_shims/lora/request.py /usr/local/lib/python3.12/dist-packages/vllm/lora/request.py
# COPY docker/vllm_shims/lora/worker_manager.py /usr/local/lib/python3.12/dist-packages/vllm/lora/worker_manager.py

# Reset the entrypoint (vllm image sets it to run the server by default)
ENTRYPOINT []

# Default command
CMD ["/bin/bash"]

# Build and run:
# docker build -t persona_mem_v2 .
# docker run -it --gpus all -v /pool/bwjiang/PersonaMem-v2:/app/PersonaMem-v2 persona_mem_v2 bash


################################################################################
# Option 2: If you are using AzureML jobs, you can uncomment the following part of this Dockerfile as your environment's context.
# FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:33

# WORKDIR /app

# RUN git clone https://github.com/volcengine/verl.git
# RUN cd verl && pip install -e .

# RUN pip3 install azureml-mlflow
# RUN pip install vllm==0.8.3
# RUN pip install "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
# RUN pip install torchdata==0.11.0
# RUN pip install jupyterlab
# RUN pip install wandb
# RUN pip install aiolimiter
# RUN pip uninstall -y deepspeed
# RUN pip install sentence-transformers
# RUN pip install scikit-learn
# RUN pip install json_repair
# RUN pip install timeout_decorator
# RUN pip install pytz