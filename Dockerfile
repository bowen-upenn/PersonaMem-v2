# Option 1: If you are not using AzureML, you can use this Dockerfile to build the container.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install git first
RUN apt-get update && apt-get install -y git && apt-get install -y tmux && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/volcengine/verl.git

# Create workspace directory for volume mount
RUN mkdir -p /workspace
WORKDIR /workspace

# Copy requirements file first for dependency installation
COPY requirements.txt /tmp/requirements.txt

# Install project requirements
RUN pip install -r /tmp/requirements.txt

RUN cd /app/verl && pip install -e .
RUN pip3 install azureml-mlflow
RUN pip install vllm==0.8.3
RUN pip install "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
RUN pip install torchdata==0.11.0
RUN pip install jupyterlab
RUN pip install wandb
RUN pip install aiolimiter
RUN pip uninstall -y deepspeed
RUN pip install sentence-transformers
RUN pip install scikit-learn
RUN pip install json_repair
RUN pip install timeout_decorator
RUN pip install pytz


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
