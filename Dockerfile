# FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:33
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN git clone https://github.com/volcengine/verl.git
RUN cd verl && pip install -e .
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