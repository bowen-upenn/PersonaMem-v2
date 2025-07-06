FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:33

WORKDIR /app

RUN pip3 install azureml-mlflow
RUN pip install torchdata==0.11.0

RUN pip install --no-cache-dir \
    wandb \
    openai \
    numpy \
    torch \
    transformers \
    tiktoken \
    json_repair \
    datasets \
    timeout_decorator