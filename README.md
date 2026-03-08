# This is the official repository of [PersonaMem-v2: Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory](https://arxiv.org/abs/2512.06688)

### 🎉 Our dataset has been downloaded over **12,000 times** on [Hugging Face](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2). Thank you for your support!

[![Paper](https://img.shields.io/badge/arXiv-2512.06688-b31b1b.svg)](https://arxiv.org/abs/2512.06688)
[![Dataset](https://img.shields.io/badge/HuggingFace-PersonaMem--v2-ffd21e.svg)](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2)
[![PersonaMem v1](https://img.shields.io/badge/GitHub-PersonaMem--v1-0075b6.svg?logo=github)](https://github.com/bowen-upenn/PersonaMem)

![Overview](overview.png)

PersonaMem-v2 is the state-of-the-art LLM personalization benchmark, covering 1000 comprehensive user personas and 20,000+ preferences across 300+ scenarios. It especially focuses on realistic cases where user preferences are implicitly revealed through long-context conversations rather than stated explicitly. We simulate realistic long-form chat histories to test how well LLMs and agentic memory systems can infer these signals and deliver  personalized responses over time.

---

## Installation

### Environment

We use Python 3.10 with CUDA 12.6. A [Dockerfile](Dockerfile) is provided:

```bash
# Build
docker build -t persona_mem .

# Run with all GPUs
docker run -it --gpus all -v /path/to/PersonaMem-v2:/workspace persona_mem /bin/bash
```

### API Keys

```bash
cp .env.example .env
```

Then fill in your credentials in [.env](.env) by choosing one of the following:

**OpenAI**
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-chat
OPENAI_MODEL_EMBED=text-embedding-3-large
```

**Microsoft Azure**
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_azure_openai_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-chat
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_DEPLOYMENT_NAME_EMBED=text-embedding-3-large
AZURE_OPENAI_API_VERSION_EMBED=your_embed_api_version
```

### Data

Download the data from [HuggingFace](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2) and place it at `data/`.

---

## Inference and Evaluation

To run inference over the benchmark `data/benchmark/multimodal/benchmark.csv` on frontier LLMs, run

```bash
bash scripts/inference_scripts/run_gpt5_chat.sh
```

Inference scripts for other models are in [scripts/inference_scripts/](scripts/inference_scripts/).

---

## Training

First, download the initial model checkpoint:

```bash
bash verl_custom/scripts/download_model.sh
```

### Train a model using vanilla GRPO over long context with the verl framework:

Our modified training code is kept under [verl_custom/](verl_custom/) separately from the original [verl](https://github.com/verl-project/verl) for clarity.

Prepare the data with the format needed for training:

```bash
python verl_custom/data_preprocess_rft.py
python verl_custom/data_preprocess_sft.py
```

Optionally, cold-start the base model with SFT before GRPO:

```bash
bash verl_custom/scripts/run_qwen3_4b_sft.sh
```

Then run training with GRPO:

```bash
bash verl_custom/scripts/run_qwen3_4b_grpo.sh
```

Run inference with a trained checkpoint:

```bash
bash verl_custom/scripts/run_qwen3_4b_inference.sh
```

See [verl_custom/](verl_custom/) for configuration details and model options.

### Train a model using Agentic Memory with the verl framework:

We adapt the original [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent) pipeline, but shift its focus from long-context knowledge QA to scalable personalization and user memory that grows with each user over time. Our modifications also remove the original dependency on future user queries, which violates causality principles in a realistic personalization setting.

First, prepare the data with the format needed for training:

```bash
python MemAgent/data/data_preprocess.py
```

Then run training with GRPO:

```bash
bash MemAgent/run_qwen3_4b_grpo.sh
```

Run inference with a trained checkpoint:

```bash
bash MemAgent/run_qwen3_4b_inference.sh
```

See [MemAgent/](MemAgent/) for configuration details and model options.

---

## Data Generation

To regenerate the benchmark from scratch, run the full pipeline:

```bash
bash scripts/data_gen_scripts/run_generate_all.sh
```

This runs five sequential steps: image embedding setup, conversation generation, Q&A generation, chat history building (32k and 128k), and benchmark CSV preparation. Individual steps are available as separate scripts in [scripts/data_gen_scripts/](scripts/data_gen_scripts/).

Key parameters in the script:
- `--num_persona`: number of personas to generate (default: 1000)
- `--rate_limit_per_min`: API rate limit
- `--parallel`: enable parallel processing

Run `PYTHONPATH=. python data_generation/main.py --help` for all options.

See [data_generation/](data_generation/) for more configuration details.
