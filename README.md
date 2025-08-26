## 🌟 ImplicitPersona: Continuous Learning of Implicit User Personas via Reinforcement Finetuning for LLM Personalization 🌟

##### TODO Human Eval
- Does the conversation sound natural and realistic?
- Can the ground truth user preference be inferred from the conversation?
- Are the user query, correct answer, and incorrect answers well formatted?
- Can the correct answer be inferred from the ground-truth user preference, but not from incorrect options?
  

### 📦 Installation

We use Python 3.10.16 with CUDA version 12.6. We have prepared the [Dockerfile](Dockerfile). You can run

```bash
# Build the image (run in same directory as Dockerfile)
docker build -t implicit_persona .

# Run the container with volume mount for file synchronization
# Option 1: Use all available GPUs
docker run -it --gpus all -v /path/to/your/local/project:/workspace implicit_persona /bin/bash

# Option 2: Use specific GPU(s) by device ID
docker run -it --gpus device=0,1,2,3 -v /path/to/your/local/project:/workspace implicit_persona /bin/bash

# Option 3: Use specific number of GPUs
docker run -it --gpus 4 -v /path/to/your/local/project:/workspace implicit_persona /bin/bash

# Option 4: Use CPU only (if you are going to run data generation and inference only, without running verl or memagent)
docker run -it -v /path/to/your/local/project:/workspace implicit_persona /bin/bash
```


### ⚙️ API Keys Setup

Configure your API credentials in the [.env](.env) file, following examples in [.env.example](.env.example). We use OpenAI's [GPT-5](https://platform.openai.com/docs/models/gpt-5-chat-latest) to generate all data samples in our benchmark. Preparing multimodal samples for the very first time also requires [text-embedding-3-large](https://platform.openai.com/docs/models/text-embedding-3-large). Choose only one of the following two options:

**Option 1: OpenAI Configuration**
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-chat
OPENAI_MODEL_EMBED=text-embedding-3-large
```

**Option 2: Microsoft Azure OpenAI Configuration**
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_azure_openai_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-chat
AZURE_OPENAI_API_VERSION=your_azure_openai_api_version_of_gpt-5-chat
AZURE_OPENAI_DEPLOYMENT_NAME_EMBED=text-embedding-3-large
AZURE_OPENAI_API_VERSION_EMBED=your_azure_openai_api_version_of_text-embedding-3-large
```

### 📁 Data Setup

1. **Download image data**: The [PhotoBook](https://dmg-photobook.github.io/datasets.html) image dataset is already available under [data/photobook_images/](data/photobook_images/).

2. **Persona Hub**: The [PersonaHub](https://github.com/tencent-ailab/persona-hub) dataset is already available in [data/Persona_Hub_200000.jsonl](data/Persona_Hub_200000.jsonl).

3. **Output directory**: Generated persona files will be saved to [data/raw_data/](data/raw_data/).

### 🚀 Usage

Run the following scripts in order to generate personalized conversation data:

#### 1. Prepare Images
Create image embeddings database for persona matching:
```bash
bash scripts/run_prepare_images.sh
```

#### 2. Generate Conversations and Q&A Pairs
Generate user personas, preferences, convert them into user-chatbot conversations, and prepare Q&A pairs with quality validation processes:
```bash
bash scripts/run_generate_all.sh
```

### 📋 Script Configuration

- **`run_prepare_images.sh`**: Preprocesses images and creates embedding database
- **`run_generate_all.sh`**: Generates personas and conversations with configurable parameters:
  - `--num_persona`: Number of personas to generate
  - `--data_types`: User-chatbot conversation types (see the script for all supported types)
  - `--persona_start_idx` / `--persona_end_idx`: Process specific persona ranges
  - `--rate_limit_per_min`: API rate limit (default: 5 personas in parallel)
  - `--parallel`: Enable parallel processing of multiple personas

For detailed parameter customization, edit the scripts or run `python main.py --help`. 🌟
