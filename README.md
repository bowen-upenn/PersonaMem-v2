## 🌟 ImplicitPersona: Continuous Learning of Implicit User Personas via Reinforcement Finetuning for LLM Personalization 🌟

##### TODO Human Eval
- Does the conversation sound natural and realistic?
- Can the ground truth user preference be inferred from the conversation?
- Are the user query, correct answer, and incorrect answers well formatted?
- Can the correct answer be inferred from the ground-truth user preference, but not from incorrect options?
  

### 📦 Installation

We use Python 3.10.16 with CUDA version 12.6. To install the required dependencies, run

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
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

1. **Download image data**: Follow the instructions in `data/download_phptobook_imges.md` to download the required image dataset.

2. **Persona Hub**: The PersonaHub dataset is already available in `data/Persona_Hub_200000.jsonl`.

3. **Output directory**: Generated persona files will be saved to `data/raw_data/`.

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
