## 🌟 ImplicitPersona: Continuous Learning of Implicit User Personas via Reinforcement Finetuning for LLM Personalization 🌟

 [![Slides](https://img.shields.io/badge/Google_Slides-Link-yellow)](https://docs.google.com/presentation/d/1ZKQ0SLK2CLvrioqLTs5mRsEm00NHCL32HwBeb_bJmRM/edit?usp=sharing) 

### 📦 Installation

Install the required dependencies:

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### � API Keys Setup

Configure your API credentials in the `.env` file, following examples in `.env.example`.

**Option 1: OpenAI Configuration**
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=o4-mini
```

**Option 2: Microsoft Azure OpenAI Configuration**
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_azure_openai_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=o4-mini
AZURE_OPENAI_API_VERSION=your_azure_openai_api_version
```

### 📁 Data Setup

1. **Download image data**: Follow the instructions in `data/download_phptobook_imges.md` to download the required image dataset.

2. **Persona Hub**: The PersonaHub dataset is already available in `data/Persona_Hub_200000.jsonl`.

3. **Output directory**: Generated persona files will be saved to `data/raw_data/`.

### ⚙️ Usage

Run the following scripts in order to generate personalized conversation data:

#### 1. Prepare Images
Create image embeddings database for persona matching:
```bash
bash scripts/run_prepare_images.sh
```

#### 2. Generate Conversations
Generate user personas, preferences, and convert them into user-chatbot conversations:
```bash
bash scripts/run_convo_gen.sh
```

#### 3. Generate Q&A Pairs
Generate question-answer pairs from the conversations:
```bash
bash scripts/run_qa_gen.sh
```

### 📋 Script Configuration

- **`run_prepare_images.sh`**: Preprocesses images and creates embedding database
- **`run_convo_gen.sh`**: Generates personas and conversations with configurable parameters:
  - `--num_persona`: Number of personas to generate
  - `--data_types`: Conversation types (email, chat, writing, etc.)
  - `--rate_limit_per_min`: API rate limit (default: 10 requests/minute)
  - `--parallel`: Enable parallel processing for faster generation
- **`run_qa_gen.sh`**: Generates Q&A pairs with persona range support:
  - `--persona_start_idx` / `--persona_end_idx`: Process specific persona ranges
  - `--rate_limit_per_min`: API rate limit (default: 10 requests/minute)
  - `--parallel`: Enable parallel processing

For detailed parameter customization, edit the scripts or run `python main.py --help`. 🌟
