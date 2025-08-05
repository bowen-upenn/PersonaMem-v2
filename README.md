## 🌟 ImplicitPersona: Continuous Learning of Implicit User Personas via Reinforcement Finetuning for LLM Personalization 🌟

 [![Slides](https://img.shields.io/badge/Google_Slides-Link-yellow)](https://docs.google.com/presentation/d/1ZKQ0SLK2CLvrioqLTs5mRsEm00NHCL32HwBeb_bJmRM/edit?usp=sharing) 

### TODOs
##### TODO 1
Edit the code in [contexts_builder.py](contexts_builder.py)

Given each JSON file, i.e., one persona and its conversations, concatenate all its conversations included in content as a single list of dictionaries.
Rules:
    - Randomly shuffle conversation blocks with one exception.
    - If a conversation block has the key "prev_pref", it must appear after its prev_pref block.
    - You can use "prev_pref" to locate the position of the previous block.

Count the final number of tokens in the concatenated list, using only the content of each message.

Save the final list of dictionaries to a JSON file named "context_{timestamp}_{persona_id}.json" in the "data/contexts" directory.
Follow the same timestamp and persona_id in the filename of the input JSON file.

##### TODO 2
Add a new code file named [prepare_benchmark.py](prepare_benchmark.py)

Reformat data into a CSV table.
Each row of the table is one QA pair, mixed from all personas we have.
We can prepare different versions of the table, such as 32k and 128k ones.

We will have the following columns (we already have all the meta-data in our JSON files)
- persona_id
- question
- correct_answer
- all_answers (a list of all possible answers, including the correct one)
- groundtruth_preference
- preference_type (e.g., "stereotypical_pref", "anti_stereotypical_pref", "ask_to_forget", "sensitive_info", etc.)
- conversation_scenario (e.g., "personal_email", "multimodal", etc.)
- preference_topic
- query_topic
- who
- updated
- previous_preference (if any)
- persona (everything in the JSON file before the conversations)
- context_file_path
- num_tokens
- distance_from_query_to_target_conversation

##### TODO 3
Add a new code file named [inference.py](inference.py)
Inference script that runs Azure OpenAI models on this CSV table. 
For each model, the script should iterate over all questions, in parallel processing on batches, and save output responses and options to a JSON file.

### 📦 Installation

Install the required dependencies:

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### ⚙️ API Keys Setup

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

### 🚀 Usage

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
