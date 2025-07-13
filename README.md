## 🌟 Unspoken Selves: Continuous Learning of Implicit User Personas via Reinforcement Finetuning for LLM Personalization 🌟

 [![Slides](https://img.shields.io/badge/Google_Slides-Link-yellow)](https://docs.google.com/presentation/d/1ZKQ0SLK2CLvrioqLTs5mRsEm00NHCL32HwBeb_bJmRM/edit?usp=sharing) 

### 📦 Installating Requirements

    python -m venv myenv
    source myenv/bin/activate
    pip install -r requirements.txt

### ⚙️ Usage

Run ```main.py``` with customizable parameters to generate personalized data:

    python main.py \
      --model gpt-4.1 \
      --conv_output_path data/interactions.jsonl \
      --result_path results/ \
      --num_persona 1 \
      --data_types email \
      --context_length 32000 \
      --self_verify \
      --clean \
      --verbose

📌 Command-line Arguments:

-  ```--model```: Choose your LLM model (default: gpt-4.1).
-  ```--conv_output_path```: Path to store generated conversations.
-  ```--result_path```: Path to store final outputs.
-  ```--num_persona```: Number of personas to generate.
-  ```--data_types```: Conversation types (e.g., email, etc.) for implicit preference detection. We only support ```email``` currently.
-  ```--context_length```: Length of the context in tokens, including irrelevant ones.
-  ```--self_verify```: Enable self-verification to assess whether each user preference aligns with the model's inferred average preference for the current demographic group.
-  ```--clean```: Remove exisitng data and start from scratch.
-  ```--verbose```: Enable detailed logging.
