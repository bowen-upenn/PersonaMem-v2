## 🌟 Redefining Implicit User Preferences in LLM Personalization 🌟

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
      --verbose

📌 Command-line Arguments:

-  ```--model```: Choose your LLM model (default: gpt-4.1).
-  ```--conv_output_path```: Path to store generated conversations.
-  ```--result_path```: Path to store final outputs.
-  ```--num_persona```: Number of personas to generate.
-  ```--data_types```: Conversation types (e.g., email, etc.) for implicit preference detection. We only support ```email``` currently.
-  ```--context_length```: Length of the context in tokens, including irrelevant ones.
-  ```--self_verify```: Enable self-verification to assess whether each user preference aligns with the model's inferred average preference for the current demographic group.
-  ```--verbose```: Enable detailed logging.
