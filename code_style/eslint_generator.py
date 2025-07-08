import os
import sys
import json

# Add parent directory to Python path to import query_llm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from query_llm import QueryLLM

# Number of unique ESLint configs to generate
NUM_CONFIGS = 10  # Change this as needed

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'synthetic_eslint_dataset.json')

# Load previous configs from the current run (in-memory)
def build_prompt(previous_configs):
    prompt = """
You are an expert in JavaScript and ESLint. Please generate a unique, creative, and valid ESLint configuration file (in JSON format, suitable for .eslintrc.json) that has not been generated before. Each config should have a different set of rules, plugins, or stylistic choices. Avoid duplicating any previous config.
"""
    if previous_configs:
        prompt += "\nHere are the previously generated ESLint configs (do not repeat any of these):\n"
        for idx, conf in enumerate(previous_configs):
            prompt += f"Config #{idx+1}:\n```json\n{json.dumps(conf, indent=2)}\n```\n"
    prompt += "\nNow, generate a new, unique ESLint config as a JSON object. Only output the JSON, no explanations."
    return prompt

def main():
    # Minimal args for QueryLLM (you may need to adjust if QueryLLM requires more)
    args = {'models': {'llm_model': 'gpt-4.1-mini-2025-04-14', 'max_tokens': 2048}}
    llm = QueryLLM(args)
    configs = []
    for i in range(NUM_CONFIGS):
        prompt = build_prompt(configs)
        print(f"Querying LLM for ESLint config #{i+1}...")
        response = llm.query_llm(prompt, use_history=False, verbose=True)
        # Try to extract JSON from response
        try:
            config = json.loads(response)
        except Exception:
            import re
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                config_str = match.group(1)
                config = json.loads(config_str)
            else:
                raise ValueError(f"Could not extract JSON from LLM response for config #{i+1}.")
        configs.append(config)
    # Save all configs as a list
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(configs, f, ensure_ascii=False, indent=2)
    print(f"Saved {NUM_CONFIGS} ESLint configs to {OUTPUT_PATH}.")

if __name__ == '__main__':
    main()
