#!/usr/bin/env python3
"""
Preprocess the ImplicitPersona dataset (new benchmark schema) to VERL-readable parquet format.

This version:
- Reads fixed dataset paths:
    Text:
        data/benchmark/text/train.csv
        data/benchmark/text/val.csv
    Multimodal:
        data/benchmark/multimodal/train.csv
        data/benchmark/multimodal/val.csv
- Emits 8 parquet files (split × dtype × window):
    train_text_32k.parquet,    val_text_32k.parquet
    train_text_128k.parquet,   val_text_128k.parquet
    train_multimodal_32k.parquet,  val_multimodal_32k.parquet
    train_multimodal_128k.parquet, val_multimodal_128k.parquet
- Keeps original VERL structure and functionality.
- IMPORTANT: extra_info contains ONLY the active window's fields (32k OR 128k), never both.
"""

# Original path in verl: verl/examples/data_preprocess/gsm8k.py

import argparse
import json
import os
from typing import List, Dict, Any, Optional
import ast

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


# ----------------------------
# General I/O helpers
# ----------------------------

def load_persona_data(csv_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(csv_path)


def load_conversation_context(context_file_path: str) -> List[Dict[str, Any]]:
    """
    Load conversation context from JSON file.

    Supports both:
      - {"messages": [...]}
      - [...] (list of messages directly)
    """
    try:
        with open(context_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'messages' in data:
            return data['messages']
        elif isinstance(data, list):
            return data
        else:
            print(f"Warning: Unexpected JSON structure in {context_file_path}")
            return []

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load context file {context_file_path}: {e}")
        return []


# ----------------------------
# Parsing helpers
# ----------------------------

def safe_eval_list(value: Any) -> List[str]:
    """
    Safely evaluate string representation of list, returning List[str].
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == '':
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    try:
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            else:
                return [str(parsed)]
        else:
            return [str(value)]
    except (ValueError, SyntaxError):
        return [str(value)]


def safe_eval_dict(value: Any) -> Dict[str, Any]:
    """
    Safely evaluate string representation of a dict. Returns {} on failure.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == '':
        return {}
    if isinstance(value, dict):
        return value
    try:
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
    except (ValueError, SyntaxError):
        pass
    return {}


# ----------------------------
# Processing
# ----------------------------

def _select_context_fields(row: pd.Series, window: str) -> Dict[str, Any]:
    """
    Select context-dependent fields for 32k or 128k and return ONLY those.
    Keys returned:
      - context_file_path (str)
      - chat_history_link (str)
      - num_tokens (int)
      - distance (int/float)
      - num_persona_relevant_tokens (int)
      - num_persona_irrelevant_tokens (int)
    """
    window = window.strip().lower()
    if window not in ("32k", "128k"):
        raise ValueError(f"window must be '32k' or '128k', got {window}")

    if window == "32k":
        chat_link = row.get('chat_history_32k_link', '')
        context_file_path = chat_link
        num_tokens = row.get('total_tokens_in_chat_history_32k', 0)
        distance = row.get('distance_from_related_snippet_to_query_32k', 0)
        num_rel = row.get('num_persona_relevant_tokens_32k', 0)
        num_irrel = row.get('num_persona_irrelevant_tokens_32k', 0)
    else:
        chat_link = row.get('chat_history_128k_link', '')
        context_file_path = chat_link
        num_tokens = row.get('total_tokens_in_chat_history_128k', 0)
        distance = row.get('distance_from_related_snippet_to_query_128k', 0)
        num_rel = row.get('num_persona_relevant_tokens_128k', 0)
        num_irrel = row.get('num_persona_irrelevant_tokens_128k', 0)

    def _nz_str(v):
        return '' if (isinstance(v, float) and pd.isna(v)) else v

    return {
        "context_file_path": _nz_str(context_file_path),
        "chat_history_link": _nz_str(chat_link),
        "num_tokens": 0 if pd.isna(num_tokens) else num_tokens,
        "distance": 0 if pd.isna(distance) else distance,
        "num_persona_relevant_tokens": 0 if pd.isna(num_rel) else num_rel,
        "num_persona_irrelevant_tokens": 0 if pd.isna(num_irrel) else num_irrel,
    }


def _extract_question_from_row(row: pd.Series) -> str:
    """
    Build the user question string from new schema.

    'user_query' is typically a JSON-like string with {'role': 'user', 'content': '...'}.
    """
    user_query = row.get('user_query', '')
    question_text = ''

    if isinstance(user_query, dict):
        question_text = str(user_query.get('content', ''))
    elif isinstance(user_query, str):
        uq_dict = safe_eval_dict(user_query)
        if uq_dict:
            question_text = str(uq_dict.get('content', ''))
        else:
            question_text = user_query
    else:
        question_text = str(user_query) if user_query is not None else ''

    question_text = (question_text or '') + \
        " Think step by step using <think> and </think> tokens, then provide your response to the user after </think>."
    return question_text


def convert_to_verl_format(row: pd.Series, idx: int, window: str) -> Dict[str, Any]:
    """
    Convert a single row to VERL-readable format, adapted to new column names.
    window: '32k' or '128k'
    """
    try:
        # Window-specific fields (ONLY the chosen window)
        ctx = _select_context_fields(row, window)
        context_file_path = ctx["context_file_path"]

        # Load conversation context if available
        if context_file_path and os.path.exists(context_file_path):
            conversations = load_conversation_context(context_file_path)
        else:
            if context_file_path:
                print(f"Warning: Context file not found for row {idx}: {context_file_path}")
            conversations = []

        # Build user question (with appended <think> instruction)
        question = _extract_question_from_row(row)

        # Compose messages: system + conversation + user
        messages = []
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant that provides personalized responses based on the user's preferences in conversation history."
        })
        if conversations:
            messages.extend(conversations)
        messages.append({
            "role": "user",
            "content": question
        })

        # Answers and preferences
        correct_answer = row.get('correct_answer', '')
        incorrect_answers = safe_eval_list(row.get('incorrect_answers', '[]'))
        all_answers = []
        if correct_answer != '':
            all_answers.append(str(correct_answer))
        all_answers.extend(str(x) for x in incorrect_answers)

        groundtruth_preference = row.get('preference', row.get('groundtruth_preference', ''))

        # Persona parsing (expanded_persona preferred)
        persona_data = safe_eval_dict(row.get('expanded_persona', '')) or safe_eval_dict(row.get('short_persona', '')) or {}

        # Build VERL record
        verl_data = {
            "data_source": "implicit_persona",
            "prompt": messages,
            "ability": "personalization",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "groundtruth_preference": groundtruth_preference,
                    "correct_answer": correct_answer
                },
            },
            "extra_info": {
                # Base identifiers and Q/A
                "index": idx,
                "persona_id": row.get('persona_id', ''),
                "question": question,
                "correct_answer": correct_answer,
                "all_answers": all_answers,
                "groundtruth_preference": groundtruth_preference,

                # Mapped metadata
                "preference_type": row.get('pref_type', ''),
                "conversation_scenario": row.get('conversation_scenario', ''),
                "preference_topic": row.get('topic_preference', ''),
                "query_topic": row.get('topic_query', ''),
                "who": row.get('who', ''),
                "updated": row.get('updated', ''),
                "previous_preference": row.get('prev_pref', ''),
                "persona": persona_data,

                # Per-window ONLY (as requested)
                "context_window": window,
                "context_file_path": ctx["context_file_path"],
                "chat_history_link": ctx["chat_history_link"],
                "num_tokens": ctx["num_tokens"],
                "distance_from_query_to_target_conversation": ctx["distance"],
                "num_persona_relevant_tokens": ctx["num_persona_relevant_tokens"],
                "num_persona_irrelevant_tokens": ctx["num_persona_irrelevant_tokens"],

                # Keep general pass-through (window-agnostic)
                "raw_persona_file": row.get('raw_persona_file', ''),
                "related_conversation_snippet": row.get('related_conversation_snippet', ''),
                "sensitive_info": row.get('sensitive_info', ''),
            },
        }

        return verl_data

    except Exception as e:
        print(f"Error converting row {idx} to VERL format (window={window}): {e}")
        return None


def process_persona_dataset(csv_path: str, window: str) -> pd.DataFrame:
    """
    Process the dataset at csv_path and return a DataFrame for the requested window.
    window: '32k' or '128k'
    """
    print(f"Loading data from {csv_path}...")
    raw_data = load_persona_data(csv_path)
    print(f"Loaded {len(raw_data)} instances")

    print(f"Converting to VERL format for window={window}...")
    verl_data = []
    for idx, row in raw_data.iterrows():
        verl_instance = convert_to_verl_format(row, idx, window)
        if verl_instance:
            verl_data.append(verl_instance)

    print(f"Successfully converted {len(verl_data)} instances to VERL format (window={window})")
    dataset = pd.DataFrame(verl_data)
    return dataset


# ----------------------------
# Main CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess ImplicitPersona dataset to VERL format (new benchmark schema), generating parquet files for text/multimodal × train/val × 32k/128k. extra_info contains ONLY the active window's info.")
    # Fixed default paths per your request
    parser.add_argument(
        "--text_train_csv",
        default="data/benchmark/text/train.csv",
        help="Path to the TEXT train CSV file."
    )
    parser.add_argument(
        "--text_val_csv",
        default="data/benchmark/text/val.csv",
        help="Path to the TEXT val CSV file."
    )
    parser.add_argument(
        "--multimodal_train_csv",
        default="data/benchmark/multimodal/train.csv",
        help="Path to the MULTIMODAL train CSV file."
    )
    parser.add_argument(
        "--multimodal_val_csv",
        default="data/benchmark/multimodal/val.csv",
        help="Path to the MULTIMODAL val CSV file."
    )
    parser.add_argument(
        "--local_dir",
        default="verl_custom/data/implicit_persona",
        help="Local directory to save processed data"
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="HDFS directory to copy processed data (optional)"
    )

    args = parser.parse_args()

    # Prepare output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    def maybe_process(csv_path: Optional[str], dtype: str, split: str, window: str):
        """
        Process a single (csv_path, dtype, split, window) combination and save parquet.
        """
        if not csv_path or not os.path.exists(csv_path):
            print(f"Skipping {dtype} {split} ({window}): CSV not found at {csv_path}")
            return

        df = process_persona_dataset(csv_path, window)
        out_path = os.path.join(local_dir, f"{split}_{dtype}_{window}.parquet")
        print(f"Saving {dtype} {split} ({window}) dataset to {out_path}...")
        df.to_parquet(out_path)

        # Quick stats/sample
        print(f"{dtype.upper()} {split.upper()} ({window}) -> {len(df)} instances")
        if len(df) > 0:
            sample_row = df.iloc[0]
            print(f"  Data source: {sample_row.get('data_source', 'N/A')}")
            print(f"  Ability: {sample_row.get('ability', 'N/A')}")
            print(f"  Sample prompt length: {len(str(sample_row.get('prompt', '')))} characters")
            ei = sample_row.get('extra_info', {})
            if isinstance(ei, dict):
                print(f"  Sample persona_id: {ei.get('persona_id', 'N/A')}")
                print(f"  Sample preference_type: {ei.get('preference_type', 'N/A')}")
                print(f"  Sample context_window: {ei.get('context_window', 'N/A')}")
                print(f"  Sample num_tokens: {ei.get('num_tokens', 'N/A')}")
                print(f"  Sample distance: {ei.get('distance_from_query_to_target_conversation', 'N/A')}")
                print(f"  Sample question: {ei.get('question', 'N/A')}")
                print(f"  Sample correct_answer: {ei.get('correct_answer', 'N/A')}")

    # TEXT: train/val × 32k/128k
    print("\n=== Processing TEXT splits ===")
    for split, csv_path in (("train", args.text_train_csv), ("val", args.text_val_csv)):
        for window in ("32k", "128k"):
            maybe_process(csv_path, dtype="text", split=split, window=window)

    # MULTIMODAL: train/val × 32k/128k
    print("\n=== Processing MULTIMODAL splits ===")
    for split, csv_path in (("train", args.multimodal_train_csv), ("val", args.multimodal_val_csv)):
        for window in ("32k", "128k"):
            maybe_process(csv_path, dtype="multimodal", split=split, window=window)

    # Copy to HDFS if specified (copy the entire output directory with all outputs)
    if args.hdfs_dir is not None:
        print(f"\nCopying all outputs to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

    print("\nAll requested parquet files generated successfully.")


if __name__ == "__main__":
    main()
