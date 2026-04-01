#!/usr/bin/env python3
"""
RAG inference script for frontier models (gpt-5-chat, gpt-5-mini).

Reads preprocessed RAG parquet files (produced by verl_custom/rag.py),
sends each prompt to the specified model, and scores responses.

Usage (from project root):
    python rag/inference_rag.py \
        --parquet_file rag/data/benchmark_text_32k_mcq.parquet \
        --model_name gpt-5-chat \
        --result_path rag/results/gpt-5-chat \
        --eval_mode mcq \
        --parallel 4
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from query_llm import QueryLLM
from inference_utils import evaluate_narrow_judge, extract_judge_decision

# Import directly from file to avoid triggering verl_custom/__init__.py
import importlib.util as _ilu
_rag_spec = _ilu.spec_from_file_location(
    "rag_module", Path(__file__).parent.parent / "verl_custom" / "rag.py"
)
_rag_module = _ilu.module_from_spec(_rag_spec)
_rag_spec.loader.exec_module(_rag_module)
load_conversation_context = _rag_module.load_conversation_context
FULL_CONTEXT_INSTRUCTION = _rag_module.FULL_CONTEXT_INSTRUCTION


# ---------------------------------------------------------------------------
# Answer extraction (mirrors inference.py logic, handles both upper/lowercase)
# ---------------------------------------------------------------------------

def extract_final_answer(response: str) -> str:
    """Extract predicted MCQ letter from model response."""
    if not response:
        return ""

    patterns = [
        r'\$\\boxed\{([A-Za-z])\}\$',
        r'\\boxed\{([A-Za-z])\}',
        r'Final Answer:\s*([A-Za-z])',
        r'final answer:\s*([A-Za-z])',
        r'Answer:\s*([A-Za-z])',
        r'answer:\s*([A-Za-z])',
        r'final answer is\s*\$?\\boxed\{([A-Za-z])\}\$?',
        r'final answer is\s*([A-Za-z])',
        r'the answer is\s*\$?\\boxed\{([A-Za-z])\}\$?',
        r'the answer is\s*([A-Za-z])',
        r'\b([A-Za-z])\.\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).lower()

    return ""


def check_mcq_correctness(predicted_letter: str, correct_answer: str, all_answers: List[str]) -> bool:
    """
    Check if predicted letter is correct.

    correct_answer is formatted as "(a) Some answer text" by rag.py.
    all_answers is a list of 4 answer texts in a/b/c/d order.
    """
    if not predicted_letter:
        return False

    # Extract the correct letter from correct_answer, e.g. "(b) ..."
    letter_match = re.match(r'^\(([a-d])\)', correct_answer.strip())
    if letter_match:
        correct_letter = letter_match.group(1).lower()
        return predicted_letter.lower() == correct_letter

    # Fallback: map letter to index and compare text
    try:
        idx = ord(predicted_letter.lower()) - ord('a')
        if 0 <= idx < len(all_answers):
            # Extract expected text from correct_answer
            text_match = re.match(r'^\([a-d]\)\s*(.*)', correct_answer.strip())
            correct_text = text_match.group(1) if text_match else correct_answer
            return all_answers[idx].strip() == correct_text.strip()
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Single-row processor
# ---------------------------------------------------------------------------

def _append_full_context(prompt_messages: List[Dict], chat_history_path: str) -> List[Dict]:
    """Insert full conversation history between RAG excerpts and the final user message."""
    conversations = load_conversation_context(chat_history_path)
    if not conversations:
        return prompt_messages
    # Strip system message from conversations
    conv_messages = [m for m in conversations if isinstance(m, dict) and m.get("role") != "system"]
    # Insert before the last user message
    return (
        prompt_messages[:-1]
        + [{"role": "system", "content": FULL_CONTEXT_INSTRUCTION}]
        + conv_messages
        + [prompt_messages[-1]]
    )


def process_row(
    row: pd.Series,
    query_llm: QueryLLM,
    eval_mode: str,
    judge_llm: Optional[QueryLLM],
    full_context: bool = False,
    path_prefix: str = "",
) -> Dict[str, Any]:
    """Process one parquet row: call LLM, score, return result dict."""
    prompt_messages = json.loads(row["prompt"])
    reward_model = json.loads(row["reward_model"])
    extra_info = json.loads(row["extra_info"])
    ground_truth = reward_model["ground_truth"]

    persona_id = extra_info.get("persona_id", "")
    question = extra_info.get("question", "")
    correct_answer = ground_truth.get("correct_answer", "")
    all_answers = ground_truth.get("all_answers", [])
    pref_type = ground_truth.get("pref_type", "")
    groundtruth_preference = ground_truth.get("groundtruth_preference", "")
    is_mcq = ground_truth.get("is_mcq", False)

    # Optionally append full conversation history to the RAG prompt
    if full_context:
        chat_history_path = row.get("chat_history_link", "")
        if chat_history_path:
            if path_prefix:
                chat_history_path = path_prefix + chat_history_path
            prompt_messages = _append_full_context(prompt_messages, chat_history_path)

    result = {
        "persona_id": persona_id,
        "question": question,
        "correct_answer": correct_answer,
        "pref_type": pref_type,
        "groundtruth_preference": groundtruth_preference,
        "model_response_mcq": "",
        "predicted_answer": "",
        "is_correct": "",
        "model_response_generative": "",
        "judge_score": "",
    }

    # MCQ evaluation
    if eval_mode in ("mcq", "both") and is_mcq:
        try:
            response = query_llm.query_llm(prompt_messages, use_history=True)
            predicted = extract_final_answer(response)
            is_correct = check_mcq_correctness(predicted, correct_answer, all_answers)
            result["model_response_mcq"] = response
            result["predicted_answer"] = predicted
            result["is_correct"] = str(is_correct)
        except Exception as e:
            result["model_response_mcq"] = f"ERROR: {e}"

    # Generative / open-ended evaluation
    if eval_mode in ("generative", "both"):
        # For generative, strip MCQ options from the last user message if present
        gen_messages = _strip_mcq_from_messages(prompt_messages)
        try:
            response_gen = query_llm.query_llm(gen_messages, use_history=True)
            result["model_response_generative"] = response_gen

            if judge_llm is not None:
                judge_row = {
                    "user_query": question,
                    "preference": groundtruth_preference,
                    "pref_type": pref_type,
                }
                try:
                    score = evaluate_narrow_judge(judge_row, response_gen, judge_llm.query_llm, None)
                    result["judge_score"] = str(score)
                except Exception as e:
                    result["judge_score"] = f"ERROR: {e}"
        except Exception as e:
            result["model_response_generative"] = f"ERROR: {e}"

    return result


def _strip_mcq_from_messages(messages: List[Dict]) -> List[Dict]:
    """Return messages with MCQ options block removed from the last user message."""
    if not messages:
        return messages
    msgs = [m.copy() for m in messages]
    last = msgs[-1]
    if last.get("role") == "user" and isinstance(last.get("content"), str):
        # Remove the MCQ block added by rag.py (starts with "\n\nYou are performing")
        content = re.sub(
            r'\n\nYou are performing a multiple-choice question task\..*$',
            '',
            last["content"],
            flags=re.DOTALL,
        )
        # Also strip the thinking instruction appended by rag.py
        content = re.sub(
            r'\s*Always perform your reasoning inside <think> and </think> tags.*$',
            '',
            content,
            flags=re.DOTALL,
        )
        last["content"] = content.strip()
    return msgs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run frontier model inference on preprocessed RAG parquet"
    )
    parser.add_argument(
        "--parquet_file",
        default="rag/data/benchmark_text_32k_mcq.parquet",
        help="Path to RAG parquet file (output of verl_custom/rag.py)",
    )
    parser.add_argument(
        "--model_name",
        default="gpt-5-chat",
        help="Model name: gpt-5-chat | gpt-5-mini",
    )
    parser.add_argument(
        "--result_path",
        default="rag/results",
        help="Output directory for results CSV",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Max rows to process (default: all)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel threads (default: 1)",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["mcq", "generative", "both"],
        default="mcq",
        help="Evaluation mode (default: mcq)",
    )
    parser.add_argument(
        "--full_context",
        action="store_true",
        help="Append full conversation history to RAG prompt at inference time",
    )
    parser.add_argument(
        "--context_size",
        choices=["32k", "128k"],
        default="32k",
        help="Which chat history to append when --full_context is set (default: 32k)",
    )
    parser.add_argument(
        "--path_prefix",
        default="data/",
        help="Prefix prepended to chat history paths from extra_info (default: data/)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAG Inference — Frontier Models")
    print("=" * 60)
    print(f"Parquet file : {args.parquet_file}")
    print(f"Model        : {args.model_name}")
    print(f"Eval mode    : {args.eval_mode}")
    print(f"Full context : {args.full_context} (context_size={args.context_size})")
    print(f"Result path  : {args.result_path}")
    print(f"Parallel     : {args.parallel}")
    print(f"Max items    : {args.max_items or 'all'}")
    print("=" * 60)

    # Resolve which chat history column to use for full-context mode
    chat_history_col = f"chat_history_{args.context_size}_link"

    # Load parquet
    df = pd.read_parquet(args.parquet_file)
    print(f"Loaded {len(df)} rows from {args.parquet_file}")

    if args.max_items is not None and len(df) > args.max_items:
        df = df.head(args.max_items)
        print(f"Capped to {len(df)} rows")

    # Build QueryLLM config (matches config.yaml schema expected by QueryLLM)
    config = {
        "models": {
            "llm_model": args.model_name,
            "max_tokens": 1024,
        }
    }
    query_llm = QueryLLM(config)

    # Judge LLM (reuse same model for open-ended scoring, or None for MCQ-only)
    judge_llm = query_llm if args.eval_mode in ("generative", "both") else None

    # Output file
    result_dir = Path(args.result_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    output_file = result_dir / f"results_{args.eval_mode}_{timestamp}.csv"

    fieldnames = [
        "persona_id", "question", "correct_answer", "pref_type",
        "groundtruth_preference", "model_response_mcq", "predicted_answer",
        "is_correct", "model_response_generative", "judge_score",
    ]

    correct_count = 0
    total_count = 0
    file_lock = threading.Lock()

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        def _process_and_write(row_tuple):
            idx, row = row_tuple
            # Inject the resolved chat_history_link so process_row can find it
            if args.full_context:
                extra = json.loads(row["extra_info"])
                row = row.copy()
                row["chat_history_link"] = extra.get(chat_history_col, "")
            result = process_row(
                row, query_llm, args.eval_mode, judge_llm,
                full_context=args.full_context,
                path_prefix=args.path_prefix,
            )
            with file_lock:
                writer.writerow(result)
                f.flush()
            return result

        rows = list(df.iterrows())

        if args.parallel > 1:
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = {executor.submit(_process_and_write, r): r for r in rows}
                for future in tqdm(as_completed(futures), total=len(rows), desc="Inference"):
                    result = future.result()
                    total_count += 1
                    if result.get("is_correct") == "True":
                        correct_count += 1
        else:
            for row_tuple in tqdm(rows, desc="Inference"):
                result = _process_and_write(row_tuple)
                total_count += 1
                if result.get("is_correct") == "True":
                    correct_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_file}")
    if args.eval_mode in ("mcq", "both") and total_count > 0:
        accuracy = correct_count / total_count
        print(f"MCQ Accuracy: {accuracy:.3f} ({correct_count}/{total_count})")

    # Write summary txt
    summary_file = output_file.parent / f"{output_file.stem}_summary.txt"
    with open(summary_file, "w") as sf:
        sf.write("=" * 60 + "\n")
        sf.write("RAG INFERENCE SUMMARY\n")
        sf.write("=" * 60 + "\n")
        sf.write(f"Model:         {args.model_name}\n")
        sf.write(f"Parquet file:  {args.parquet_file}\n")
        sf.write(f"Eval mode:     {args.eval_mode}\n")
        sf.write(f"Total rows:    {total_count}\n")
        if args.eval_mode in ("mcq", "both"):
            sf.write(f"MCQ Accuracy:  {correct_count / total_count:.3f} ({correct_count}/{total_count})\n")
    print(f"Summary saved to: {summary_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
