#!/usr/bin/env python3
"""
Orchestrator for PersonaMem-v2 evaluation via Claude Code subagents.

Commands:
  init   -- Initialize work items and checkpoint from benchmark CSV
  next   -- Return next unprocessed work item as JSON; write prompt to /tmp/eval_prompt_*.txt
  save   -- Record a subagent response file, grade it, update checkpoint + CSV
  status -- Print progress summary
  summary -- Generate accuracy breakdown to evaluation_results_summary.txt
"""

import argparse
import ast
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompt builders (mirroring inference.py logic, no LLM calls)
# ---------------------------------------------------------------------------

def _load_chat_history(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "conversations" in data:
            return data["conversations"]
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, dict) and "conversations" in v:
                    return v["conversations"]
                if isinstance(v, list):
                    return v
        return []
    except Exception as e:
        print(f"Warning: could not load chat history {path}: {e}", file=sys.stderr)
        return []


def _create_mcq_options(correct_answer, incorrect_answers, seed=None):
    rng = random.Random(seed)
    options = [correct_answer] + incorrect_answers
    rng.shuffle(options)
    option_mapping = {}
    parts = []
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        option_mapping[letter] = opt
        parts.append(f"{letter}. {opt}")
    correct_letter = next(l for l, a in option_mapping.items() if a == correct_answer)
    instruction = (
        "Please choose the best answer from the following options:\n\n"
        + "\n".join(parts)
        + "\n\nThink step by step about which answer best fits the user's query and "
        "conversation context. Provide your reasoning first, then give your final "
        "answer as 'Final Answer: [Letter]'"
    )
    return instruction, option_mapping, correct_letter


def build_prompt(row, size, mode, project_root):
    """Build the full prompt text for a single work item."""
    # Parse user_query
    try:
        uq = json.loads(row["user_query"])
    except (json.JSONDecodeError, KeyError):
        try:
            uq = ast.literal_eval(row["user_query"])
        except Exception:
            uq = {"role": "user", "content": str(row.get("user_query", ""))}

    if uq.get("content"):
        uq["content"] += " Please recall my related preferences from our conversation history to give personalized responses."

    # Load chat history
    size_col = f"chat_history_{size}_link"
    ch_path = row.get(size_col) or row.get("chat_history_link", "")
    if ch_path and not os.path.isabs(ch_path):
        ch_path = os.path.join(project_root, ch_path)
    chat_history = _load_chat_history(ch_path) if ch_path else []
    full_history = chat_history + [uq]

    # Build prompt text
    lines = []
    lines.append("You are a helpful AI assistant. Below is a long conversation history between a user and an AI assistant. Read it carefully, then answer the final question.\n")
    lines.append("=" * 60)
    lines.append("CONVERSATION HISTORY:")
    lines.append("=" * 60)
    for msg in full_history[:-1]:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        # Handle multimodal content (list of dicts)
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(text_parts)
        lines.append(f"\n[{role}]: {content}")

    lines.append("\n" + "=" * 60)
    lines.append("QUESTION:")
    lines.append("=" * 60)
    final_msg = full_history[-1]
    final_content = final_msg.get("content", "")
    if isinstance(final_content, list):
        text_parts = [p.get("text", "") for p in final_content if isinstance(p, dict) and p.get("type") == "text"]
        final_content = " ".join(text_parts)
    lines.append(f"\n[USER]: {final_content}")

    if mode == "mcq":
        try:
            incorrect = json.loads(row.get("incorrect_answers", "[]") or "[]")
        except (json.JSONDecodeError, ValueError):
            incorrect = []
        seed = hash(f"{row['persona_id']}_{final_content}") % 2**32
        mcq_instr, option_mapping, correct_letter = _create_mcq_options(
            row["correct_answer"], incorrect, seed=seed
        )
        lines.append("\n" + mcq_instr)
        # Embed grading metadata as a comment at the end (hidden from model but used by save)
        meta = json.dumps({"correct_letter": correct_letter, "correct_answer": row["correct_answer"],
                           "option_mapping": option_mapping})
        lines.append(f"\n<!-- GRADING_META: {meta} -->")
    else:
        lines.append("\nPlease provide a detailed, personalized response based on the user's preferences and history.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _state_path(result_path):
    return Path(result_path) / "state.json"


def _checkpoint_path(result_path):
    return Path(result_path) / "checkpoint.json"


def _results_csv_path(result_path):
    return Path(result_path) / "evaluation_results.csv"


def _load_state(result_path):
    p = _state_path(result_path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _save_state(result_path, state):
    with open(_state_path(result_path), "w") as f:
        json.dump(state, f, indent=2)


def _load_checkpoint(result_path):
    p = _checkpoint_path(result_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _save_checkpoint(result_path, ckpt):
    with open(_checkpoint_path(result_path), "w") as f:
        json.dump(ckpt, f, indent=2)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args):
    result_path = Path(args.result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    # Load benchmark rows
    rows = []
    with open(args.benchmark_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(dict(row))
            if args.max_items and len(rows) >= args.max_items:
                break

    sizes = ["32k", "128k"] if args.size == "both" else [args.size]
    modes = ["mcq", "generative"] if args.eval_mode == "both" else [args.eval_mode]

    # Determine subagent model
    subagent_model = args.subagent_model or _infer_subagent_model(args.model_name)

    work_items = []
    for idx, row in enumerate(rows):
        for size in sizes:
            for mode in modes:
                work_items.append({"row_idx": idx, "size": size, "mode": mode})

    state = {
        "benchmark_file": str(args.benchmark_file),
        "result_path": str(result_path),
        "model_name": args.model_name,
        "subagent_model": subagent_model,
        "sizes": sizes,
        "modes": modes,
        "fieldnames": fieldnames,
        "total_rows": len(rows),
        "work_items": work_items,
        "next_idx": 0,
    }
    _save_state(result_path, state)

    # Persist rows
    rows_path = result_path / "rows.json"
    with open(rows_path, "w") as f:
        json.dump(rows, f)

    print(json.dumps({
        "status": "initialized",
        "total_rows": len(rows),
        "total_work_items": len(work_items),
        "sizes": sizes,
        "modes": modes,
        "subagent_model": subagent_model,
        "result_path": str(result_path),
    }))


def _infer_subagent_model(model_name):
    name = (model_name or "").lower()
    if "opus" in name:
        return "opus"
    if "haiku" in name:
        return "haiku"
    return "sonnet"


def cmd_next(args):
    state = _load_state(args.result_path)
    if state is None:
        print(json.dumps({"error": "Not initialized. Run: python3 eval_orchestrator.py init ..."}))
        return

    ckpt = _load_checkpoint(args.result_path)
    work_items = state["work_items"]

    # Find next unprocessed item
    while state["next_idx"] < len(work_items):
        item = work_items[state["next_idx"]]
        key = f"{item['row_idx']}_{item['size']}_{item['mode']}"
        if key not in ckpt:
            break
        state["next_idx"] += 1

    if state["next_idx"] >= len(work_items):
        print(json.dumps({"done": True, "total": len(work_items)}))
        return

    item = work_items[state["next_idx"]]
    row_idx = item["row_idx"]
    size = item["size"]
    mode = item["mode"]

    # Load row
    rows_path = Path(state["result_path"]) / "rows.json"
    with open(rows_path) as f:
        rows = json.load(f)
    row = rows[row_idx]

    # Determine project root (directory containing the benchmark file)
    project_root = str(Path(state["benchmark_file"]).parent.parent.parent)
    if not os.path.isabs(project_root):
        project_root = os.path.abspath(project_root)

    # Build prompt
    prompt = build_prompt(row, size, mode, project_root)

    # Write prompt to temp file
    prompt_file = f"/tmp/eval_prompt_{row_idx}_{size}_{mode}.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    processed = sum(1 for k in ckpt)
    total = len(work_items)

    result = {
        "done": False,
        "row_idx": row_idx,
        "size": size,
        "mode": mode,
        "persona_id": row.get("persona_id", ""),
        "prompt_file": prompt_file,
        "response_file": f"/tmp/response_{row_idx}_{size}_{mode}.txt",
        "subagent_model": state.get("subagent_model", "sonnet"),
        "progress": f"{processed}/{total}",
        "remaining": total - processed,
    }

    # Advance pointer
    state["next_idx"] += 1
    _save_state(args.result_path, state)

    print(json.dumps(result))


def _extract_final_answer(response):
    patterns = [
        r'\$\\boxed\{([A-Z])\}\$',
        r'\\boxed\{([A-Z])\}',
        r'Final Answer:\s*([A-Z])',
        r'final answer:\s*([A-Z])',
        r'Answer:\s*([A-Z])',
        r'answer:\s*([A-Z])',
        r'final answer is\s*\$?\\boxed\{([A-Z])\}\$?',
        r'final answer is\s*([A-Z])',
        r'the answer is\s*\$?\\boxed\{([A-Z])\}\$?',
        r'the answer is\s*([A-Z])',
        r'\b([A-Z])\.\s*$',
    ]
    for pattern in patterns:
        m = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return ""


def cmd_save(args):
    state = _load_state(args.result_path)
    if state is None:
        print(json.dumps({"error": "Not initialized"}))
        return

    row_idx = args.idx
    size = args.size
    mode = args.mode
    response_file = args.response_file

    if not os.path.exists(response_file):
        print(json.dumps({"error": f"Response file not found: {response_file}"}))
        return

    with open(response_file, encoding="utf-8") as f:
        response = f.read().strip()

    # Load the corresponding prompt to extract grading metadata
    prompt_file = f"/tmp/eval_prompt_{row_idx}_{size}_{mode}.txt"
    grading_meta = {}
    if os.path.exists(prompt_file):
        with open(prompt_file, encoding="utf-8") as f:
            prompt_text = f.read()
        m = re.search(r'<!-- GRADING_META: (.+?) -->', prompt_text)
        if m:
            try:
                grading_meta = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    # Grade
    is_correct_mcq = ""
    predicted_answer = ""
    if mode == "mcq" and grading_meta:
        predicted_answer = _extract_final_answer(response)
        option_mapping = grading_meta.get("option_mapping", {})
        correct_answer = grading_meta.get("correct_answer", "")
        predicted_text = option_mapping.get(predicted_answer.upper(), "")
        is_correct_mcq = str(predicted_text == correct_answer)

    # Update checkpoint
    ckpt = _load_checkpoint(args.result_path)
    key = f"{row_idx}_{size}_{mode}"
    ckpt[key] = {
        "response": response,
        "predicted_answer": predicted_answer,
        "is_correct_mcq": is_correct_mcq,
    }
    _save_checkpoint(args.result_path, ckpt)

    # Check if all work items for this row are complete; if so, write to CSV
    rows_path = Path(state["result_path"]) / "rows.json"
    with open(rows_path) as f:
        rows = json.load(f)
    row = rows[row_idx]

    row_keys = [f"{row_idx}_{s}_{m}" for s in state["sizes"] for m in state["modes"]]
    row_complete = all(k in ckpt for k in row_keys)

    if row_complete:
        _append_row_to_csv(state, row, row_idx, ckpt)
        print(json.dumps({
            "status": "saved",
            "row_complete": True,
            "row_idx": row_idx,
            "is_correct_mcq": is_correct_mcq,
            "predicted_answer": predicted_answer,
        }))
    else:
        remaining_for_row = [k for k in row_keys if k not in ckpt]
        print(json.dumps({
            "status": "saved",
            "row_complete": False,
            "row_idx": row_idx,
            "remaining_for_row": remaining_for_row,
            "is_correct_mcq": is_correct_mcq,
            "predicted_answer": predicted_answer,
        }))


def _append_row_to_csv(state, row, row_idx, ckpt):
    """Append a fully-completed row to the results CSV."""
    results_csv = _results_csv_path(state["result_path"])
    output_row = dict(row)

    for size in state["sizes"]:
        for mode in state["modes"]:
            key = f"{row_idx}_{size}_{mode}"
            entry = ckpt.get(key, {})
            if mode == "mcq":
                output_row[f"model_response_mcq_{size}"] = entry.get("response", "")
                output_row[f"predicted_answer_mcq_{size}"] = entry.get("predicted_answer", "")
                output_row[f"is_correct_mcq_{size}"] = entry.get("is_correct_mcq", "")
            else:
                output_row[f"model_response_openended_{size}"] = entry.get("response", "")
                output_row[f"is_correct_openended_{size}"] = ""

    # Build fieldnames
    base_fields = state["fieldnames"]
    extra_fields = []
    for size in state["sizes"]:
        for mode in state["modes"]:
            if mode == "mcq":
                extra_fields += [
                    f"model_response_mcq_{size}",
                    f"predicted_answer_mcq_{size}",
                    f"is_correct_mcq_{size}",
                ]
            else:
                extra_fields += [
                    f"model_response_openended_{size}",
                    f"is_correct_openended_{size}",
                ]
    all_fields = base_fields + [f for f in extra_fields if f not in base_fields]

    file_exists = results_csv.exists()
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(output_row)


def cmd_status(args):
    state = _load_state(args.result_path)
    if state is None:
        print("Not initialized.")
        return

    ckpt = _load_checkpoint(args.result_path)
    total = len(state["work_items"])
    done = len(ckpt)

    # Count correct MCQ
    correct_by_size = defaultdict(int)
    total_by_size = defaultdict(int)
    for key, entry in ckpt.items():
        parts = key.split("_", 2)  # row_idx_size_mode
        if len(parts) == 3 and parts[2] == "mcq":
            size = parts[1]
            total_by_size[size] += 1
            if entry.get("is_correct_mcq") == "True":
                correct_by_size[size] += 1

    print(f"Progress: {done}/{total} work items complete")
    for size in state.get("sizes", []):
        t = total_by_size.get(size, 0)
        c = correct_by_size.get(size, 0)
        acc = f"{c/t:.3f}" if t > 0 else "N/A"
        print(f"  MCQ accuracy ({size}): {acc} ({c}/{t})")


def cmd_summary(args):
    state = _load_state(args.result_path)
    if state is None:
        print("Not initialized.")
        return

    results_csv = _results_csv_path(args.result_path)
    if not results_csv.exists():
        print("No results CSV found yet.")
        return

    rows = []
    with open(results_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    summary_path = Path(args.result_path) / "evaluation_results_summary.txt"
    sizes = state.get("sizes", ["32k", "128k"])

    categories = [
        "topic_query", "topic_preference", "conversation_scenario",
        "pref_type", "who", "updated", "sensitive_info",
        "distance_from_related_snippet_to_query_32k",
        "distance_from_related_snippet_to_query_128k",
        "num_persona_relevant_tokens_128k",
    ]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total rows: {len(rows)}\n")
        f.write(f"Model: {state.get('model_name', 'unknown')}\n\n")

        for size in sizes:
            col = f"is_correct_mcq_{size}"
            vals = [r[col] for r in rows if r.get(col) in ("True", "False")]
            correct = sum(1 for v in vals if v == "True")
            total = len(vals)
            acc = f"{correct/total:.3f}" if total else "N/A"
            f.write(f"Overall MCQ Accuracy ({size}): {acc} ({correct}/{total})\n")

        for size in sizes:
            f.write(f"\n{'='*60}\nCATEGORY BREAKDOWN FOR {size.upper()}\n{'='*60}\n")
            col = f"is_correct_mcq_{size}"
            for cat in categories:
                stats = defaultdict(lambda: {"correct": 0, "total": 0})
                for row in rows:
                    if row.get(col) not in ("True", "False"):
                        continue
                    val = row.get(cat, "")
                    is_dist = "distance_from_related_snippet_to_query" in cat
                    if is_dist:
                        try:
                            v = int(val)
                            b = (v // 1024) * 1024
                            val = f"{b}-{b+1023}"
                        except (ValueError, TypeError):
                            pass
                    stats[val]["total"] += 1
                    if row[col] == "True":
                        stats[val]["correct"] += 1

                f.write(f"\nACCURACY BY {cat.upper()} ({size}):\n" + "-" * 40 + "\n")
                for k, s in sorted(stats.items()):
                    a = f"{s['correct']/s['total']:.3f}" if s["total"] else "N/A"
                    f.write(f"{k}: {a} ({s['correct']}/{s['total']})\n")

    print(f"Summary written to: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PersonaMem-v2 eval orchestrator")
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init")
    p_init.add_argument("--benchmark_file", required=True)
    p_init.add_argument("--result_path", required=True)
    p_init.add_argument("--eval_mode", default="both", choices=["mcq", "generative", "both"])
    p_init.add_argument("--size", default="both", choices=["32k", "128k", "both"])
    p_init.add_argument("--max_items", type=int, default=None)
    p_init.add_argument("--model_name", default="claude-opus-4-6")
    p_init.add_argument("--subagent_model", default=None,
                        help="opus|sonnet|haiku (auto-derived from model_name if omitted)")

    p_next = sub.add_parser("next")
    p_next.add_argument("--result_path", required=True)

    p_save = sub.add_parser("save")
    p_save.add_argument("--result_path", required=True)
    p_save.add_argument("--idx", type=int, required=True)
    p_save.add_argument("--size", required=True)
    p_save.add_argument("--mode", required=True)
    p_save.add_argument("--response_file", required=True)

    p_status = sub.add_parser("status")
    p_status.add_argument("--result_path", required=True)

    p_summary = sub.add_parser("summary")
    p_summary.add_argument("--result_path", required=True)

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "next":
        cmd_next(args)
    elif args.command == "save":
        cmd_save(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "summary":
        cmd_summary(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
