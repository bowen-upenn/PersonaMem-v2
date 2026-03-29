#!/usr/bin/env python3
"""
Post-processing script to score already-generated open-ended responses
using the narrow judge, without re-running the full inference pipeline.

Usage:
    python mem0/scripts/score_openended.py \
        --input results/mem0/gpt-5-chat/evaluation_results_both_32k_03282026_112123.csv \
        --output results/mem0/gpt-5-chat/evaluation_results_scored.csv \
        --config_path config.yaml \
        --parallel 8
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

from query_llm import QueryLLM
from inference_utils import evaluate_narrow_judge


def score_row(row: dict, response_col: str, query_llm_func) -> str:
    response = row.get(response_col, "")
    if not response or response.startswith("ERROR"):
        return ""
    try:
        score, _ = evaluate_narrow_judge(row, response, query_llm_func, None)
        return str(score)
    except Exception as e:
        return f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--config_path", default="config.yaml")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--size", default="32k", help="Context size suffix, e.g. 32k")
    args = parser.parse_args()

    import yaml
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    query_llm = QueryLLM(config)

    response_col = f"model_response_openended_{args.size}"
    score_col = f"is_correct_openended_{args.size}"

    with open(args.input, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {args.input}")
    to_score = [r for r in rows if r.get(response_col, "") and not r.get(score_col, "")]
    print(f"Rows needing OE scoring: {len(to_score)}")

    row_index = {id(r): r for r in rows}
    lock = Lock()
    scored = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(score_row, r, response_col, query_llm.query_llm): r
            for r in to_score
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring OE"):
            row = futures[future]
            result = future.result()
            with lock:
                row[score_col] = result
                scored += 1

    # Ensure output fieldnames include score_col
    out_fields = list(fieldnames)
    if score_col not in out_fields:
        out_fields.append(score_col)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    scores = []
    for r in rows:
        v = r.get(score_col, "")
        try:
            scores.append(float(v))
        except (ValueError, TypeError):
            pass

    print(f"\nResults saved to {args.output}")
    print(f"Scored: {scored} rows")
    if scores:
        print(f"Mean OE judge score: {sum(scores)/len(scores):.3f} ({len(scores)} rows)")


if __name__ == "__main__":
    main()
