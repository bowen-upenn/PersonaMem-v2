#!/usr/bin/env python3
"""
Mem0-based evaluation for the PersonaMem-v2 benchmark.

Instead of passing the full raw chat history as context, this script:
  1. Ingests the chat history into a Mem0 memory store (per persona_id)
  2. Retrieves the top-k most relevant memories for each query
  3. Builds a concise prompt from those memories and queries the model

Run from the mem0/ directory:
    python inference_mem0.py --model_name gpt-5-mini --max_items 5 --eval_mode mcq --size 32k

Or use the provided shell scripts in mem0/scripts/.
"""

import sys
import os

# Allow imports from project root (query_llm, inference_utils, utils)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
import argparse
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml
from collections import defaultdict
import time
from tqdm import tqdm
from datetime import datetime
import re
import random
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from query_llm import QueryLLM
from inference_utils import evaluate_narrow_judge, evaluate_broad_judge
from mem0_memory import PersonaMemory


class Mem0BenchmarkEvaluator:
    def __init__(
        self,
        config_path: str = "../config.yaml",
        model_name: str = None,
        result_path: str = "../results/mem0/",
        verbose: bool = False,
        mem0_top_k: int = 10,
        mem0_llm: str = None,
        mem0_embedding: str = None,
        mem0_use_all: bool = False,
        full_context: bool = False,
    ):
        self.config = self._load_config(config_path)
        self.verbose = verbose
        self.mem0_top_k = mem0_top_k
        self.mem0_use_all = mem0_use_all
        self.full_context = full_context
        self.mem0_llm = mem0_llm      # Azure deployment name for Mem0 extraction LLM
        self.mem0_embedding = mem0_embedding  # Azure deployment name for Mem0 embedder

        if model_name:
            self.config["models"]["llm_model"] = self._map_model_name(model_name)

        self.query_llm = QueryLLM(self.config)
        self.results_dir = Path(result_path)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Chat history cache (path → conversations)
        self.chat_history_cache = {}
        self.file_lock = Lock()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _map_model_name(self, model_name: str) -> str:
        model_mapping = {
            "gpt-4o": "gpt-4o-0806",
            "gemini-pro": "gemini-2.5-pro",
            "gemini-flash": "gemini-2.5-flash",
            "claude-sonnet": "claude-3-5-sonnet-20241022",
            "claude-haiku": "claude-3-5-haiku-20241022",
        }
        return model_mapping.get(model_name, model_name)

    def load_chat_history(self, chat_history_path: str, size: str = "32k", use_cache: bool = True) -> List[Dict]:
        if use_cache and chat_history_path in self.chat_history_cache:
            return self.chat_history_cache[chat_history_path]

        try:
            with open(chat_history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    conversations = data
                elif isinstance(data, dict) and "conversations" in data:
                    conversations = data["conversations"]
                elif isinstance(data, dict):
                    conversations = []
                    for key, value in data.items():
                        if isinstance(value, dict) and "conversations" in value:
                            conversations = value["conversations"]
                            break
                        elif isinstance(value, list):
                            conversations = value
                            break
                else:
                    conversations = []

                if chat_history_path not in self.chat_history_cache:
                    if len(self.chat_history_cache) >= 2:
                        self.chat_history_cache.clear()
                    self.chat_history_cache[chat_history_path] = conversations

                return conversations
        except Exception as e:
            print(f"Error loading chat history from {chat_history_path}: {e}")
            return []

    def create_mcq_options(
        self, correct_answer: str, incorrect_answers: List[str], seed: int = None
    ) -> Tuple[str, Dict[str, str]]:
        if seed is not None:
            random.seed(seed)

        options = [correct_answer] + incorrect_answers
        random.shuffle(options)

        option_mapping = {}
        option_parts = []
        for i, option in enumerate(options):
            letter = chr(65 + i)
            option_mapping[letter] = option
            option_parts.append(f"{letter}. {option}")

        mcq_instruction = (
            "Please choose the best answer from the following options:\n\n"
            + "\n".join(option_parts)
            + "\n\nThink step by step about which answer best fits the user's query and conversation context. "
            "Provide your reasoning first, then give your final answer as 'Final Answer: [Letter]'"
        )
        return mcq_instruction, option_mapping

    def _build_mem0_prompt(self, retrieved_memories: str, user_query_content: str,
                           chat_history: List[Dict] = None) -> List[Dict]:
        """
        Build a message list for the model using Mem0-retrieved memories,
        optionally augmented with the full conversation history.
        """
        if retrieved_memories:
            context_block = (
                "The following memories have been extracted from the user's conversation history:\n\n"
                + retrieved_memories
            )
        else:
            context_block = "No relevant memories were found in the user's conversation history."

        system_content = (
            "You are a personalized AI assistant. You have access to relevant memories "
            "extracted from the user's past conversations. Use these memories to provide "
            "personalized, context-aware responses."
        )

        if self.full_context and chat_history:
            # Include full conversation history after the memories
            system_content += (
                "\n\nIn addition to the extracted memories above, the user's full conversation "
                "history is provided below for additional context."
            )
            # Build messages: system + memories block + conversation history + user query
            conv_messages = []
            for msg in chat_history:
                content = msg.get("content", "")
                if isinstance(content, list):
                    text = " ".join(
                        part.get("text", "") for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ).strip()
                    if text:
                        conv_messages.append({"role": msg["role"], "content": text})
                elif isinstance(content, str) and content.strip():
                    conv_messages.append(msg)

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": context_block},
            ] + conv_messages + [
                {
                    "role": "user",
                    "content": (
                        f"User's current request: {user_query_content}\n\n"
                        "Please recall my related preferences from our conversation history to give personalized responses."
                    ),
                },
            ]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": (
                        f"{context_block}\n\n"
                        f"User's current request: {user_query_content}\n\n"
                        "Please recall my related preferences from our conversation history to give personalized responses."
                    ),
                },
            ]
        return messages

    def evaluate_row(
        self,
        row: Dict[str, Any],
        eval_mode: str = "mcq",
        size: str = "32k",
        persona_memory: PersonaMemory = None,
    ) -> Dict[str, Any]:
        """Evaluate a single benchmark row using Mem0 memory retrieval."""
        # Parse user query
        try:
            user_query_dict = json.loads(row["user_query"])
        except json.JSONDecodeError:
            try:
                user_query_dict = ast.literal_eval(row["user_query"])
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing user_query for persona {row['persona_id']}: {e}")
                user_query_dict = {"role": "user", "content": str(row["user_query"]).strip('"').strip("'")}

        user_query_content = user_query_dict.get("content", "")

        # Determine chat history path
        size_column = f"chat_history_{size}_link"
        if size_column in row:
            chat_history_path = row[size_column]
        elif "chat_history_link" in row:
            chat_history_path = row["chat_history_link"]
        else:
            available_columns = list(row.keys())
            raise KeyError(f"Missing '{size_column}'. Available: {available_columns}")

        # Load chat history
        chat_history = self.load_chat_history(chat_history_path, size)

        # Build Mem0 memory from history, retrieve relevant memories, then reset
        persona_id = str(row["persona_id"])
        try:
            persona_memory.build_from_history(chat_history, user_id=persona_id)
            if self.mem0_use_all:
                retrieved = persona_memory.get_all(user_id=persona_id)
            else:
                retrieved = persona_memory.retrieve(user_query_content, user_id=persona_id, top_k=self.mem0_top_k)
        finally:
            persona_memory.reset(persona_id)

        if self.verbose:
            print(f"  [Mem0] Retrieved {len(retrieved.splitlines())} memories for persona {persona_id}")
            if retrieved:
                print(f"  [Mem0] Memories:\n{retrieved[:500]}...")

        # Build base messages from Mem0 memories (optionally with full context)
        base_messages = self._build_mem0_prompt(
            retrieved, user_query_content,
            chat_history=chat_history if self.full_context else None
        )

        row_seed = hash(f"{row['persona_id']}_{user_query_content}") % 2**32

        result = {
            "model_response_mcq": "",
            "predicted_answer_mcq": "",
            "is_correct_mcq": "",
            "retrieved_memories": retrieved,
            "model_response_openended": "",
            "is_correct_openended": "",
        }

        if eval_mode in ["mcq", "both"]:
            try:
                incorrect_answers = json.loads(row["incorrect_answers"]) if row["incorrect_answers"] else []
            except json.JSONDecodeError:
                incorrect_answers = []

            mcq_instruction, option_mapping = self.create_mcq_options(
                row["correct_answer"], incorrect_answers, seed=row_seed
            )

            correct_mcq_option = "N/A"
            for letter, answer in option_mapping.items():
                if answer == row["correct_answer"]:
                    correct_mcq_option = letter
                    break

            messages_mcq = base_messages + [{"role": "system", "content": mcq_instruction}]
            response_mcq = self.query_llm.query_llm(messages_mcq, use_history=True)

            final_answer = self.extract_final_answer(response_mcq)
            is_correct = self.check_mcq_correctness(final_answer, row["correct_answer"], option_mapping)

            result["model_response_mcq"] = response_mcq
            result["predicted_answer_mcq"] = final_answer
            result["is_correct_mcq"] = str(is_correct)
            result["correct_mcq_option"] = correct_mcq_option

        if eval_mode in ["generative", "both"]:
            response_openended = self.query_llm.query_llm(base_messages, use_history=True)
            result["model_response_openended"] = response_openended
            try:
                score, _ = evaluate_narrow_judge(
                    row, response_openended, self.query_llm.query_llm, None
                )
                result["is_correct_openended"] = str(score)
            except Exception as e:
                result["is_correct_openended"] = f"ERROR: {e}"

        return result

    def extract_final_answer(self, response: str) -> str:
        if not response:
            return ""
        patterns = [
            r"\$\\boxed\{([A-Z])\}\$",
            r"\\boxed\{([A-Z])\}",
            r"Final Answer:\s*\*{0,2}([A-Z])\*{0,2}",
            r"final answer:\s*\*{0,2}([A-Z])\*{0,2}",
            r"Answer:\s*\*{0,2}([A-Z])\*{0,2}",
            r"answer:\s*\*{0,2}([A-Z])\*{0,2}",
            r"final answer is\s*\$?\\boxed\{([A-Z])\}\$?",
            r"final answer is\s*\*{0,2}([A-Z])\*{0,2}",
            r"the answer is\s*\$?\\boxed\{([A-Z])\}\$?",
            r"the answer is\s*\*{0,2}([A-Z])\*{0,2}",
            r"\b([A-Z])\.\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        return ""

    def check_mcq_correctness(
        self, predicted_answer: str, correct_answer: str, option_mapping: Dict[str, str]
    ) -> bool:
        if not predicted_answer or not option_mapping:
            return False
        predicted_text = option_mapping.get(predicted_answer.upper(), "")
        return predicted_text == correct_answer

    def _process_single_row(
        self,
        row: Dict[str, Any],
        row_index: int,
        eval_mode: str,
        sizes_to_evaluate: List[str],
        fieldnames: List[str],
    ) -> Dict[str, Any]:
        """Process a single benchmark row (thread-safe: creates its own PersonaMemory)."""
        # Each worker gets its own PersonaMemory instance (not thread-safe to share)
        persona_memory = PersonaMemory(
            llm_deployment=self.mem0_llm,
            embedding_deployment=self.mem0_embedding,
        )

        try:
            output_row = row.copy()
            for eval_size in sizes_to_evaluate:
                output_row[f"model_response_mcq_{eval_size}"] = ""
                output_row[f"predicted_answer_mcq_{eval_size}"] = ""
                output_row[f"is_correct_mcq_{eval_size}"] = ""
                output_row[f"retrieved_memories_{eval_size}"] = ""
                output_row[f"model_response_openended_{eval_size}"] = ""
                output_row[f"is_correct_openended_{eval_size}"] = ""

            all_results = {}
            for eval_size in sizes_to_evaluate:
                result = self.evaluate_row(row, eval_mode, eval_size, persona_memory)
                all_results[eval_size] = result

                output_row[f"model_response_mcq_{eval_size}"] = result.get("model_response_mcq", "")
                output_row[f"predicted_answer_mcq_{eval_size}"] = result.get("predicted_answer_mcq", "")
                output_row[f"is_correct_mcq_{eval_size}"] = result.get("is_correct_mcq", "")
                output_row[f"retrieved_memories_{eval_size}"] = result.get("retrieved_memories", "")
                output_row[f"model_response_openended_{eval_size}"] = result.get("model_response_openended", "")
                output_row[f"is_correct_openended_{eval_size}"] = result.get("is_correct_openended", "")

            if self.verbose:
                BLUE = "\033[94m"
                RESET = "\033[0m"
                print(f"  Verbose output for row {row_index + 1}:")
                print(f"    {BLUE}user_query{RESET}: {row.get('user_query', 'N/A')}")
                print(f"    {BLUE}correct_answer{RESET}: {row.get('correct_answer', 'N/A')}")
                for eval_size in sizes_to_evaluate:
                    result = all_results[eval_size]
                    print(f"    --- {eval_size.upper()} Results ---")
                    print(f"    {BLUE}correct_mcq_option_{eval_size}{RESET}: {result.get('correct_mcq_option', 'N/A')}")
                    print(f"    {BLUE}is_correct_mcq_{eval_size}{RESET}: {output_row[f'is_correct_mcq_{eval_size}']}")
                print("-" * 50)

            return {"success": True, "output_row": output_row, "all_results": all_results, "row_index": row_index}

        except Exception as e:
            print(f"Error processing row {row_index + 1}: {e}")
            output_row = row.copy()
            for eval_size in sizes_to_evaluate:
                output_row[f"model_response_mcq_{eval_size}"] = f"ERROR: {str(e)}"
                output_row[f"predicted_answer_mcq_{eval_size}"] = ""
                output_row[f"is_correct_mcq_{eval_size}"] = ""
                output_row[f"retrieved_memories_{eval_size}"] = ""
                output_row[f"model_response_openended_{eval_size}"] = ""
                output_row[f"is_correct_openended_{eval_size}"] = ""
            return {"success": False, "output_row": output_row, "all_results": {}, "row_index": row_index}

        finally:
            persona_memory.cleanup()

    def run_evaluation(
        self,
        benchmark_file: str,
        eval_mode: str = "mcq",
        max_items: int = None,
        size: str = "32k",
        parallel: int = 1,
    ) -> str:
        print(f"Starting Mem0 evaluation...")
        print(f"Benchmark file: {benchmark_file}")
        print(f"Evaluation mode: {eval_mode}")
        print(f"Size: {size}")
        print(f"Mem0 top_k: {self.mem0_top_k}")
        print(f"Parallel threads: {parallel}")

        rows = []
        with open(benchmark_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)
                if max_items and len(rows) >= max_items:
                    break

        print(f"Loaded {len(rows)} rows from benchmark")

        if size == "both":
            sizes_to_evaluate = ["32k", "128k"]
        else:
            sizes_to_evaluate = [size]

        run_timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
        size_suffix = "" if size == "both" else f"_{size}"
        output_file = self.results_dir / f"evaluation_results_{eval_mode}{size_suffix}_{run_timestamp}.csv"

        output_fieldnames = list(fieldnames)
        for eval_size in sizes_to_evaluate:
            output_fieldnames.extend([
                f"model_response_mcq_{eval_size}",
                f"predicted_answer_mcq_{eval_size}",
                f"is_correct_mcq_{eval_size}",
                f"retrieved_memories_{eval_size}",
                f"model_response_openended_{eval_size}",
                f"is_correct_openended_{eval_size}",
            ])

        processed_count = 0
        correct_count = 0

        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()

            if parallel > 1:
                print(f"Using parallel processing with {parallel} threads")
                with ThreadPoolExecutor(max_workers=parallel) as executor:
                    future_to_row = {
                        executor.submit(
                            self._process_single_row,
                            row, i, eval_mode, sizes_to_evaluate, output_fieldnames,
                        ): i
                        for i, row in enumerate(rows)
                    }
                    for future in tqdm(as_completed(future_to_row), total=len(rows), desc="Processing rows"):
                        try:
                            result_data = future.result()
                            if result_data["success"] and result_data["all_results"]:
                                first_size = sizes_to_evaluate[0]
                                if result_data["all_results"].get(first_size, {}).get("is_correct_mcq") == "True":
                                    correct_count += 1
                            processed_count += 1
                            with self.file_lock:
                                writer.writerow(result_data["output_row"])
                                f.flush()
                        except Exception as e:
                            print(f"Error in parallel processing: {e}")
            else:
                for i, row in enumerate(tqdm(rows, desc="Processing rows")):
                    result_data = self._process_single_row(
                        row, i, eval_mode, sizes_to_evaluate, output_fieldnames
                    )
                    if result_data["success"] and result_data["all_results"]:
                        first_size = sizes_to_evaluate[0]
                        if result_data["all_results"].get(first_size, {}).get("is_correct_mcq") == "True":
                            correct_count += 1
                    processed_count += 1
                    writer.writerow(result_data["output_row"])
                    f.flush()

        print(f"\nResults saved to {output_file}")

        if eval_mode in ["mcq", "both"] and processed_count > 0:
            accuracy = correct_count / processed_count
            print(f"\n{'='*50}")
            print(f"EVALUATION STATISTICS (Mem0)")
            print(f"{'='*50}")
            print(f"Total processed: {processed_count}")
            print(f"Overall MCQ Accuracy: {accuracy:.3f} ({correct_count}/{processed_count})")

        return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Mem0-based PersonaMem-v2 benchmark evaluation")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g. gpt-5-mini)")
    parser.add_argument("--benchmark_file", type=str, default="../data/benchmark/multimodal/benchmark.csv")
    parser.add_argument("--config_path", type=str, default="../config.yaml")
    parser.add_argument("--eval_mode", type=str, default="mcq", choices=["mcq", "generative", "both"])
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--size", type=str, default="32k", choices=["32k", "128k", "both"])
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mem0_top_k", type=int, default=10, help="Number of Mem0 memories to retrieve per query")
    parser.add_argument("--mem0_llm", type=str, default=None, help="Azure deployment name for Mem0 extraction LLM")
    parser.add_argument("--mem0_embedding", type=str, default=None, help="Azure deployment name for Mem0 embedding model")
    parser.add_argument("--mem0_use_all", action="store_true", help="Use ALL extracted memories instead of top-k retrieval")
    parser.add_argument("--full_context", action="store_true", help="Append full conversation history alongside Mem0 memories")
    args = parser.parse_args()

    result_path = args.result_path or f"../results/mem0/{args.model_name}"

    evaluator = Mem0BenchmarkEvaluator(
        config_path=args.config_path,
        model_name=args.model_name,
        result_path=result_path,
        verbose=args.verbose,
        mem0_top_k=args.mem0_top_k,
        mem0_llm=args.mem0_llm,
        mem0_embedding=args.mem0_embedding,
        mem0_use_all=args.mem0_use_all,
        full_context=args.full_context,
    )

    output_file = evaluator.run_evaluation(
        benchmark_file=args.benchmark_file,
        eval_mode=args.eval_mode,
        max_items=args.max_items,
        size=args.size,
        parallel=args.parallel,
    )

    print(f"\nDone. Results at: {output_file}")


if __name__ == "__main__":
    main()
