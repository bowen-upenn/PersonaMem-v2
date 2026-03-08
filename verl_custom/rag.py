#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) baseline for ImplicitPersona benchmark.

Reads the benchmark CSV, applies dense retrieval (embed + cosine similarity)
to shorten each row's prompt, and outputs VERL-compatible parquet files.
The existing VERL pipeline (run_qwen3_4b_inference.sh) then runs Qwen3-4B
inference on the RAG parquet.

Usage:
    python rag.py --benchmark_csv data/benchmark/text/benchmark.csv \
                  --output_dir verl_custom/data/implicit_persona_rag \
                  --chunk_size 6 --chunk_overlap 2 --top_k 10 \
                  --sample_size 100
"""

import argparse
import ast
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# -------------------------------------------------------
# Embedding client
# -------------------------------------------------------

def init_embedding_client() -> Tuple[Any, str]:
    """
    Initialize OpenAI/Azure embedding client following image_matcher.py pattern.

    Returns:
        (client, model_name) tuple
    """
    load_dotenv(override=True)

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBED")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION_EMBED")

    if azure_endpoint and azure_key and azure_deployment and azure_api_version:
        print("Using Azure OpenAI for embeddings")
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=azure_api_version,
        )
        return client, azure_deployment

    openai_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-large")
    if openai_key:
        print("Using OpenAI for embeddings")
        client = OpenAI(api_key=openai_key)
        return client, openai_model

    raise ValueError(
        "No embedding configuration found. Set AZURE_OPENAI_DEPLOYMENT_NAME_EMBED "
        "or OPENAI_KEY + OPENAI_MODEL_EMBED in .env"
    )


def get_embeddings_batch(
    client, model_name: str, texts: List[str], batch_size: int = 100
) -> np.ndarray:
    """Embed a list of texts in batches. Returns (N, dim) array."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model_name, input=batch)
        batch_embeddings = [np.array(item.embedding) for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return np.array(all_embeddings)


# -------------------------------------------------------
# Chat history loading (from data_preprocess_rft.py)
# -------------------------------------------------------

def load_conversation_context(context_file_path: str) -> List[Dict[str, Any]]:
    """
    Load conversation context from JSON file.
    Supports {"chat_history": [...]}, {"conversations": [...]}, or [...].
    """
    try:
        with open(context_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "chat_history" in data:
            return data["chat_history"]
        elif isinstance(data, dict) and "conversations" in data:
            return data["conversations"]
        elif isinstance(data, list):
            return data
        else:
            # Try nested structure
            for value in data.values():
                if isinstance(value, dict) and "conversations" in value:
                    return value["conversations"]
                if isinstance(value, list):
                    return value
            print(f"Warning: Unexpected JSON structure in {context_file_path}")
            return []
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load context file {context_file_path}: {e}")
        return []


# -------------------------------------------------------
# Chunking
# -------------------------------------------------------

def chunk_chat_history(
    messages: List[Dict[str, str]],
    chunk_size: int = 6,
    chunk_overlap: int = 2,
) -> List[Dict[str, Any]]:
    """
    Split conversation messages into overlapping chunks.

    Args:
        messages: List of message dicts (system message already removed).
        chunk_size: Number of messages per chunk.
        chunk_overlap: Overlap between adjacent chunks.

    Returns:
        List of dicts with keys: 'messages', 'start_idx', 'text'
    """
    if not messages:
        return []

    stride = max(1, chunk_size - chunk_overlap)
    chunks = []

    for start in range(0, len(messages), stride):
        end = min(start + chunk_size, len(messages))
        chunk_messages = messages[start:end]

        # Serialize for embedding
        text_parts = []
        for msg in chunk_messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            # Handle multimodal content lists
            if isinstance(content, list):
                text_pieces = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                content = " ".join(text_pieces)
            text_parts.append(f"{role}: {content}")

        chunks.append(
            {
                "messages": chunk_messages,
                "start_idx": start,
                "text": "\n".join(text_parts),
            }
        )

        if end >= len(messages):
            break

    return chunks


# -------------------------------------------------------
# Embedding cache
# -------------------------------------------------------

class EmbeddingCache:
    """Two-level cache: in-memory dict + disk .npz files."""

    def __init__(self, cache_dir: str, chunk_size: int, chunk_overlap: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._mem: Dict[str, np.ndarray] = {}

    def _key(self, filepath: str) -> str:
        raw = f"{filepath}_cs{self.chunk_size}_co{self.chunk_overlap}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, filepath: str) -> Optional[np.ndarray]:
        if filepath in self._mem:
            return self._mem[filepath]
        disk_path = self.cache_dir / f"{self._key(filepath)}.npz"
        if disk_path.exists():
            data = np.load(disk_path)
            arr = data["embeddings"]
            self._mem[filepath] = arr
            return arr
        return None

    def put(self, filepath: str, embeddings: np.ndarray):
        self._mem[filepath] = embeddings
        disk_path = self.cache_dir / f"{self._key(filepath)}.npz"
        np.savez(disk_path, embeddings=embeddings)


# -------------------------------------------------------
# Retrieval
# -------------------------------------------------------

def retrieve_chunks(
    query_embedding: np.ndarray,
    chunks: List[Dict[str, Any]],
    chunk_embeddings: np.ndarray,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Retrieve top_k chunks by cosine similarity, then re-sort by original position.
    """
    if len(chunks) <= top_k:
        return chunks  # All chunks fit, no retrieval needed

    similarities = cosine_similarity(
        query_embedding.reshape(1, -1), chunk_embeddings
    )[0]

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    retrieved = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["similarity_score"] = float(similarities[idx])
        retrieved.append(chunk)

    # Re-sort by original position to preserve temporal order
    retrieved.sort(key=lambda c: c["start_idx"])
    return retrieved


# -------------------------------------------------------
# Prompt construction
# -------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a helpful assistant that provides personalized responses "
    "based on the user's preferences in conversation history."
)

RAG_INSTRUCTION = (
    "The following are relevant excerpts from your conversation history "
    "with this user. Use them to personalize your response."
)

THINKING_INSTRUCTION = (
    " Always perform your reasoning inside <think> and </think> tags "
    "before your final answer."
)


def build_rag_prompt(
    retrieved_chunks: List[Dict[str, Any]],
    question: str,
) -> List[Dict[str, str]]:
    """
    Assemble the final message list for the VERL parquet prompt field.
    Deduplicates overlapping messages between adjacent chunks using start_idx.
    """
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "system", "content": RAG_INSTRUCTION},
    ]

    # Track which original message indices have been added to avoid duplicates
    # from chunk overlaps. Chunks are already sorted by start_idx.
    added_indices = set()
    for chunk in retrieved_chunks:
        start = chunk["start_idx"]
        for offset, msg in enumerate(chunk["messages"]):
            msg_idx = start + offset
            if msg_idx not in added_indices:
                added_indices.add(msg_idx)
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)

    messages.append({"role": "user", "content": question})
    return messages


# -------------------------------------------------------
# CSV helpers (from data_preprocess_rft.py)
# -------------------------------------------------------

def safe_eval_list(value) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            return [str(parsed)]
        return [str(value)]
    except (ValueError, SyntaxError):
        return [str(value)]


def extract_question_from_row(row: pd.Series) -> str:
    user_query = row.get("user_query", "")
    if isinstance(user_query, dict):
        return str(user_query.get("content", ""))
    if isinstance(user_query, str):
        try:
            uq_dict = ast.literal_eval(user_query)
            if isinstance(uq_dict, dict) and "content" in uq_dict:
                return str(uq_dict["content"])
        except (ValueError, SyntaxError):
            pass
        return user_query
    return str(user_query) if user_query is not None else ""


# -------------------------------------------------------
# Main processing
# -------------------------------------------------------

def process_row_rag(
    row: pd.Series,
    idx: int,
    embed_client,
    embed_model: str,
    embedding_cache: EmbeddingCache,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    is_mcq: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Process a single benchmark row: load chat history, apply RAG, build VERL record.
    """
    try:
        # 1. Load chat history
        chat_history_path = row.get("chat_history_32k_link", "")
        if not chat_history_path or not os.path.exists(chat_history_path):
            print(f"Warning: Chat history not found for row {idx}: {chat_history_path}")
            return None

        conversations = load_conversation_context(chat_history_path)
        if not conversations:
            print(f"Warning: Empty chat history for row {idx}")
            return None

        # 2. Separate system message from conversation messages
        system_msg = None
        conv_messages = []
        for msg in conversations:
            if isinstance(msg, dict):
                if msg.get("role") == "system" and system_msg is None:
                    system_msg = msg
                else:
                    conv_messages.append(msg)

        # 3. Chunk conversation messages
        chunks = chunk_chat_history(conv_messages, chunk_size, chunk_overlap)
        if not chunks:
            print(f"Warning: No chunks created for row {idx}")
            return None

        # 4. Embed chunks (with caching)
        cached = embedding_cache.get(chat_history_path)
        if cached is not None and cached.shape[0] == len(chunks):
            chunk_embeddings = cached
        else:
            chunk_texts = [c["text"] for c in chunks]
            chunk_embeddings = get_embeddings_batch(
                embed_client, embed_model, chunk_texts
            )
            embedding_cache.put(chat_history_path, chunk_embeddings)

        # 5. Build user question
        question = extract_question_from_row(row)

        # 6. Embed query and retrieve
        query_embedding = get_embeddings_batch(
            embed_client, embed_model, [question]
        )[0]
        retrieved = retrieve_chunks(query_embedding, chunks, chunk_embeddings, top_k)

        # 7. Add MCQ options if requested
        correct_answer_text = row.get("correct_answer", "")
        incorrect_answers = safe_eval_list(row.get("incorrect_answers", "[]"))
        all_answers = []
        if correct_answer_text != "":
            all_answers.append(str(correct_answer_text))
        all_answers.extend(str(x) for x in incorrect_answers)

        if is_mcq and len(all_answers) >= 4:
            shuffled_answers = all_answers[:4].copy()
            random.seed(42 + idx)
            random.shuffle(shuffled_answers)

            correct_index = shuffled_answers.index(str(correct_answer_text))
            correct_letter = chr(97 + correct_index)
            correct_answer = f"({correct_letter}) {correct_answer_text}"

            options_text = "\n".join(
                f"({chr(97 + i)}) {option}"
                for i, option in enumerate(shuffled_answers)
            )
            mcq_prompt = (
                f"\n\nYou are performing a multiple-choice question task. "
                f"You must choose the best response from the following options "
                f"to answer the user query above:\n{options_text}\n\n"
                f"Provide your answer in the format: \\boxed{{a}}, \\boxed{{b}}, "
                f"\\boxed{{c}}, or \\boxed{{d}}."
            )
            question = question + mcq_prompt
            all_answers = shuffled_answers
        else:
            correct_answer = correct_answer_text

        # Add thinking instruction
        question += THINKING_INSTRUCTION

        # 8. Build RAG prompt
        prompt_messages = build_rag_prompt(retrieved, question)

        # 9. Build VERL record
        pref_type = row.get("pref_type", "")
        groundtruth_preference = row.get(
            "preference", row.get("groundtruth_preference", "")
        )

        verl_data = {
            "data_source": "implicit_persona",
            "prompt": prompt_messages,
            "ability": "personalization",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "groundtruth_preference": str(groundtruth_preference),
                    "correct_answer": str(correct_answer),
                    "all_answers": [str(opt) for opt in all_answers],
                    "pref_type": str(pref_type),
                    "is_mcq": is_mcq,
                },
            },
            "extra_info": {
                "index": idx if pd.notna(idx) else 0,
                "persona_id": row.get("persona_id"),
                "question": question,
            },
        }

        return verl_data

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="RAG baseline: create VERL parquet with retrieval-shortened prompts"
    )
    parser.add_argument(
        "--benchmark_csv",
        default="benchmark/text/benchmark.csv",
        help="Path to benchmark CSV file",
    )
    parser.add_argument(
        "--output_dir",
        default="verl_custom/data/implicit_persona_rag",
        help="Output directory for RAG parquet files",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=6,
        help="Number of messages per chunk (default: 6)",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=2,
        help="Overlap between adjacent chunks (default: 2)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of rows to process (default: all)",
    )
    parser.add_argument(
        "--cache_dir",
        default="data/rag_embedding_cache",
        help="Directory for cached embeddings",
    )
    args = parser.parse_args()

    # Setup
    print("=" * 60)
    print("RAG Baseline - Parquet Preprocessing")
    print("=" * 60)
    print(f"Benchmark CSV: {args.benchmark_csv}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Chunk size:    {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print(f"Top-k:         {args.top_k}")
    print(f"Sample size:   {args.sample_size or 'all'}")
    print("=" * 60)

    # Initialize embedding client
    embed_client, embed_model = init_embedding_client()
    embedding_cache = EmbeddingCache(args.cache_dir, args.chunk_size, args.chunk_overlap)

    # Load benchmark data
    raw_data = pd.read_csv(args.benchmark_csv)
    print(f"Loaded {len(raw_data)} rows from benchmark")

    if args.sample_size is not None and len(raw_data) > args.sample_size:
        raw_data = raw_data.sample(n=args.sample_size, random_state=42).reset_index(
            drop=True
        )
        print(f"Sampled {len(raw_data)} rows")

    # Process both open-ended and MCQ versions
    os.makedirs(args.output_dir, exist_ok=True)

    for is_mcq, suffix in [(False, ""), (True, "_mcq")]:
        label = "MCQ" if is_mcq else "open-ended"
        print(f"\nProcessing {label} version...")

        verl_records = []
        for idx, row in tqdm(
            raw_data.iterrows(), total=len(raw_data), desc=f"RAG ({label})"
        ):
            record = process_row_rag(
                row=row,
                idx=idx,
                embed_client=embed_client,
                embed_model=embed_model,
                embedding_cache=embedding_cache,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                top_k=args.top_k,
                is_mcq=is_mcq,
            )
            if record is not None:
                verl_records.append(record)

        print(f"Successfully processed {len(verl_records)} / {len(raw_data)} rows")

        if not verl_records:
            print(f"Warning: No records produced for {label}, skipping parquet output")
            continue

        # Convert to DataFrame and serialize nested structures as JSON strings
        df = pd.DataFrame(verl_records)
        df["prompt"] = df["prompt"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
        df["reward_model"] = df["reward_model"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
        df["extra_info"] = df["extra_info"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )

        out_path = os.path.join(args.output_dir, f"benchmark_text_32k{suffix}.parquet")
        df.to_parquet(out_path, engine="pyarrow")
        print(f"Saved {len(df)} records to {out_path}")

        # Print prompt length stats
        prompt_lengths = df["prompt"].apply(len)
        print(
            f"  Prompt string lengths: "
            f"min={prompt_lengths.min()}, "
            f"mean={prompt_lengths.mean():.0f}, "
            f"max={prompt_lengths.max()}"
        )

    print("\nDone! RAG parquet files ready for VERL inference.")


if __name__ == "__main__":
    main()
