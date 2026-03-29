"""
Mem0 memory wrapper for PersonaMem-v2 benchmark.

Uses Azure OpenAI (same credentials from .env) for both LLM and embedder,
falling back to regular OpenAI if Azure env vars are not set.

Required .env additions:
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small  # or text-embedding-ada-002
"""

import os
import shutil
import tempfile
from dotenv import load_dotenv

load_dotenv(override=True)


def _build_mem0_config(llm_deployment: str = None, embedding_deployment: str = None, qdrant_path: str = None, history_db_path: str = None) -> dict:
    """Build Mem0 config using Azure OpenAI if available, else regular OpenAI."""
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    vector_store = {
        "provider": "qdrant",
        "config": {"collection_name": "personamem", "path": qdrant_path, "embedding_model_dims": 3072},
    }

    base_config = {}
    if history_db_path:
        base_config["history_db_path"] = history_db_path

    if azure_endpoint and azure_key and azure_api_version:
        llm_dep = llm_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
        emb_dep = (embedding_deployment
                   or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBED")
                   or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"))
        emb_api_version = os.getenv("AZURE_OPENAI_API_VERSION_EMBED") or azure_api_version
        print(f"[Mem0] Using Azure OpenAI — LLM: {llm_dep}, Embedder: {emb_dep}")
        return {
            **base_config,
            "llm": {
                "provider": "azure_openai",
                "config": {
                    "model": llm_dep,
                    "azure_kwargs": {
                        "azure_deployment": llm_dep,
                        "azure_endpoint": azure_endpoint,
                        "api_key": azure_key,
                        "api_version": azure_api_version,
                    },
                },
            },
            "embedder": {
                "provider": "azure_openai",
                "config": {
                    "model": emb_dep,
                    "azure_kwargs": {
                        "azure_deployment": emb_dep,
                        "azure_endpoint": azure_endpoint,
                        "api_key": azure_key,
                        "api_version": emb_api_version,
                    },
                },
            },
            "vector_store": vector_store,
        }
    else:
        # Fallback: regular OpenAI
        llm_model = llm_deployment or "gpt-4o-mini"
        emb_model = embedding_deployment or "text-embedding-3-small"
        openai_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        print(f"[Mem0] Using OpenAI — LLM: {llm_model}, Embedder: {emb_model}")
        return {
            **base_config,
            "llm": {
                "provider": "openai",
                "config": {"model": llm_model, "api_key": openai_key},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": emb_model, "api_key": openai_key},
            },
            "vector_store": vector_store,
        }


class PersonaMemory:
    """
    Manages Mem0 memory for a single worker/thread.
    Each parallel worker should instantiate its own PersonaMemory.
    Uses a unique temp directory for Qdrant to avoid cross-thread conflicts.
    """

    def __init__(self, llm_deployment: str = None, embedding_deployment: str = None):
        from mem0 import Memory
        self._qdrant_path = tempfile.mkdtemp(prefix="qdrant_pm_")
        self._history_db_path = os.path.join(self._qdrant_path, "history.db")
        self.memory = Memory.from_config(
            _build_mem0_config(
                llm_deployment, embedding_deployment,
                qdrant_path=self._qdrant_path,
                history_db_path=self._history_db_path,
            )
        )

    @staticmethod
    def _flatten_conversations(conversations: list) -> list:
        """Strip image/multimodal content — Mem0 crashes on list-content messages without vision enabled."""
        flat = []
        for msg in conversations:
            content = msg.get("content", "")
            if isinstance(content, list):
                text = " ".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ).strip()
                if text:
                    flat.append({"role": msg["role"], "content": text})
            elif isinstance(content, dict) and content.get("type") != "text":
                pass  # skip pure image dicts
            else:
                flat.append(msg)
        return flat

    @staticmethod
    def _system_as_user(conversations: list) -> list:
        """
        Mem0 ignores system messages when extracting facts.
        Convert the first system message (persona description) to a user message
        so Mem0 extracts those facts too.
        """
        out = []
        for msg in conversations:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content:
                    out.append({"role": "user", "content": f"Here is information about me: {content}"})
            else:
                out.append(msg)
        return out

    def build_from_history(self, conversations: list, user_id: str) -> None:
        """Ingest a chat history list (list of {role, content} dicts) into Mem0.

        Chunks into batches of 20 messages — sending 250+ messages at once causes the
        extraction LLM to return empty JSON (silent failure → 0 facts stored).
        """
        flat = self._flatten_conversations(conversations)
        flat = self._system_as_user(flat)
        if not flat:
            return
        chunk_size = 20
        for i in range(0, len(flat), chunk_size):
            self.memory.add(flat[i:i + chunk_size], user_id=user_id)

    def retrieve(self, query: str, user_id: str, top_k: int = 10) -> str:
        """Return a newline-joined string of top-k memories relevant to the query."""
        results = self.memory.search(query, user_id=user_id, limit=top_k)
        if not results:
            return ""
        # mem0 returns list of dicts with 'memory' key
        memories = results.get("results", results) if isinstance(results, dict) else results
        return "\n".join(f"- {r['memory']}" for r in memories if r.get("memory"))

    def reset(self, user_id: str) -> None:
        """Delete all memories for a user_id (call after each benchmark row)."""
        self.memory.delete_all(user_id=user_id)

    def cleanup(self) -> None:
        """Remove the temp Qdrant directory for this worker."""
        shutil.rmtree(self._qdrant_path, ignore_errors=True)
