"""Compatibility shim: vllm.lora.worker_manager for vLLM 0.13.0+.
In vLLM 0.8.x this module existed; in 0.13.0 it was reorganized.
verl bc2cc6b monkey-patches LRUCacheWorkerLoRAManager._load_adapter (only used with LoRA)."""


class LRUCacheWorkerLoRAManager:
    """Stub LRUCacheWorkerLoRAManager for import compatibility."""

    def _load_adapter(self, *args, **kwargs):
        raise NotImplementedError("LRUCacheWorkerLoRAManager stub — LoRA not enabled")
