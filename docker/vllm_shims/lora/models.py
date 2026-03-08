"""Compatibility shim: vllm.lora.models for vLLM 0.13.0+.
In vLLM 0.8.x this module existed; in 0.13.0 it was reorganized into model_manager.py.
verl bc2cc6b imports LoRAModel but only uses it in the LoRA code path (not used without LoRA)."""


class LoRAModel:
    """Stub LoRAModel for import compatibility."""

    @classmethod
    def from_lora_tensors(cls, *args, **kwargs):
        raise NotImplementedError("LoRAModel stub — LoRA not enabled")

    @classmethod
    def from_local_checkpoint(cls, *args, **kwargs):
        raise NotImplementedError("LoRAModel stub — LoRA not enabled")
