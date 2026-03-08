"""Compatibility shim: vllm.lora.request for vLLM 0.13.0+.
In vLLM 0.8.x this module existed; in 0.13.0 it was removed.
verl bc2cc6b uses LoRARequest as a base class for TensorLoRARequest (msgspec.Struct)."""

import msgspec


class LoRARequest(
    msgspec.Struct,
    tag=True,
    tag_field="__type",
    rename={"lora_int_id": "lora_id"},
):
    """Stub LoRARequest for import compatibility."""

    lora_name: str = ""
    lora_int_id: int = 0
    lora_path: str = ""
