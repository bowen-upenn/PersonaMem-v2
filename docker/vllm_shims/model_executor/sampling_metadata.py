"""Compatibility shim: vllm.model_executor.sampling_metadata for vLLM 0.13.0+.
In vLLM 0.8.x this module existed directly; in 0.13.0 it was moved to vllm.v1.sample.metadata.
This shim is only used as a type annotation in verl and is never instantiated at runtime."""


class SamplingMetadata:
    """Stub SamplingMetadata for type annotation compatibility."""
    pass
