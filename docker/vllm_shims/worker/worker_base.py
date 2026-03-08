"""Compatibility shim: vllm.worker.worker_base for vLLM 0.13.0+.
In vLLM 0.8.x this module existed directly; in 0.13.0 it was moved to vllm.v1.worker.worker_base.
verl bc2cc6b imports this but only uses it in the async rollout path (not used in sync mode)."""

try:
    from vllm.v1.worker.worker_base import WorkerWrapperBase
except ImportError:
    class WorkerWrapperBase:
        """Stub WorkerWrapperBase for import compatibility."""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("WorkerWrapperBase stub — use sync rollout mode")
