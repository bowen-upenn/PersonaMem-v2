"""flash_attn.ops.triton.rotary stub for V100 compatibility.
vLLM 0.13.0 imports this at init time but only uses it in forward_hip (AMD).
On CUDA, forward_cuda uses vllm.vllm_flash_attn instead."""


def apply_rotary(x, cos, sin, interleaved=False, inplace=False, conjugate=False):
    """Stub — only called on AMD GPUs (forward_hip path)."""
    raise NotImplementedError("flash_attn Triton rotary not available (V100 stub)")
