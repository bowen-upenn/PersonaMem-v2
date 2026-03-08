"""flash_attn stub: provides bert_padding utilities without CUDA flash attention kernels."""
__version__ = "2.7.0"


def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("flash_attn CUDA kernels not available (V100 stub)")


def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError("flash_attn CUDA kernels not available (V100 stub)")


def flash_attn_with_kvcache(*args, **kwargs):
    raise NotImplementedError("flash_attn CUDA kernels not available (V100 stub)")
