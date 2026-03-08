"""flash_attn.layers.rotary stub — only used when attn_implementation='flash_attention_2'.
With attn_implementation='eager' on V100, these functions are never called."""


def apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False, conjugate=False):
    raise NotImplementedError("flash_attn rotary kernels not available (V100 stub)")


class RotaryEmbedding:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("flash_attn RotaryEmbedding not available (V100 stub)")
