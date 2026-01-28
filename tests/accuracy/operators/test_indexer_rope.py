import random
import os
import pytest
import torch

from mojo_opset.utils.platform import get_platform
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoLinear, MojoGroupLinear

from mojo_opset import MojoIndexerRoPE


dtype_str_map = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}

def precompute_freqs_cis(seqlen, dim, device="npu") -> torch.Tensor:
    base = 10000.0
    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs, device=device), freqs)
    return freqs_cis

@pytest.mark.parametrize(
    "batch_size, seq_len, n_qh, head_dim, rope_head_dim, dtype",
    [
        (batch_size, 1024, 128, 128, 64, dtype)
        for batch_size in [2, 8]
        # for dtype in ["bfloat16"]
        for dtype in ["bfloat16", "float16", "float32"]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_indexer_rope(batch_size, seq_len, n_qh, head_dim, rope_head_dim, dtype):
    device = get_platform()
    map_tol = {
        "bfloat16": (1.6e-2, 1e-5),
        "float16": (1e-3, 1e-5),
        "float32": (1.3e-6, 1e-5),
    }
    if device == 'npu':
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
    atol, rtol = map_tol[dtype]
    dtype = dtype_str_map[dtype]
    
    # create input tensor
    q = torch.randn(batch_size, seq_len, n_qh, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)

    freqs_cis = precompute_freqs_cis(seq_len, rope_head_dim)

    cos = freqs_cis.real.unsqueeze(0).expand(batch_size, -1, -1)
    sin = freqs_cis.imag.unsqueeze(0).expand(batch_size, -1, -1)

    # indexer rope
    indexer_rope = MojoIndexerRoPE()

    # ref indexer rope
    indexer_rope_ref = MojoIndexerRoPE._registry.get("torch")()
    indexer_rope.forward_diff_with(indexer_rope_ref, q, k, cos, sin, rope_head_dim, atol=atol, rtol=rtol)
    # breakpoint()

    # print("end")