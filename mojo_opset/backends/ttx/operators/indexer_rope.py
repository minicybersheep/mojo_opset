import torch

from mojo_opset.backends.ttx.kernels import indexer_rope

from mojo_opset.core import MojoIndexerRoPE


class TTXIndexerRoPE(MojoIndexerRoPE):
    supported_platforms_list = ["npu"]
    
    def forward(
        self,
        q : torch.Tensor,
        k : torch.Tensor,
        cos : torch.Tensor,
        sin : torch.Tensor,
        rope_head_dim : int = None
    )-> tuple[torch.Tensor, torch.Tensor]:
        # assert

        return indexer_rope(q, k, cos, sin, rope_head_dim)
