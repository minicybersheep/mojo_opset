from typing import Optional
from typing import Tuple

import torch

from ..operator import MojoOperator


class MojoRoPE(MojoOperator):
    def __init__(
        self,
        rotary_offset: int = 0,
        interleaved: bool = False,
        dynamic_ntk: bool = False,
        max_seq_len: Optional[int] = None,
        is_varlen: bool = True,
    ):
        """
        Initialize rotary position embedding configuration.

        Args:
            rotary_offset (int, default=0): Non-negative rotation offset applied to positions.
            interleaved (bool, default=False): If True, use interleaved head layout when applying rotary.
            dynamic_ntk (bool, default=False): Enable dynamic NTK scaling to extend effective context length.
            max_seq_len (Optional[int], default=None): Positive max sequence length for precomputation; or None.
            is_varlen (bool, default=True): Prefer token-first TND layout when True; else use BSND.

        Raises:
            ValueError: If `rotary_offset` < 0 or `max_seq_len` <= 0 when provided.
            TypeError: If `interleaved`, `dynamic_ntk`, or `is_varlen` are not bools.

        Notes:
            This initializer performs light validation and stores configuration flags; the
            actual rotary application happens in the forward path.
        """
        super().__init__()

        if not isinstance(rotary_offset, int) or rotary_offset < 0:
            raise ValueError("rotary_offset should be non-negative integer.")
        if not isinstance(interleaved, bool):
            raise TypeError("interleaved should be bool.")
        if not isinstance(dynamic_ntk, bool):
            raise TypeError("dynamic_ntk should be bool.")
        if max_seq_len is not None and (not isinstance(max_seq_len, int) or max_seq_len <= 0):
            raise ValueError("max_seq_len should be positive integer or None.")
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen should be bool.")

        self.rotary_offset = rotary_offset
        self.interleaved = interleaved
        self.dynamic_ntk = dynamic_ntk
        self.max_seq_len = max_seq_len
        self.is_varlen = is_varlen

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cum_sum_query_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings (RoPE) to queries and keys.

        Args:
            q (torch.Tensor): Query tensor; last dimension must be even to allow rotation.
            k (torch.Tensor): Key tensor; same shape as `q`.
            cos (torch.Tensor): Precomputed cosine tensor, broadcastable to `q`/`k`.
            sin (torch.Tensor): Precomputed sine tensor, broadcastable to `q`/`k`.
            position_ids (Optional[torch.Tensor], default=None): Reserved; not used here.
            cum_sum_query_len (Optional[torch.Tensor], default=None): Reserved; not used here.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `(q_rot, k_rot)` with the same shape/dtype as inputs.

        Notes:
            Uses standard RoPE: `x_rot = x * cos + rotate_half(x) * sin`, where `rotate_half`
            swaps the two halves of the last dimension with a sign flip.
        """

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot

class MojoIndexerRoPE(MojoOperator):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q : torch.Tensor,
        k : torch.Tensor,
        cos : torch.Tensor,
        sin : torch.Tensor,
        rope_head_dim : int = None
        )-> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to two tensors simultaneously
        
        Args:
            q: First input tensor [batch, seq_len, dim] or [batch, seq_len, n_heads, dim]
            k: Second input tensor [batch, seq_len, dim] or [batch, seq_len, n_heads, dim]
            cos: Cosine component [batch, seq_len, dim//2]
            sin: Sine component [batch, seq_len, dim//2]
            rope_head_dim: the dim neads apply rope
        Returns:
            q_rope, k_rope: Rotated tensors
        """
        # Ensure cos and sin have correct dimensions
        assert cos.dim() == 3 and sin.dim() == 3, "cos and sin must be 3D tensors"
        assert cos.shape == sin.shape, "cos and sin must have the same shape"
        q_pe, q_nope = torch.split(q, [rope_head_dim, q.size(-1) - rope_head_dim], dim=-1)
        k_pe, k_nope = torch.split(k, [rope_head_dim, k.size(-1) - rope_head_dim], dim=-1)

        def _apply_rotary_emb_single(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            """Apply rotary position embedding to a single tensor"""
            dtype = tensor.dtype
            original_shape = tensor.shape
            tensor_float = tensor.float()
            
            if tensor.dim() == 4:
                # [batch, seq_len, n_heads, dim]
                batch, seq_len, n_heads, dim = tensor.shape
                cos_expanded = cos.unsqueeze(2)  # [batch, seq_len, 1, dim//2]
                sin_expanded = sin.unsqueeze(2)  # [batch, seq_len, 1, dim//2]
            elif tensor.dim() == 3:
                # [batch, seq_len, dim]
                batch, seq_len, dim = tensor.shape
                cos_expanded = cos
                sin_expanded = sin
            else:
                raise ValueError(f"Input tensor must be 3D or 4D, got {tensor.dim()}D")
            
            assert dim % 2 == 0, f"dim must be even, got {dim}"
            assert cos_expanded.size(-1) == dim // 2, f"cos last dim must be {dim//2}, got {cos_expanded.size(-1)}"
            
            # Reshape tensor to non-interleaved format: [x0, x1, x2, x3, ...] -> [x0, x2, ..., x1, x3, ...]
            if tensor.dim() == 4:
                # [batch, seq_len, n_heads, dim] -> [batch, seq_len, n_heads, 2, dim//2]
                tensor_reshaped = tensor_float.view(batch, seq_len, n_heads, 2, dim // 2)
                tensor_reshaped = tensor_reshaped.transpose(3, 4).contiguous()
                # [batch, seq_len, n_heads, dim//2, 2]
                x1 = tensor_reshaped[..., 0]  # Real part
                x2 = tensor_reshaped[..., 1]  # Imaginary part
            else:
                # [batch, seq_len, dim] -> [batch, seq_len, 2, dim//2]
                tensor_reshaped = tensor_float.view(batch, seq_len, 2, dim // 2)
                tensor_reshaped = tensor_reshaped.transpose(2, 3).contiguous()
                # [batch, seq_len, dim//2, 2]
                x1 = tensor_reshaped[..., 0]  # Real part
                x2 = tensor_reshaped[..., 1]  # Imaginary part
            
            # Apply rotation matrix: [x1, x2] * [cos, -sin; sin, cos] = [x1*cos - x2*sin, x1*sin + x2*cos]
            x1_rot = x1 * cos_expanded - x2 * sin_expanded
            x2_rot = x1 * sin_expanded + x2 * cos_expanded
            
            # Convert back to interleaved format
            if tensor.dim() == 4:
                # [batch, seq_len, n_heads, dim//2, 2]
                rotated = torch.stack([x1_rot, x2_rot], dim=-1)
                rotated = rotated.transpose(3, 4).contiguous()
                rotated = rotated.view(batch, seq_len, n_heads, dim)
            else:
                # [batch, seq_len, dim//2, 2]
                rotated = torch.stack([x1_rot, x2_rot], dim=-1)
                rotated = rotated.transpose(2, 3).contiguous()
                rotated = rotated.view(batch, seq_len, dim)
            
            return rotated.to(dtype)

        # Apply rotary position embedding to x and y separately
        q_pe = _apply_rotary_emb_single(q_pe, cos, sin)
        k_pe = _apply_rotary_emb_single(k_pe, cos, sin)

        q_rope = torch.cat([q_pe, q_nope], dim=-1)
        k_rope = torch.cat([k_pe, k_nope], dim=-1)
        return q_rope, k_rope

class MojoRoPEStoreKV(MojoOperator):
    pass


class MojoNormRoPE(MojoOperator):
    pass


class MojoNormRoPEStoreKV(MojoOperator):
    pass
