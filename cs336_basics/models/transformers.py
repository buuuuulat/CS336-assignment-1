import torch
import torch.nn as nn

from cs336_basics.models.modules import RoPE, SwiGLU, RMSNorm, MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            rms_norm_eps: float = 1e-5,
            use_causal: bool = True,
            rope: RoPE | None = None,
            use_rope: bool = True,
            theta: float = 10000.0,
            max_seq_len: int = 256,
    ) -> None:
        """
        d_model: Dimensionality of the Transformer block inputs.
        num_heads: Number of heads to use in multi-head self-attention.
        d_ff: Dimensionality of the position-wise feed-forward layer.
        """
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=rms_norm_eps, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model = d_model,
            num_heads = num_heads,
            device = device,
            dtype = dtype,
            use_causal = use_causal,
            use_rope = use_rope,
            theta = theta,
            max_seq_len = max_seq_len,
            rope = rope,
        )
        self.ffn_norm = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.ffn = SwiGLU(
            d_model = d_model,
            d_ff = d_ff,
            device = device,
            dtype = dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.attn(self.attn_norm(x))
        z = y + self.ffn(self.ffn_norm(y))
        return z
