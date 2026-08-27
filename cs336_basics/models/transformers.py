import torch
import torch.nn as nn

from cs336_basics.models.modules import Embedding, Linear, RoPE, SwiGLU, RMSNorm, MultiHeadSelfAttention


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
        
        self.ffn_norm = RMSNorm(d_model, eps=rms_norm_eps, device=device, dtype=dtype)
        
        self.ffn = SwiGLU(
            d_model = d_model,
            d_ff = d_ff,
            device = device,
            dtype = dtype,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        y = self.attn_norm(x + self.attn(x, token_positions))
        z = self.ffn_norm(y + self.ffn(y))
        return z


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            rms_norm_eps: float = 1e-5,
            use_causal: bool = True,
            use_rope: bool = True,
            theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(
            num_embeddings = vocab_size,
            embedding_dim = d_model,
            device = device,
            dtype = dtype,
        )

        if use_rope:
            d_k = d_model // num_heads
            self.rope = RoPE(theta, d_k, context_length, device=device)
        else:
            self.rope = None
        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            d_model = d_model,
            num_heads = num_heads,
            d_ff = d_ff,
            device = device,
            dtype = dtype,
            rms_norm_eps = rms_norm_eps,
            use_causal = use_causal,
            rope = self.rope,
            use_rope = use_rope,
            theta = theta,
            max_seq_len = context_length,
        ) for _ in range(num_layers)])

        self.norm = RMSNorm(d_model=d_model, eps=rms_norm_eps, device=device, dtype=dtype)
        self.linear = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        x = self.norm(x)
        x = self.linear(x)
        return x
