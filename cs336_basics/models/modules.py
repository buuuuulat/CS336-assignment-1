import math
from typing import cast

import torch
import torch.nn as nn
from einops import einsum, rearrange

from cs336_basics.models.utils import scaled_dot_product_attention


def init_linear_(w: torch.Tensor, d_in: int, d_out: int) -> torch.Tensor:
    std = math.sqrt(2 / (d_in + d_out))
    return nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)


class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.W = nn.Parameter(
            init_linear_(
                torch.empty(
                    out_features,
                    in_features,
                    device=device,
                    dtype=dtype,
                ),
                d_in=in_features,
                d_out=out_features,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "... out_features in_features, ... in_features -> ... out_features")


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(
                    num_embeddings,
                    embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
                mean=0.0,
                std=1.0,
                a=-3.0,
                b=3.0,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings[x]


class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gs = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        result = einsum(x / rms, self.gs, "... d_in, ... d_in -> ... d_in")
        return result.to(original_dtype)


class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.W1 = nn.Parameter(
            init_linear_(
                torch.empty(
                    d_ff,
                    d_model,
                    device=device,
                    dtype=dtype,
                ),
                d_in=d_model,
                d_out=d_ff,
            ),
        )
        self.W2 = nn.Parameter(
            init_linear_(
                torch.empty(
                    d_model,
                    d_ff,
                    device=device,
                    dtype=dtype,
                ),
                d_in=d_ff,
                d_out=d_model,
            ),
        )
        self.W3 = nn.Parameter(
            init_linear_(
                torch.empty(
                    d_ff,
                    d_model,
                    device=device,
                    dtype=dtype,
                ),
                d_in=d_model,
                d_out=d_ff,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        w3x = einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = w1x * w1x.sigmoid()
        gated = w3x * silu
        result = einsum(self.W2, gated, "... d_model d_ff, ... d_ff -> ... d_model")
        return result


class RoPE(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0, "d_k should be even"
        pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (max_seq_len)
        k = torch.arange(d_k // 2, dtype=torch.float32, device=device)  # (d_k // 2)
        freq = theta ** (-2 * k / d_k)  # (d_k // 2)
        angles = cast(torch.Tensor, einsum(pos, freq, "max_seq_len, num_pairs -> max_seq_len num_pairs"))

        sin_t = torch.sin(angles)
        cos_t = torch.cos(angles)
        self.register_buffer("sin", sin_t, persistent=False)
        self.register_buffer("cos", cos_t, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        x_pairs = rearrange(x, "... seq_len (pairs two) -> ... seq_len pairs two", two=2)
        x1 = x_pairs[..., 0]  # (..., seq_len, pairs)
        x2 = x_pairs[..., 1]  # (..., seq_len, pairs)

        sin = self.sin[token_positions]  # (..., seq_len, pairs)
        cos = self.cos[token_positions]  # (..., seq_len, pairs)
        x1_r = cos * x1 - sin * x2
        x2_r = sin * x1 + cos * x2

        x_r = torch.stack((x1_r, x2_r), dim=-1)  # (..., seq_len, pairs, 2)
        x_r = rearrange(x_r, "... seq_len pairs two -> ... seq_len (pairs two)")
        return x_r


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            use_causal: bool = True,
            use_rope: bool = True,
            theta: float = 10000.0,
            max_seq_len: int = 256,
            rope: RoPE | None = None,  # if passed, use_rope becomes True
    ):
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_v = d_model // num_heads
        self.num_heads = num_heads
        self.split_sizes = [num_heads * d_k, num_heads * d_k, num_heads * d_v]
        self.use_causal = use_causal
        self.Wqkv = nn.Parameter(
            init_linear_(
                torch.empty(
                    num_heads * d_k + num_heads * d_k + num_heads * d_v,
                    d_model,
                    device=device,
                    dtype=dtype,
                ),
                d_in=d_model,
                d_out=d_model,
            ),
        )
        self.Wo = nn.Parameter(
            init_linear_(
                torch.empty(
                    d_model,
                    num_heads * d_v,
                    device=device,
                    dtype=dtype,
                ),
                d_in=d_model,
                d_out=num_heads * d_v,
            ),
        )
        if rope is not None:
            self.rope = rope
        elif use_rope:
            self.rope = RoPE(theta, d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x, token_positions=None):  # x: (..., seq_len, d_model)
        if self.use_causal:
            mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2], device=x.device, dtype=torch.bool))
        else:
            mask = None

        qkv = einsum(x, self.Wqkv, "... seq_len d_model, qkv d_model -> ... seq_len qkv")
        q, k, v = torch.split(qkv, self.split_sizes, dim=-1)
        q = rearrange(q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        k = rearrange(k, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        v = rearrange(v, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        multihead = scaled_dot_product_attention(q, k, v, mask=mask)  # (..., q_n, v)
        multihead = rearrange(multihead, "... h seq d_v -> ... seq (h d_v)")
        attention = einsum(self.Wo, multihead, "d_model d_v_out, ... seq d_v_out -> ... seq d_model")
        return attention
