import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.W = nn.Parameter(nn.init.trunc_normal_(torch.empty(out_features, in_features, device=device, dtype=dtype)))

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
                    dtype=dtype
                )
            )
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
        self.W1 = nn.Parameter(nn.init.trunc_normal_(torch.empty(d_ff, d_model, device=device, dtype=dtype)))
        self.W2 = nn.Parameter(nn.init.trunc_normal_(torch.empty(d_model, d_ff, device=device, dtype=dtype)))
        self.W3 = nn.Parameter(nn.init.trunc_normal_(torch.empty(d_ff, d_model, device=device, dtype=dtype)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        w3x = einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = w1x * w1x.sigmoid()
        gated = w3x * silu
        result = einsum(self.W2, gated, "... d_model d_ff, ... d_ff -> ... d_model")
        return result
