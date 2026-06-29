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
