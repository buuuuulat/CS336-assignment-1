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
