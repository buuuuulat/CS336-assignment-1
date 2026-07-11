import torch


def softmax(x: torch.Tensor, dim=-1) -> torch.Tensor:
    normalized = x - x.max(dim=dim, keepdim=True).values
    return normalized.exp() / normalized.exp().sum(dim=dim, keepdim=True)
