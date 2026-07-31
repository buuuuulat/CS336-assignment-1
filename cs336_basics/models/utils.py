import torch
from einops import einsum


def softmax(x: torch.Tensor, dim=-1) -> torch.Tensor:
    normalized = x - x.max(dim=dim, keepdim=True).values
    return normalized.exp() / normalized.exp().sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    queries: torch.Tensor, shape: (batch_size, ..., n, d_k)
    keys: torch.Tensor, shape: (batch_size, ..., m, d_k)
    values: torch.Tensor, shape: (batch_size, ..., m, d_v)
    mask: torch.Tensor, shape: (seq_len, seq_len)
    """
    pre_softmax = einsum(queries, keys, "... n k, ... m k -> ... n m") / keys.shape[-1]**0.5
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float("-inf"))
    normalized_scores = softmax(pre_softmax, dim=-1)
    attention = einsum(normalized_scores, values, "... n m, ... m v -> ... n v")
    return attention


def cross_entropy_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
) -> torch.Tensor:
    # Subtracting the largest element for numerical stability
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    loss = (torch.logsumexp(shifted, dim=-1) - torch.gather(shifted, -1, targets.unsqueeze(-1)).squeeze(-1)).mean()
    return loss
