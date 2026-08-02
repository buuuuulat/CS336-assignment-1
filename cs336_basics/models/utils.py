import math
from collections.abc import Iterable

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


def lr_scheduler(
        t: int,
        lr_max: float,
        lr_min: float,
        t_warmup: int,
        t_c: int,
) -> float:
    """
    Returns learning rate for iteration t.
    """
    assert 0 <= t_warmup <= t_c, f"Need 0 <= t_warmup <= t_c, but {t_warmup}, {t_c} were given."
    if t < t_warmup:
        return (t / t_warmup) * lr_max
    elif t <= t_c:
        return lr_min + 0.5 * (1 + math.cos(math.pi * (t - t_warmup) / (t_c - t_warmup))) * (lr_max - lr_min)
    else:
        return lr_min


# noinspection unsupported-operator
def gradient_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    with torch.no_grad():
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            return
        total_norm = torch.sqrt(torch.stack([(g.float()**2).sum() for g in grads]).sum())
        if total_norm > max_l2_norm:
            scale = max_l2_norm / (total_norm + 1e-6)
            for g in grads:
                g.mul_(scale)
