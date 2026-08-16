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
def gradient_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> torch.Tensor:
    """
    Clips gradients in-place, returns the total l2 norm before clipping.
    """
    with torch.no_grad():
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            return torch.tensor(0.0)
        total_norm = torch.sqrt(torch.stack([(g.float()**2).sum() for g in grads]).sum())
        if total_norm > max_l2_norm:
            scale = max_l2_norm / (total_norm + 1e-6)
            for g in grads:
                g.mul_(scale)
        return total_norm


def apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs - sorted_probs > p
    sorted_probs[mask] = 0.0
    result = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
    return result / result.sum(dim=-1, keepdim=True)


def decode(
        prompt,
        model,
        tokenizer,
        max_new_tokens=1024,
        eof_token="<|endoftext|>",
        temperature=1.0,
        top_p=1.0,
        device=None,
):
    token_ids = tokenizer.encode(prompt)
    tokens = torch.tensor(token_ids, dtype=torch.int64, device=device).unsqueeze(0)  # (1, seq_len)
    eof_id = tokenizer.encode(eof_token)[0]
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens)  # (1, seq_len, vocab_size)

        if temperature <= 0:
            next_logits = logits[0, -1, :]  # (vocab_size)
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = logits[0, -1, :] / temperature  # (vocab_size)
            probs = apply_top_p(softmax(next_logits, dim=-1), top_p)
            next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == eof_id:
            break
    return tokenizer.decode(tokens[0].tolist())
