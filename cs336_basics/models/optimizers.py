import math
import torch
from typing import Optional
from collections.abc import Callable


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]  # Get the learning rate.
                for p in group["params"]:
                    p: torch.Tensor
                    if p.grad is None:
                        continue
                state = self.state[p]  # Get the state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                p -= lr / math.sqrt(t + 1) * p.grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr = 1e-3,
            betas = (0.9, 0.999),
            eps = 1e-8,
            weight_decay = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                params: list[torch.Tensor] = group["params"]
                for p in params:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["t"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                    m, v = state["m"], state["v"]
                    t = state["t"] + 1
                    g = p.grad

                    # Computing the coefficients
                    alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                    p -= lr * weight_decay * p
                    m = beta1 * m + (1 - beta1) * g
                    v = beta2 * v + (1 - beta2) * g.pow(2)
                    p -= alpha_t * m / (torch.sqrt(v) + eps)

                    state["t"] = t
                    state["m"] = m
                    state["v"] = v
        return loss
