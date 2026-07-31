import math
import torch
from typing import Optional
from collections.abc import Callable, Iterable


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
