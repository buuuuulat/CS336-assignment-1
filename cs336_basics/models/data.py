import os
import typing

import torch
import numpy as np
import numpy.typing as npt


def get_batch(
        x: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: torch.device,
        rng = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rng is None:
        rng = np.random.default_rng()
    starts = rng.integers(0, len(x) - context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length)

    xs = torch.from_numpy(x[idx].astype(np.int64)).to(device)
    ys = torch.from_numpy(x[idx + 1].astype(np.int64)).to(device)
    return xs, ys


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        config: dict,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "config": config,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
) -> int:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
