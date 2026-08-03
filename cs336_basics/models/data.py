import os
import typing

import torch
import numpy as np
import numpy.typing as npt


def get_batch(
        x: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str = 'cpu',
        rng = None,
):
    if device not in ['cpu', 'mps'] and 'cuda' not in device:
        raise ValueError(f"Invalid device type: {device}")
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
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(src)
    model_dict, optimizer_dict, iteration = checkpoint["model"], checkpoint["optimizer"], checkpoint["iteration"]
    model.load_state_dict(model_dict)
    optimizer.load_state_dict(optimizer_dict)
    return iteration
