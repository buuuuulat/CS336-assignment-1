import time
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import yaml
import torch
import numpy as np

from cs336_basics.models.transformers import TransformerLM
from cs336_basics.models.optimizers import AdamW, SGD
from cs336_basics.models.utils import cross_entropy_loss, lr_scheduler, gradient_clip
from cs336_basics.models.data import get_batch, save_checkpoint


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in DTYPES:
        raise ValueError(f"Unknown dtype {name}, available: {list(DTYPES)}")
    return DTYPES[name]


def build_optimizer(model, optim_cfg) -> torch.optim.Optimizer:
    registry = {"adamw": AdamW, "sgd": SGD}
    optim_name = optim_cfg["name"]
    if optim_name not in registry:
        raise ValueError(f"Unknown optimizer {optim_name}, available: {list(registry)}")
    if optim_name not in optim_cfg:
        raise ValueError(f"No '{optim_name}' section in optimizer config")
    return registry[optim_name](model.parameters(), **optim_cfg[optim_name])


class Run:
    def __init__(self, config: dict):
        self.config = config
        cfg = config["train"]

        root = Path(cfg["run_dir"])
        name = cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")

        run_dir = root / name
        n = 1
        while run_dir.exists():
            run_dir = root / f"{name}_{n}"
            n += 1

        self.dir = run_dir
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True)

        with open(self.dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

        self.history = []
        self.best_loss = float("inf")
        print(f"run dir: {self.dir}")

    def log(self, record: dict):
        self.history.append(record)
        tmp = self.dir / "log.json.tmp"
        with open(tmp, "w") as f:
            json.dump(self.history, f, indent=2)
        tmp.replace(self.dir / "log.json")

    def _save(self, model, optimizer, step, path: Path):
        tmp = path.with_suffix(".tmp")
        save_checkpoint(model, optimizer, step, self.config, out=tmp)
        tmp.replace(path)

    def save_periodic(self, model, optimizer, step):
        self._save(model, optimizer, step, self.ckpt_dir / f"step_{step:06d}.pt")
        self._save(model, optimizer, step, self.dir / "last.pt")
        self._rotate()

    def save_best(self, model, optimizer, step, loss: float):
        if loss >= self.best_loss:
            return
        self.best_loss = loss
        self._save(model, optimizer, step, self.dir / "best.pt")

    def save_final(self, model, optimizer, step):
        self._save(model, optimizer, step, self.dir / "final.pt")

    def _rotate(self):
        keep = self.config["train"].get("keep_last")
        if not keep:
            return
        files = sorted(self.ckpt_dir.glob("step_*.pt"))
        for f in files[:-keep]:
            f.unlink()


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    run = Run(config)

    seed = config["train"]["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device(config["runtime"]["device"])
    autocast_dtype = resolve_dtype(config["runtime"]["dtype"])
    use_amp = autocast_dtype != torch.float32

    model = TransformerLM(**config["model"]).to(device=device)
    model.train()
    optimizer = build_optimizer(model, config["optimizer"])

    train_data = np.load(config["data"]["path"], mmap_mode="r")

    num_steps = config["train"]["num_steps"]
    sched = config["train"]["lr_schedule"]
    lr_max = optimizer.param_groups[0]["lr"]
    lr_min = lr_max * sched["lr_min_ratio"]
    t_c = sched["t_c"] or num_steps

    log_every = config["train"]["log_every"]
    ckpt_every = config["train"]["ckpt_every"]
    window = deque(maxlen=log_every)

    for step in range(num_steps):
        lr = lr_scheduler(step, lr_max, lr_min, sched["t_warmup"], t_c)
        for group in optimizer.param_groups:
            group["lr"] = lr

        xs, ys = get_batch(
            train_data,
            batch_size=config["data"]["batch_size"],
            context_length=config["model"]["context_length"],
            device=device,
            rng=rng,
        )

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            logits = model(xs)
            loss = cross_entropy_loss(logits, ys)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clip(model.parameters(), config["train"]["max_l2_norm"])
        optimizer.step()

        window.append(loss.detach())

        if (step + 1) % log_every == 0:
            avg = torch.stack(list(window)).mean().item()
            run.log({"step": step, "loss": avg, "lr": lr})
            print(f"step {step:>6} | loss {avg:.4f} | lr {lr:.2e}")
            run.save_best(model, optimizer, step, avg)

        if (step + 1) % ckpt_every == 0:
            run.save_periodic(model, optimizer, step)

    run.save_final(model, optimizer, num_steps)
    print(f"done. best loss: {run.best_loss:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/train_config.yaml")
    args = parser.parse_args()
    main(args.config)
