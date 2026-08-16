import time
import json
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path

import yaml
import torch
import numpy as np

from cs336_basics.models.transformers import TransformerLM
from cs336_basics.models.optimizers import AdamW, SGD
from cs336_basics.models.utils import cross_entropy_loss, lr_scheduler, gradient_clip
from cs336_basics.models.data import get_batch, save_checkpoint, load_checkpoint


DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        raise ValueError("float16 needs a GradScaler to not blow up into NaNs, use bfloat16")
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


def sync(device: torch.device):
    """Waits for the queued kernels, otherwise we time the queueing and not the work."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def git_commit() -> str | None:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return head.stdout.strip() if head.returncode == 0 else None


def parse_value(text: str):
    """yaml.safe_load reads 3e-4 as a string (YAML 1.1 wants 3.0e-4), so numbers go first."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return yaml.safe_load(text)


def apply_overrides(config: dict, pairs: list[str]):
    """Applies CLI overrides like optimizer.adamw.lr=3e-4 in place."""
    for pair in pairs:
        key, value = pair.split("=", 1)
        *path, leaf = key.split(".")
        node = config
        for part in path:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"Unknown config key: {key}")
            node = node[part]
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[leaf] = parse_value(value)


def name_from_overrides(pairs: list[str]) -> str:
    """optimizer.adamw.lr=3e-4 data.batch_size=64 -> lr3e-4_batch_size64"""
    parts = []
    for pair in pairs:
        key, value = pair.split("=", 1)
        leaf = key.split(".")[-1]
        if "/" in value or leaf in ("run_dir", "run_name"):  # paths make lousy names
            continue
        value = "".join(c for c in value if c.isalnum() or c in ".+-")
        parts.append(f"{leaf}{value}")
    return "_".join(parts)[:100]


@torch.no_grad()
def evaluate(
        model,
        data,
        *,
        batch_size: int,
        context_length: int,
        n_batches: int,
        device: torch.device,
        dtype: torch.dtype,
        use_amp: bool,
) -> float:
    model.eval()
    rng = np.random.default_rng(0)  # fixed: every eval of every run sees the same batches
    total = 0.0
    for _ in range(n_batches):
        xs, ys = get_batch(data, batch_size, context_length, device, rng=rng)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            total += cross_entropy_loss(model(xs), ys).item()
    model.train()
    return total / n_batches


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
        self.log_path = run_dir / "log.jsonl"

        config["meta"] = {
            "commit": git_commit(),
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

        self.best_loss = float("inf")
        self.t0 = time.perf_counter()
        self.eval_time = 0.0
        print(f"run dir: {self.dir}")

    def start_clock(self):
        """Call right before the loop, so wall_clock excludes model init and warmup."""
        self.t0 = time.perf_counter()

    def log(self, record: dict):
        elapsed = time.perf_counter() - self.t0
        record = {"wall_clock": elapsed, "train_time": elapsed - self.eval_time, **record}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _save(self, model, optimizer, step, path: Path):
        tmp = path.with_suffix(".tmp")
        save_checkpoint(model, optimizer, step, out=tmp, config=self.config)
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


def main(config: dict):
    device = torch.device(config["runtime"]["device"])
    autocast_dtype = resolve_dtype(config["runtime"]["dtype"])
    use_amp = autocast_dtype != torch.float32

    run = Run(config)  # after the config checks, so a typo leaves no empty run dir
    cfg = config["train"]

    seed = cfg["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = TransformerLM(**config["model"]).to(device=device)
    model.train()
    optimizer = build_optimizer(model, config["optimizer"])

    train_data = np.load(config["data"]["train_path"], mmap_mode="r")
    val_data = np.load(config["data"]["val_path"], mmap_mode="r")

    start_step = 0
    if cfg["resume"]:
        start_step = load_checkpoint(cfg["resume"], model, optimizer)
        print(f"resumed from {cfg['resume']} at step {start_step}")

    num_steps = cfg["num_steps"]
    sched = cfg["lr_schedule"]
    lr_max = optimizer.param_groups[0]["lr"]
    lr_min = lr_max * sched["lr_min_ratio"]
    t_c = sched["t_c"] or num_steps

    batch_size = config["data"]["batch_size"]
    context_length = config["model"]["context_length"]
    tokens_per_step = batch_size * context_length

    log_every = cfg["log_every"]
    eval_every = cfg["eval_every"]
    ckpt_every = cfg["ckpt_every"]
    losses = deque(maxlen=log_every)
    norms = deque(maxlen=log_every)

    eval_kwargs = dict(
        batch_size=batch_size,
        context_length=context_length,
        n_batches=cfg["eval_batches"],
        device=device,
        dtype=autocast_dtype,
        use_amp=use_amp,
    )

    sync(device)
    run.start_clock()

    for step in range(start_step, num_steps):
        lr = lr_scheduler(step, lr_max, lr_min, sched["t_warmup"], t_c)
        for group in optimizer.param_groups:
            group["lr"] = lr

        xs, ys = get_batch(train_data, batch_size, context_length, device, rng=rng)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            logits = model(xs)
            loss = cross_entropy_loss(logits, ys)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = gradient_clip(model.parameters(), cfg["max_l2_norm"])
        optimizer.step()

        losses.append(loss.detach())
        norms.append(norm)

        done = step + 1  # steps completed, so the axes of every log record agree
        tokens = done * tokens_per_step

        if done % log_every == 0:
            avg = torch.stack(list(losses)).mean().item()
            grad_norm = torch.stack(list(norms)).mean().item()
            run.log({"step": done, "tokens": tokens, "train_loss": avg, "lr": lr, "grad_norm": grad_norm})
            print(f"step {done:>6} | loss {avg:.4f} | lr {lr:.2e} | |g| {grad_norm:.2f}")

        if done % eval_every == 0 or done == num_steps:
            sync(device)
            eval_start = time.perf_counter()
            val_loss = evaluate(model, val_data, **eval_kwargs)
            run.eval_time += time.perf_counter() - eval_start
            run.log({"step": done, "tokens": tokens, "val_loss": val_loss})
            print(f"step {done:>6} | val loss {val_loss:.4f}")
            run.save_best(model, optimizer, done, val_loss)

        if done % ckpt_every == 0:
            run.save_periodic(model, optimizer, done)

    run.save_final(model, optimizer, num_steps)
    print(f"done in {time.perf_counter() - run.t0:.1f}s | best val loss: {run.best_loss:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/train_config.yaml")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="config overrides, e.g. --set optimizer.adamw.lr=3e-4 data.batch_size=64",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    apply_overrides(config, args.set)
    if not config["train"].get("run_name"):
        config["train"]["run_name"] = name_from_overrides(args.set)

    main(config)
