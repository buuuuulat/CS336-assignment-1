import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # we write files, no windows to pop up

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# how to draw a given metric: color, scale, whether to smooth the noise
STYLE = {
    "train_loss": dict(label="Train loss",     color="#2f6fdb", yscale="log",    smooth=0.6),
    "val_loss":   dict(label="Val loss",       color="#d1495b", yscale="log",    smooth=0.0),
    "lr":         dict(label="Learning rate",  color="#3f8f5c", yscale="linear", smooth=0.0),
    "grad_norm":  dict(label="Grad norm",      color="#b07d2b", yscale="linear", smooth=0.8),
    "wall_clock": dict(label="Wall clock, s",  color="#6c6f7a", yscale="linear", smooth=0.0),
    "train_time": dict(label="Train time, s",  color="#6c6f7a", yscale="linear", smooth=0.0),
}
AXIS_LABEL = {"step": "Step", "tokens": "Tokens", "wall_clock": "Wall clock, s", "train_time": "Train time, s"}
RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelcolor": "#4a4d55",
    "xtick.color": "#8b8e96",
    "ytick.color": "#8b8e96",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 10,
    "legend.frameon": False,
}


def read_log(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def ema(ys, alpha):
    out, acc = [], ys[0]
    for y in ys:
        acc = alpha * acc + (1 - alpha) * y
        out.append(acc)
    return out


def si(v, _=None):
    """1234567 -> 1.23M"""
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= div:
            return f"{v / div:.3g}{suf}"
    return f"{v:.3g}"


def fmt_value(v):
    return f"{v:.3e}" if 0 < abs(v) < 1e-3 else f"{v:,.4g}"


def series(rows, x, metric):
    """(x, metric) points in log order, None when there is nothing to draw."""
    pts = [(r[x], r[metric]) for r in rows if x in r and isinstance(r.get(metric), (int, float))]
    if len(pts) < 2:
        return None
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def draw(ax, xs, ys, st, dy=10):
    """Raw line, an EMA on top of it for noisy metrics, and the last value annotated."""
    ax.plot(xs, ys, lw=1.4, color=st["color"], alpha=0.28 if st["smooth"] else 1.0,
            label=None if st["smooth"] else st["label"], zorder=2)
    if st["smooth"]:
        ax.plot(xs, ema(ys, st["smooth"]), lw=2.0, color=st["color"],
                label=f"{st['label']}, EMA {st['smooth']}", zorder=3)
    ax.scatter([xs[-1]], [ys[-1]], s=26, color=st["color"], zorder=4)
    ax.annotate(fmt_value(ys[-1]), (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(-6, dy), ha="right", fontsize=9, color=st["color"], fontweight="bold")


def save(fig, ax, path, title, x, logscale, legend):
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(AXIS_LABEL.get(x, x))
    ax.xaxis.set_major_formatter(FuncFormatter(si))
    ax.margins(x=0.02)
    ax.grid(True, which="major", axis="both", lw=0.6, color="#e3e5ea", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d0d3d9")
    if logscale:
        ax.set_yscale("log")
    if legend:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)  # otherwise the figures pile up in memory over a batch of runs


def plot_run(run_dir, x: str = "step", metrics: list[str] | None = None) -> list[Path]:
    """Draws one chart per metric found in log.jsonl, saves the png files to <run>/plots."""
    run_dir = Path(run_dir)
    rows = read_log(run_dir / "log.jsonl")
    metrics = metrics or [m for m in STYLE if m != x and any(m in r for r in rows)]

    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    suffix = "" if x == "step" else f"_by_{x}"
    saved = []

    with plt.rc_context(RC):
        for m in metrics:
            data = series(rows, x, m)
            if data is None:
                continue
            st = STYLE.get(m, dict(label=m, color="#2f6fdb", yscale="linear", smooth=0.0))
            fig, ax = plt.subplots(figsize=(8, 4.6))
            draw(ax, *data, st)
            path = out_dir / f"{m}{suffix}.png"
            save(fig, ax, path, st["label"], x, st["yscale"] == "log" and min(data[1]) > 0, st["smooth"] > 0)
            saved.append(path)

        both = [series(rows, x, m) for m in ("train_loss", "val_loss")]
        if all(both):  # both curves on the same axes, the one that goes into the writeup
            fig, ax = plt.subplots(figsize=(8, 4.6))
            for data, m, dy in zip(both, ("train_loss", "val_loss"), (10, -16)):
                draw(ax, *data, STYLE[m], dy)  # the end labels are split apart, they overlap otherwise
            path = out_dir / f"loss{suffix}.png"
            save(fig, ax, path, "Train / val loss", x, min(min(d[1]) for d in both) > 0, True)
            saved.append(path)

    return saved


def all_runs(root="./runs") -> list[Path]:
    return sorted(p for p in Path(root).iterdir() if (p / "log.jsonl").exists())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", help="run dirs, defaults to every run in ./runs")
    parser.add_argument("--root", default="./runs")
    parser.add_argument("--x", default="step", choices=list(AXIS_LABEL))
    args = parser.parse_args()

    for run in [Path(r) for r in args.runs] or all_runs(args.root):
        saved = plot_run(run, x=args.x)
        print(f"{run}: {len(saved)} png -> {run / 'plots'}")
