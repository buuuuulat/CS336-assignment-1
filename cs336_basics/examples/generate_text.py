import pickle
from pathlib import Path

import torch

from cs336_basics.train import paint
from cs336_basics.models.utils import decode
from cs336_basics.models.transformers import TransformerLM
from cs336_basics.tokenizer.tokenizer import Tokenizer


def latest_run(run_dir: str = "./runs") -> Path:
    runs = [p for p in Path(run_dir).iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs in {run_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def load(ckpt_path, tokenizer_dir, device) -> tuple[TransformerLM, Tokenizer, int]:
    """The checkpoint carries its own config, so the shape of the model comes from it and not from a yaml."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt["config"]["model"]
    model = TransformerLM(**model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with open(Path(tokenizer_dir) / "ints_to_tokens.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(Path(tokenizer_dir) / "merges.pkl", "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    print(paint(f"{ckpt_path} · step {ckpt['iteration']} · {device}", "dim"))
    return model, tokenizer, model_cfg["context_length"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="Once upon a time")
    parser.add_argument("--ckpt", default=None, help="default: best.pt of the newest run")
    parser.add_argument("--tokenizer", default="./outputs/bpe_tinystories")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    ckpt_path = args.ckpt or latest_run() / "best.pt"
    model, tokenizer, context_length = load(ckpt_path, args.tokenizer, args.device)

    # RoPE tables end at context_length, a longer sequence would index past them
    budget = context_length - len(tokenizer.encode(args.prompt))
    if budget <= 0:
        raise ValueError(f"Prompt alone fills the {context_length} token context")
    max_new_tokens = min(args.max_new_tokens, budget)

    for _ in range(args.samples):
        text = decode(
            args.prompt,
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
        head, tail = (args.prompt, text[len(args.prompt):]) if text.startswith(args.prompt) else ("", text)
        print(paint(head, "dim") + tail + "\n")
        print(f"Generated length: {len(tail)}")
