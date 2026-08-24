import yaml

from cs336_basics.train import main as train
from cs336_basics.train import apply_overrides, name_from_overrides

LEARNING_RATES = [6e-2, 6e-3, 1e-3, 6e-4, 1e-4]


if __name__ == '__main__':
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

    for lr in LEARNING_RATES:
        apply_overrides(config, [f"optimizer.adamw.lr={lr}"])
        if not config["train"].get("run_name"):
            config["train"]["run_name"] = name_from_overrides(args.set)
        train(config, break_val_diverge=3)
