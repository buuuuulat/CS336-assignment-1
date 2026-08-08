import yaml


with open("./cs336_basics/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)
