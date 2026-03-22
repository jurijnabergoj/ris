import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def resolve(p):
    p = Path(p)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def resolve_cfg_paths(cfg):
    cfg["data"]["data_dir"] = resolve(cfg["data"]["data_dir"])
    cfg["data"]["csv_path"] = resolve(cfg["data"]["csv_path"])
    cfg["data"]["test_dir"] = resolve(cfg["data"]["test_dir"])
    cfg["train"]["checkpoint_path"] = resolve(cfg["train"]["checkpoint_path"])
    return cfg
