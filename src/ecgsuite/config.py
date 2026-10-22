"""Load configs/default.yaml and resolve its paths against the repository root."""
import os

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PATH_KEYS = ("data_dir", "dataset", "rpeaks", "model", "results")


def load_config(path=None, **overrides):
    """Return the config as a dict; keyword overrides that are not None replace config values."""
    with open(path or os.path.join(ROOT, "configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    for k in PATH_KEYS:
        if k in cfg and not os.path.isabs(cfg[k]):
            cfg[k] = os.path.join(ROOT, cfg[k])
    return cfg
