"""Shared utilities: config loading and reproducible seeding."""
import random
import numpy as np
import yaml


def load_config(path):
    """Load a YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Seed Python and NumPy so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
