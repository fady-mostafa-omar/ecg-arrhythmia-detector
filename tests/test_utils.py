import numpy as np
from project_pkg.utils import load_config, set_seed


def test_config_loads():
    assert load_config("configs/default.yaml")["seed"] == 42


def test_seed_is_reproducible():
    set_seed(0)
    a = np.random.rand(3)
    set_seed(0)
    b = np.random.rand(3)
    assert np.allclose(a, b)
