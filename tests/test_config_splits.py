import os
import numpy as np
from ecgsuite.config import ROOT, load_config
from ecgsuite.splits import ds1_split


def test_default_config():
    cfg = load_config()
    assert cfg["seeds"] == [0, 1, 2, 3, 4] and cfg["select_on"] == ["N", "S", "V"]
    assert os.path.isabs(cfg["model"]) and cfg["model"].startswith(ROOT)
    assert load_config(model="/tmp/x.pt")["model"] == "/tmp/x.pt"


def test_ds1_split_is_patient_wise():
    rec = np.repeat(np.arange(10), 20)
    split = np.where(rec < 7, "DS1", "DS2")
    ds = {"y": np.zeros(len(rec), np.int8), "record": rec, "split": split}
    tr, va = ds1_split(ds, fraction=0.3, seed=0)
    assert not (tr & va).any() and not ((tr | va) & (split == "DS2")).any()
    assert set(rec[tr]).isdisjoint(set(rec[va])) and (tr | va).sum() == (split == "DS1").sum()
