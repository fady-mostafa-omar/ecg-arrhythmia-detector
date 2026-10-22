"""Patient-wise split of DS1 into training and validation patients. DS2 is never touched."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def ds1_split(ds, fraction=0.2, seed=0):
    """Boolean masks (train, val) over all beats; validation is a fraction of the DS1 patients."""
    y, rec = ds["y"], ds["record"]
    idx1 = np.where(ds["split"] == "DS1")[0]
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
                  .split(idx1, y[idx1], rec[idx1]))
    train, val = np.zeros(len(y), bool), np.zeros(len(y), bool)
    train[idx1[tr]] = True
    val[idx1[va]] = True
    return train, val
