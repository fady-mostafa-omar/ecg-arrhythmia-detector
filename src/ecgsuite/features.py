"""Hand-crafted beat features (RR timing + morphology) and AAMI-style evaluation."""
import numpy as np
import pandas as pd

from .beats import PRE, CLASSES


def rr_features(rr):
    """rr columns: pre, post, local mean, record mean (s). Adds patient-normalized ratios."""
    pre, post, loc, rec = rr.T
    return np.c_[pre, post, loc, rec, pre / loc, post / loc, pre / rec, post / rec, post / pre]


def morph_features(X):
    """Downsampled waveform plus QRS amplitude, energy and width around the R peak."""
    qrs = X[:, PRE - 25: PRE + 25]                       # +-70 ms around R
    peak = np.abs(qrs).max(1, keepdims=True)
    width = (np.abs(qrs) > 0.5 * peak).sum(1)            # samples above half of the peak
    return np.c_[X[:, ::6], qrs.max(1), qrs.min(1), np.ptp(qrs, 1), (qrs ** 2).sum(1), width]


def make_features(ds):
    return np.c_[rr_features(ds["rr"]), morph_features(ds["X"])].astype(np.float32)


def aami_report(y_true, y_pred):
    """Per-class sensitivity, positive predictivity and F1 (percent), as reported in AAMI papers."""
    rows = []
    for k, name in enumerate(CLASSES):
        tp = np.sum((y_pred == k) & (y_true == k))
        fn = np.sum((y_pred != k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        se = tp / (tp + fn) if tp + fn else np.nan
        ppv = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * se * ppv / (se + ppv) if se and ppv else 0.0
        rows.append({"class": name, "n_test": int(tp + fn), "Se %": 100 * se,
                     "PPV %": 100 * ppv, "F1 %": 100 * f1})
    df = pd.DataFrame(rows).set_index("class").round(1)
    df.attrs["accuracy"] = round(100 * float(np.mean(y_true == y_pred)), 1)
    return df
