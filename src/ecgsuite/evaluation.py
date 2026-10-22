"""Patient-wise feature normalization and leakage-safe model selection on DS1."""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .features import aami_report


def per_record_normalize(Xf, records):
    """Robust z-score of every feature within each record (median / IQR). Uses no labels."""
    out = np.empty_like(Xf)
    for r in np.unique(records):
        m = records == r
        med = np.median(Xf[m], 0)
        iqr = np.subtract(*np.percentile(Xf[m], [75, 25], 0))
        out[m] = (Xf[m] - med) / np.where(iqr > 1e-6, iqr, 1.0)
    return out


def grouped_cv_score(model, X, y, groups, n_splits=5):
    """Mean F1 of S and V across folds that never share a patient (DS1 only)."""
    scores = []
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        rep = aami_report(y[va], m.predict(X[va]))
        scores.append(np.nanmean(rep.loc[["S", "V"], "F1 %"]))
    return float(np.mean(scores)), float(np.std(scores))
