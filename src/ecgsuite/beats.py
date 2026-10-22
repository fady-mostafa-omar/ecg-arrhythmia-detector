"""MIT-BIH beat extraction with AAMI labels and the de Chazal (2004) inter-patient split."""
import os
import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

FS = 360
PRE, POST = 90, 162                     # samples before / after the R peak (0.25 s / 0.45 s)
CLASSES = ["N", "S", "V", "F", "Q"]
AAMI = {**{s: 0 for s in "NLRej"},      # N: normal, bundle branch blocks, escape beats
        **{s: 1 for s in "AaJS"},       # S: supraventricular ectopic
        **{s: 2 for s in "VE"},         # V: ventricular ectopic
        "F": 3,                         # F: fusion
        **{s: 4 for s in "/fQ"}}        # Q: paced, fusion of paced, unclassifiable
PACED = ["102", "104", "107", "217"]    # excluded by the AAMI standard
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]


def bandpass(x, lo=0.5, hi=40.0, fs=FS, order=3):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def load_record(folder, rec):
    """Return the MLII lead (record 114 stores it in the second channel), R-peak samples, symbols."""
    r = wfdb.rdrecord(os.path.join(folder, rec), channel_names=["MLII"])
    a = wfdb.rdann(os.path.join(folder, rec), "atr")
    return r.p_signal[:, 0], np.asarray(a.sample), np.asarray(a.symbol)


def _local_mean(x, n=10):
    """Causal mean of the last n values of x (ignores the leading NaN)."""
    v = np.nan_to_num(x)
    c = np.cumsum(np.r_[0.0, v])
    cnt = np.cumsum(np.r_[0, ~np.isnan(x)])
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(i - n, 0)
    return (c[i] - c[lo]) / np.maximum(cnt[i] - cnt[lo], 1)


def extract_record(folder, rec):
    """Beat windows, RR features [pre, post, local mean, record mean] (s) and AAMI labels."""
    sig, samples, symbols = load_record(folder, rec)
    sig = bandpass(sig)
    is_beat = np.isin(symbols, list(AAMI))
    s, sym = samples[is_beat], symbols[is_beat]
    rr = np.diff(s) / FS
    pre, post = np.r_[np.nan, rr], np.r_[rr, np.nan]
    feats = np.c_[pre, post, _local_mean(pre), np.full(len(s), np.mean(rr))]
    keep = (s >= PRE) & (s + POST <= len(sig)) & ~np.isnan(pre) & ~np.isnan(post)
    X = sig[s[keep][:, None] + np.arange(-PRE, POST)].astype(np.float32)
    y = np.array([AAMI[c] for c in sym[keep]], dtype=np.int8)
    return X, feats[keep].astype(np.float32), y


def build_dataset(folder):
    parts = {"X": [], "rr": [], "y": [], "record": [], "split": []}
    for split, recs in (("DS1", DS1), ("DS2", DS2)):
        for rec in recs:
            X, rr, y = extract_record(folder, rec)
            parts["X"].append(X); parts["rr"].append(rr); parts["y"].append(y)
            parts["record"].append(np.full(len(y), int(rec), dtype=np.int16))
            parts["split"].append(np.full(len(y), split))
    return {k: np.concatenate(v) for k, v in parts.items()}
