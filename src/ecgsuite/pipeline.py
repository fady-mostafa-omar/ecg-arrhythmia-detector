"""End-to-end analysis of a raw ECG: detect beats, build features, classify with the CNN."""
from math import gcd
import numpy as np
import torch
from scipy.signal import resample_poly

from .beats import FS, PRE, POST, bandpass, _local_mean
from .features import rr_features
from .cnn import BeatNet, Standardizer
from .detect import detect_r_peaks


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    net = BeatNet(n_rr=ck["n_rr"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    std = Standardizer()
    std.__dict__.update({k: np.asarray(v, dtype=np.float32) for k, v in ck["std"].items()})
    return net, std


def to_360(sig, fs):
    sig = np.asarray(sig, dtype=float)
    fs = int(round(fs))
    if fs == FS:
        return sig
    g = gcd(FS, fs)
    return resample_poly(sig, FS // g, fs // g)


def beats_from_peaks(sig_f, peaks):
    """Beat windows and RR features from R peaks, with the same definitions used in training."""
    peaks = np.asarray(peaks, dtype=int)
    rr = np.diff(peaks) / FS
    pre, post = np.r_[np.nan, rr], np.r_[rr, np.nan]
    feats = np.c_[pre, post, _local_mean(pre), np.full(len(peaks), np.mean(rr))]
    keep = (peaks >= PRE) & (peaks + POST <= len(sig_f)) & ~np.isnan(pre) & ~np.isnan(post)
    X = sig_f[peaks[keep][:, None] + np.arange(-PRE, POST)].astype(np.float32)
    return peaks[keep], X, feats[keep].astype(np.float32)


def analyze(sig, fs, net, std, peaks=None):
    """Return the filtered signal (360 Hz), kept R peaks, predicted class, probabilities, RR (s)."""
    sig = to_360(sig, fs)
    if peaks is None:
        peaks = detect_r_peaks(sig)
    sig_f = bandpass(sig)
    p, X, rr = beats_from_peaks(sig_f, peaks)
    w, r = std(X, rr_features(rr))
    with torch.no_grad():
        probs = torch.softmax(net(w, r), 1).numpy()
    return {"signal": sig_f, "peaks": p, "labels": probs.argmax(1), "probs": probs, "rr": rr[:, 0]}


def match_to_expert(peaks, ref, tol):
    """Index of the nearest expert beat for every detected peak, and a mask of matches within tol."""
    ref = np.asarray(ref)
    j = np.clip(np.searchsorted(ref, peaks), 1, len(ref) - 1)
    left, right = ref[j - 1], ref[j]
    nearest = np.where(np.abs(peaks - left) <= np.abs(right - peaks), j - 1, j)
    return nearest, np.abs(ref[nearest] - peaks) <= tol
