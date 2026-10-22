"""R-peak detection for raw ECG (used by the app, where no expert labels exist)."""
import numpy as np
from wfdb import processing

from .beats import bandpass, FS


def detect_r_peaks(sig, fs=FS):
    """XQRS detection, then snap each detection to the largest |amplitude| within +-100 ms."""
    filt = bandpass(sig, fs=fs)
    xqrs = processing.XQRS(sig=filt, fs=fs)
    xqrs.detect(verbose=False)
    det = np.asarray(xqrs.qrs_inds, dtype=int)
    r = int(0.10 * fs)
    peaks = []
    for d in det:
        lo, hi = max(d - r, 0), min(d + r + 1, len(filt))
        peaks.append(lo + int(np.argmax(np.abs(filt[lo:hi]))))
    return np.unique(np.array(peaks, dtype=int))


def match_peaks(ref, det, tol):
    """One-to-one matching within tol samples. Returns tp, fp, fn and the offsets (det - ref)."""
    ref, det = np.sort(ref), np.sort(det)
    used = np.zeros(len(det), bool)
    offsets = []
    for r in ref:
        lo = np.searchsorted(det, r - tol, "left")
        hi = np.searchsorted(det, r + tol, "right")
        cand = [k for k in range(lo, hi) if not used[k]]
        if cand:
            k = min(cand, key=lambda k: abs(det[k] - r))
            used[k] = True
            offsets.append(det[k] - r)
    tp = len(offsets)
    return tp, int((~used).sum()), len(ref) - tp, np.array(offsets)
