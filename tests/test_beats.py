import os, sys, tempfile
import numpy as np
import wfdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ecgsuite import beats as B


def make_record(folder, rec, swap=False, n=36000):
    rng = np.random.default_rng(int(rec))
    sig = rng.normal(0, 0.05, (n, 2))
    r = np.arange(200, n - 200, 300)                       # R peaks every 300 samples
    sig[r, 0 if not swap else 1] += 1.0                    # spikes on the MLII lead only
    names = ["MLII", "V1"] if not swap else ["V5", "MLII"]
    wfdb.wrsamp(rec, fs=360, units=["mV", "mV"], sig_name=names, p_signal=sig,
                fmt=["212", "212"], write_dir=folder)
    syms = np.array(list("NVAFQ+") * (len(r) // 6 + 1))[: len(r)]
    syms = np.where(syms == "Q", "/", syms)               # '/' -> Q class
    wfdb.wrann(rec, "atr", r, syms, write_dir=folder)
    return r, syms


def test_aami_and_splits():
    assert set(B.DS1).isdisjoint(B.DS2) and len(B.DS1) == len(B.DS2) == 22
    all48 = {str(r) for r in list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))
             + [200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219,
                220, 221, 222, 223, 228, 230, 231, 232, 233, 234]}
    assert set(B.DS1) | set(B.DS2) | set(B.PACED) == all48
    assert B.AAMI["L"] == 0 and B.AAMI["a"] == 1 and B.AAMI["E"] == 2 and B.AAMI["f"] == 4
    assert "+" not in B.AAMI and "~" not in B.AAMI


def test_extract_and_build():
    d = tempfile.mkdtemp()
    for rec in B.DS1 + B.DS2:
        make_record(d, rec, swap=(rec == "114"))
    X, rr, y = B.extract_record(d, "114")
    assert X.shape[1] == B.PRE + B.POST and X.dtype == np.float32
    # MLII picked even when it is the second channel: the R spike sits at index PRE
    assert np.argmax(np.abs(X).mean(0)) == B.PRE
    assert set(np.unique(y)) <= {0, 1, 2, 3, 4} and 1 in y and 4 in y
    assert np.allclose(rr[5, :2], 300 / 360, atol=1e-4)      # pre/post RR of a regular rhythm
    ds = B.build_dataset(d)
    n = len(ds["y"])
    assert all(len(v) == n for v in ds.values())
    assert set(np.unique(ds["split"])) == {"DS1", "DS2"}
    assert not set(ds["record"].astype(str)) & set(B.PACED)
