import numpy as np
from ecgsuite.features import aami_report, rr_features


def test_rr_feature_ratios():
    rr = np.array([[0.8, 1.0, 0.9, 0.85], [1.0, 0.5, 1.0, 0.85]], dtype=np.float32)
    f = rr_features(rr)
    assert f.shape == (2, 9)
    assert np.isclose(f[0, 4], 0.8 / 0.9) and np.isclose(f[1, 8], 0.5 / 1.0)


def test_aami_report_counts():
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 2, 1, 0, 2, 2, 2])
    rep = aami_report(y_true, y_pred)
    assert rep.loc["N", "n_test"] == 3 and rep.loc["V", "n_test"] == 3
    assert np.isclose(rep.loc["S", "Se %"], 50.0) and np.isclose(rep.loc["S", "PPV %"], 100.0)
    assert np.isclose(rep.loc["V", "Se %"], 100.0) and np.isclose(rep.loc["V", "PPV %"], 75.0)
    assert np.isclose(rep.attrs["accuracy"], 75.0)
