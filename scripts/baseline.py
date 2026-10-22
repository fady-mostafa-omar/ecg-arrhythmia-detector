"""Classical baselines: RR-interval and morphology features, trained on DS1 and tested once on DS2."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecgsuite.config import load_config
from ecgsuite.features import aami_report, make_features

KEEP = ["N", "S", "V", "F"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("--config", "--dataset", "--results"):
        p.add_argument(k)
    a = p.parse_args()
    cfg = load_config(a.config, dataset=a.dataset, results=a.results)
    ds = dict(np.load(cfg["dataset"]))
    Xf, y = make_features(ds), ds["y"]
    tr, te = ds["split"] == "DS1", ds["split"] == "DS2"
    models = {"LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
              "Gradient boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=0)}
    rows = []
    for name, model in models.items():
        model.fit(Xf[tr], y[tr])
        rep = aami_report(y[te], model.predict(Xf[te]))
        print(f"\n{name} on DS2 (Se / PPV / F1 %)")
        print(rep.loc[KEEP].to_string())
        rows += [{"model": name, "class": c, **{k: float(rep.loc[c, k]) for k in ["n_test", "Se %", "PPV %", "F1 %"]}}
                 for c in KEEP]
    os.makedirs(cfg["results"], exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(cfg["results"], "baseline_ds2.csv"), index=False)


if __name__ == "__main__":
    main()
