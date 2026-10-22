"""Extract every MIT-BIH beat (AAMI classes, DS1/DS2 inter-patient split) into one .npz file."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecgsuite.beats import CLASSES, FS, PRE, build_dataset
from ecgsuite.config import load_config

COLORS = ["#2e9e5b", "#f39c12", "#2563eb", "#8e44ad", "#7f8c8d"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("--config", "--data_dir", "--dataset", "--results"):
        p.add_argument(k)
    a = p.parse_args()
    cfg = load_config(a.config, data_dir=a.data_dir, dataset=a.dataset, results=a.results)
    ds = build_dataset(cfg["data_dir"])
    os.makedirs(os.path.dirname(cfg["dataset"]), exist_ok=True)
    np.savez_compressed(cfg["dataset"], **ds)

    counts = pd.crosstab(pd.Series(ds["split"], name="split"),
                         pd.Series([CLASSES[k] for k in ds["y"]], name="class")).reindex(columns=CLASSES, fill_value=0)
    counts["total"] = counts.sum(1)
    os.makedirs(cfg["results"], exist_ok=True)
    counts.to_csv(os.path.join(cfg["results"], "beat_counts.csv"))
    print(f"{len(ds['y']):,} beats from {len(np.unique(ds['record']))} records -> {cfg['dataset']}")
    print(counts.to_string())

    t = (np.arange(ds["X"].shape[1]) - PRE) / FS * 1000
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, c in enumerate(CLASSES[:4]):
        m = (ds["split"] == "DS1") & (ds["y"] == k)
        if m.sum() >= 10:
            ax.plot(t, ds["X"][m].mean(0), color=COLORS[k], lw=2, label=f"{c} (n={m.sum():,})")
    ax.axvline(0, color="#999", lw=0.8, ls="--")
    ax.set(xlabel="time from R peak (ms)", ylabel="mV", title="Mean beat per class (DS1, band-passed)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["results"], "mean_beats.png"), dpi=150)


if __name__ == "__main__":
    main()
