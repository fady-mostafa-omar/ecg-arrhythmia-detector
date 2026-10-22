"""Evaluate a checkpoint on the 22 unseen DS2 patients: beat level, R-peak detection and end to end."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd
import torch

from ecgsuite.beats import AAMI, CLASSES, DS2, FS, load_record
from ecgsuite.cnn import Ensemble, load_checkpoint, per_record_s, predict
from ecgsuite.config import load_config
from ecgsuite.detect import detect_r_peaks
from ecgsuite.features import aami_report, rr_features
from ecgsuite.pipeline import analyze, match_to_expert

KEEP = ["N", "S", "V", "F"]
METRICS = ["Se %", "PPV %", "F1 %"]
COLORS = ["#2e9e5b", "#f39c12", "#2563eb", "#8e44ad", "#7f8c8d"]


def match_counts(ref, det, tol):
    """One-to-one matching of detections to reference beats within tol samples -> (tp, fp, fn)."""
    det = np.sort(np.asarray(det))
    used = np.zeros(len(det), bool)
    tp = 0
    for r, j in zip(ref, np.searchsorted(det, ref)):
        cand = [k for k in (j - 1, j) if 0 <= k < len(det) and not used[k] and abs(int(det[k]) - int(r)) <= tol]
        if cand:
            used[min(cand, key=lambda k: abs(int(det[k]) - int(r)))] = True
            tp += 1
    return tp, int(len(det) - used.sum()), int(len(ref) - tp)


def load_test(cfg, tol):
    """Raw DS2 signals, cardiologists' beats and R-peak detections (cached in cfg['rpeaks'])."""
    cache = dict(np.load(cfg["rpeaks"])) if os.path.exists(cfg["rpeaks"]) else {}
    new, test, rows = False, {}, []
    for r in DS2:
        sig, samples, symbols = load_record(cfg["data_dir"], r)
        m = np.isin(symbols, list(AAMI))
        ref, lab = samples[m], np.array([AAMI[s] for s in symbols[m]])
        if f"r{r}" not in cache:
            cache[f"r{r}"], new = detect_r_peaks(sig), True
        peaks = cache[f"r{r}"]
        tp, fp, fn = match_counts(ref, peaks, tol)
        rows.append({"record": int(r), "expert beats": len(ref), "missed": fn, "extra": fp,
                     "Se %": round(100 * tp / max(tp + fn, 1), 2), "PPV %": round(100 * tp / max(tp + fp, 1), 2)})
        test[r] = (sig, ref, lab, peaks)
    if new:
        os.makedirs(os.path.dirname(cfg["rpeaks"]), exist_ok=True)
        np.savez(cfg["rpeaks"], **cache)
    return test, pd.DataFrame(rows).set_index("record")


def end_to_end(test, net, std, tol):
    yt, yp, rec = [], [], []
    for r, (sig, ref, lab, peaks) in test.items():
        res = analyze(sig, FS, net, std, peaks=peaks)
        nearest, ok = match_to_expert(res["peaks"], ref, tol)
        yt.append(lab[nearest[ok]]); yp.append(res["labels"][ok]); rec.append(np.full(int(np.sum(ok)), int(r)))
    return np.concatenate(yt), np.concatenate(yp), np.concatenate(rec)


def confusion_png(yt, yp, path):
    cm = np.zeros((5, 5), int)
    np.add.at(cm, (yt, yp), 1)
    frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(frac, cmap="Reds", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{100 * frac[i, j]:.1f}%\n({cm[i, j]:,})", ha="center", va="center", fontsize=8,
                    color="white" if frac[i, j] > 0.5 else "black")
    ax.set_xticks(range(5), CLASSES)
    ax.set_yticks(range(5), CLASSES)
    ax.set(xlabel="model", ylabel="cardiologists", title="End to end on DS2 (row percentages)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def strip_png(test, net, std, record, path, tol):
    """10 s of an unseen patient on ECG paper: dots = model, letters = cardiologists (red = disagreement)."""
    sig, ref, lab, peaks = test[record]
    res = analyze(sig, FS, net, std, peaks=peaks)
    x, p, pl = res["signal"], res["peaks"], res["labels"]
    nearest, ok = match_to_expert(p, ref, tol)
    ab = np.where(pl != 0)[0]
    t0 = float(np.floor(max(0.0, p[ab[0]] / FS - 3))) if len(ab) else 0.0
    a, b = int(t0 * FS), int((t0 + 10) * FS)
    yl = float(np.clip(np.ceil(np.percentile(np.abs(x), 99.9) / 0.5) * 0.5 + 0.5, 1.5, 4.0))
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_facecolor("#fff6f6")
    ax.plot(np.arange(a, b) / FS, x[a:b], color="#111", lw=1.0)
    inw = np.where((p >= a) & (p < b))[0]
    for i in inw:
        ax.plot(p[i] / FS, x[p[i]], "o", ms=8, color=COLORS[pl[i]], mec="white", zorder=3)
        if ok[i]:
            e = lab[nearest[i]]
            ax.text(p[i] / FS, yl - 0.3, CLASSES[e], ha="center", fontsize=9, fontweight="bold",
                    color="#111" if e == pl[i] else "#c0392b")
    ax.set_xlim(t0, t0 + 10)
    ax.set_ylim(-yl, yl)
    for axis, major, minor in ((ax.xaxis, 0.2, 0.04), (ax.yaxis, 0.5, 0.1)):
        axis.set_major_locator(MultipleLocator(major))
        axis.set_minor_locator(MultipleLocator(minor))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}" if abs(v - round(v)) < 1e-6 else ""))
    ax.grid(which="major", color=(0.86, 0.24, 0.24, 0.55), lw=0.8)
    ax.grid(which="minor", color=(0.86, 0.24, 0.24, 0.18), lw=0.4)
    ax.tick_params(which="both", length=0)
    ax.set_axisbelow(True)
    shown = sorted(set(pl[inw].tolist()))
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COLORS[k], label=f"model: {CLASSES[k]}") for k in shown],
              loc="lower right", fontsize=8)
    ax.set(xlabel="time (s)", ylabel="mV",
           title=f"Patient {record} (never seen in training): dots = model, letters = cardiologists (red = disagreement)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("--config", "--data_dir", "--dataset", "--rpeaks", "--model", "--results"):
        p.add_argument(k)
    a = p.parse_args()
    cfg = load_config(a.config, data_dir=a.data_dir, dataset=a.dataset, rpeaks=a.rpeaks, model=a.model, results=a.results)
    out = cfg["results"]
    os.makedirs(out, exist_ok=True)
    tol = int(cfg["match_tolerance_s"] * FS)
    net, std = load_checkpoint(cfg["model"])
    meta = torch.load(cfg["model"], map_location="cpu")
    members = list(net.members) if isinstance(net, Ensemble) else [net]
    seeds = list(meta.get("seeds", range(len(members))))
    if len(seeds) != len(members):
        seeds = list(range(len(members)))

    # 1) beat level: windows at the cardiologists' beat positions
    ds = dict(np.load(cfg["dataset"]))
    te = ds["split"] == "DS2"
    w, r = std(ds["X"][te], rr_features(ds["rr"][te]))
    t0 = time.time()
    pred = predict(net, w, r)
    us = 1e6 * (time.time() - t0) / te.sum()
    beat = aami_report(ds["y"][te], pred)
    beat.to_csv(os.path.join(out, "cnn_ds2.csv"))
    print(f"Beat level on DS2 ({len(members)}-member model):")
    print(beat.loc[KEEP].to_string())

    # 2) R-peak detection
    test, rp = load_test(cfg, tol)
    rp.to_csv(os.path.join(out, "rpeak_ds2.csv"))
    tot = rp[["expert beats", "missed", "extra"]].sum()
    det_tp = tot["expert beats"] - tot["missed"]
    print(f"\nR-peak detection on DS2: Se {100 * det_tp / tot['expert beats']:.2f}% | "
          f"PPV {100 * det_tp / (det_tp + tot['extra']):.2f}% ({tot['missed']} missed, {tot['extra']} extra)")

    # 3) end to end: raw ECG -> detector -> model
    yt, yp, rec = end_to_end(test, net, std, tol)
    e2e = aami_report(yt, yp)
    e2e.to_csv(os.path.join(out, "end_to_end_ds2.csv"))
    print("\nEnd to end on DS2:")
    print(e2e.loc[KEEP].to_string())
    s_rec = pd.DataFrame(per_record_s(yt, yp, rec))
    s_rec["share of all S %"] = (100 * s_rec["S beats"] / max(int((yt == 1).sum()), 1)).round(1)
    s_rec.sort_values("S beats", ascending=False).to_csv(os.path.join(out, "s_per_record.csv"), index=False)
    wrong = yp[(yt == 0) & (yp != 0)]
    to = {CLASSES[k]: int(n) for k, n in enumerate(np.bincount(wrong, minlength=5)) if k}

    # 4) every ensemble member on its own: mean +- SD over seeds
    if len(members) > 1:
        rows = []
        for s, m in zip(seeds, members):
            m.eval()
            reps = {"beat": aami_report(ds["y"][te], predict(m, w, r)),
                    "end-to-end": aami_report(*end_to_end(test, m, std, tol)[:2])}
            for level, rep in reps.items():
                rows += [{"seed": s, "level": level, "class": c, **{k: float(rep.loc[c, k]) for k in METRICS}}
                         for c in KEEP]
        sd = pd.DataFrame(rows)
        sd.to_csv(os.path.join(out, "seeds_ds2.csv"), index=False)
        pm = lambda g: f"{g.mean():.1f} ± {g.std(ddof=1):.1f}"
        summary = sd.groupby(["level", "class"], sort=False)[METRICS].agg(pm)
        summary.to_csv(os.path.join(out, "seeds_summary.csv"))
        print(f"\nSingle members, mean ± SD over {len(members)} seeds:")
        print(summary.to_string())

    info = {"members": len(members), "seeds": seeds,
            "parameters": int(sum(q.numel() for q in net.parameters())),
            "parameters_per_member": int(sum(q.numel() for q in members[0].parameters())),
            "us_per_beat_cpu": round(us, 1), "selection_rule": meta.get("select", "N,S,V"),
            "normal_beats": int((yt == 0).sum()), "normal_mislabeled": int(len(wrong)),
            "normal_mislabeled_pct": round(100 * len(wrong) / max(int((yt == 0).sum()), 1), 1),
            "normal_mislabeled_to": to,
            "detector": {k.replace(" ", "_"): int(v) for k, v in tot.items()}}
    with open(os.path.join(out, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    confusion_png(yt, yp, os.path.join(out, "confusion_end_to_end.png"))
    strip_png(test, net, std, str(cfg["example_record"]), os.path.join(out, "example_strip.png"), tol)
    print(f"\n{info['parameters']:,} parameters | {us:.0f} µs per beat (CPU) | results -> {out}")


if __name__ == "__main__":
    main()
