"""Train the 1D-CNN with several random seeds on DS1 and save the seed ensemble. DS2 is never used here."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse
import time

import numpy as np

from ecgsuite.cnn import save_checkpoint, train
from ecgsuite.config import load_config
from ecgsuite.splits import ds1_split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("--config", "--dataset", "--model"):
        p.add_argument(k)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--select", nargs="+", help="classes whose mean validation F1 picks the epoch (default: N S V)")
    a = p.parse_args()
    cfg = load_config(a.config, dataset=a.dataset, model=a.model, epochs=a.epochs, seeds=a.seeds, select_on=a.select)
    ds = dict(np.load(cfg["dataset"]))
    train_mask, val_mask = ds1_split(ds, cfg["validation_fraction"], cfg["validation_seed"])
    print("Validation patients (DS1):", sorted(np.unique(ds["record"][val_mask]).tolist()))
    select = tuple(cfg["select_on"])
    nets = []
    for s in cfg["seeds"]:
        t0 = time.time()
        net, std, best = train(ds, train_mask, val_mask, epochs=cfg["epochs"], batch=cfg["batch_size"],
                               lr=cfg["learning_rate"], seed=s, select=select, log=lambda msg: None)
        nets.append(net)
        print(f"seed {s}: best validation F1({','.join(select)}) = {best:.1f} | {time.time() - t0:.0f} s")
    os.makedirs(os.path.dirname(cfg["model"]), exist_ok=True)
    save_checkpoint(cfg["model"], nets, std, seeds=list(cfg["seeds"]), select=",".join(select))
    print(f"Saved a {len(nets)}-member ensemble -> {cfg['model']}")


if __name__ == "__main__":
    main()
