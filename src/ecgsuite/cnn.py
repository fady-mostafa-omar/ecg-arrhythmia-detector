"""Compact 1D-CNN that fuses the beat waveform with RR-interval features, plus a seed ensemble."""
import time
import numpy as np
import torch
import torch.nn as nn

from .features import rr_features, aami_report


class BeatNet(nn.Module):
    def __init__(self, n_rr=9, n_classes=5):
        super().__init__()
        def block(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o),
                                 nn.ReLU(), nn.MaxPool1d(2))
        self.conv = nn.Sequential(block(1, 16, 7), block(16, 32, 5), block(32, 64, 5),
                                  nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(64 + n_rr, 64), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(64, n_classes))

    def forward(self, wave, rr):
        return self.head(torch.cat([self.conv(wave).flatten(1), rr], 1))


class Ensemble(nn.Module):
    """Averages the class probabilities of several BeatNets and returns log-probabilities,
    so softmax() of the output gives the averaged probabilities and argmax() the class."""
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, wave, rr):
        p = torch.stack([m(wave, rr).softmax(1) for m in self.members]).mean(0)
        return p.clamp_min(1e-8).log()


class Standardizer:
    """Global statistics computed on the training patients only."""
    def fit(self, wave, rr):
        self.w_mu, self.w_sd = wave.mean(), wave.std() + 1e-6
        self.r_mu, self.r_sd = rr.mean(0), rr.std(0) + 1e-6
        return self

    def __call__(self, wave, rr):
        w = ((wave - self.w_mu) / self.w_sd)[:, None, :].astype(np.float32)
        r = ((rr - self.r_mu) / self.r_sd).astype(np.float32)
        return torch.from_numpy(w), torch.from_numpy(r)


def predict(model, w, r, batch=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(w), batch):
            out.append(model(w[i:i + batch], r[i:i + batch]).argmax(1))
    return torch.cat(out).cpu().numpy()


def train(ds, train_mask, val_mask, epochs=15, batch=256, lr=1e-3, seed=0, device=None,
          select=("N", "S", "V"), log=print):
    """Train on train_mask beats; keep the epoch with the best mean F1 of the `select` classes on
    val_mask. N is included so that an epoch which mislabels many normal beats is not rewarded."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rr_all = rr_features(ds["rr"])
    std = Standardizer().fit(ds["X"][train_mask], rr_all[train_mask])
    wt, rt = std(ds["X"][train_mask], rr_all[train_mask])
    wv, rv = std(ds["X"][val_mask], rr_all[val_mask])
    yt = torch.from_numpy(ds["y"][train_mask].astype(np.int64))
    yv = ds["y"][val_mask]

    counts = np.bincount(ds["y"][train_mask], minlength=5).astype(float)
    weights = torch.tensor(1 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32)
    weights = weights / weights.sum() * 5

    model = BeatNet(n_rr=rt.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(yt))
        t0 = time.time()
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            w = wt[idx] * (1 + 0.2 * (torch.rand(len(idx), 1, 1) - 0.5))   # amplitude jitter
            w = w + 0.05 * torch.randn_like(w)                               # noise
            opt.zero_grad()
            loss = loss_fn(model(w.to(device), rt[idx].to(device)), yt[idx].to(device))
            loss.backward(); opt.step()
        rep = aami_report(yv, predict(model, wv.to(device), rv.to(device)))
        score = float(np.nanmean(rep.loc[list(select), "F1 %"]))
        log(f"epoch {ep:2d} | loss {loss.item():.3f} | val F1({','.join(select)}) {score:5.1f} | {time.time()-t0:.0f} s")
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.cpu().eval(), std, best


def save_checkpoint(path, members, std, **meta):
    """One file for a single model or an ensemble (list of members sharing one standardizer)."""
    torch.save({"members": [m.state_dict() for m in members],
                "n_rr": int(members[0].head[0].in_features - 64),
                "std": {k: np.asarray(v).tolist() for k, v in std.__dict__.items()}, **meta}, path)


def load_checkpoint(path):
    """Load a Step 1.7 single-model file or a v2 file. Returns (model, standardizer)."""
    ck = torch.load(path, map_location="cpu")
    states = ck["members"] if "members" in ck else [ck["state_dict"]]
    nets = []
    for sd in states:
        net = BeatNet(n_rr=ck["n_rr"])
        net.load_state_dict(sd)
        nets.append(net.eval())
    std = Standardizer()
    std.__dict__.update({k: np.asarray(v, dtype=np.float32) for k, v in ck["std"].items()})
    return (nets[0] if len(nets) == 1 else Ensemble(nets).eval()), std


def per_record_s(y_true, y_pred, records, min_beats=20):
    """S-class sensitivity and PPV for every test record with at least min_beats S beats."""
    rows = []
    for r in np.unique(records):
        m = records == r
        n_s = int(np.sum(y_true[m] == 1))
        if n_s < min_beats:
            continue
        tp = np.sum((y_true[m] == 1) & (y_pred[m] == 1))
        fp = np.sum((y_true[m] != 1) & (y_pred[m] == 1))
        rows.append({"record": int(r), "S beats": n_s, "Se %": round(100 * tp / n_s, 1),
                     "PPV %": round(100 * tp / (tp + fp), 1) if tp + fp else float("nan")})
    return rows
