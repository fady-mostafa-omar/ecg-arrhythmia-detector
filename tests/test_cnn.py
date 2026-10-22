import numpy as np
import torch
from ecgsuite.cnn import BeatNet, Ensemble, Standardizer, load_checkpoint, predict, save_checkpoint


def batch(n=6, seed=0):
    g = np.random.default_rng(seed)
    return g.normal(size=(n, 252)).astype(np.float32), g.normal(size=(n, 9)).astype(np.float32)


def test_beatnet_shape_and_size():
    net = BeatNet(n_rr=9).eval()
    w, r = Standardizer().fit(*batch(50))(*batch())
    assert net(w, r).shape == (6, 5)
    assert sum(p.numel() for p in net.parameters()) == 18309


def test_ensemble_averages_probabilities():
    torch.manual_seed(0)
    nets = [BeatNet().eval() for _ in range(3)]
    w, r = Standardizer().fit(*batch(50))(*batch())
    with torch.no_grad():
        p = Ensemble(nets).eval()(w, r).softmax(1)
        ref = torch.stack([n(w, r).softmax(1) for n in nets]).mean(0)
    assert torch.allclose(p, ref, atol=1e-5)
    assert torch.allclose(p.sum(1), torch.ones(6), atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(1)
    nets = [BeatNet().eval() for _ in range(2)]
    std = Standardizer().fit(*batch(50))
    save_checkpoint(str(tmp_path / "ens.pt"), nets, std, seeds=[0, 1])
    ens, std2 = load_checkpoint(str(tmp_path / "ens.pt"))
    (w, r), (w2, r2) = std(*batch()), std2(*batch())
    assert torch.allclose(w, w2) and torch.allclose(r, r2)
    assert (predict(Ensemble(nets), w, r) == predict(ens, w2, r2)).all()
    save_checkpoint(str(tmp_path / "one.pt"), nets[:1], std)
    assert isinstance(load_checkpoint(str(tmp_path / "one.pt"))[0], BeatNet)
