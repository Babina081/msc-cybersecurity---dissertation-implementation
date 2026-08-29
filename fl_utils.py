"""
fl_utils.py  -  shared helpers used by every Step 3 experiment
(centralised, federated, and federated+DP), so all three are compared
on identical data, the same model, and the same metric.

This file is imported by the other scripts - you do not run it directly.
"""
import json, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

SEED = 42
META = json.loads(open("data/processed/meta.json").read())
CLASS_LIST = META["class_list"]


class MLP(nn.Module):
    """features -> 64 -> 32 -> classes (identical everywhere)."""
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes))
    def forward(self, x):
        return self.net(x)


def _split_client(X, y, test_frac=0.2, seed=SEED):
    """Per-client train/test split that survives tiny classes
    (a class with 1 sample goes to train; anything larger puts >=1 in test)."""
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        n_te = max(1, int(round(len(idx) * test_frac))) if len(idx) > 1 else 0
        te += list(idx[:n_te]); tr += list(idx[n_te:])
    return np.array(tr), np.array(te)


def load_clients():
    """Load every home as a client. Returns:
       clients      : list of dicts with scaled Xtr/ytr/Xte/yte per home
       (Xtr, ytr)   : pooled training set
       (Xte, yte)   : pooled test set (same test used by all experiments)
    Scaler is fit on pooled TRAIN only, then applied to everyone (no leakage)."""
    clients, xtr_raw = [], []
    for f in sorted(glob.glob("data/processed/hh*.npz")):
        d = np.load(f); X, y = d["X"], d["y"]
        tr, te = _split_client(X, y)
        clients.append({"name": Path(f).stem,
                        "Xtr": X[tr], "ytr": y[tr], "Xte": X[te], "yte": y[te]})
        xtr_raw.append(X[tr])

    scaler = StandardScaler().fit(np.concatenate(xtr_raw))
    for c in clients:
        c["Xtr"] = np.clip(scaler.transform(c["Xtr"]), -5, 5)
        c["Xte"] = np.clip(scaler.transform(c["Xte"]), -5, 5)

    Xtr = np.concatenate([c["Xtr"] for c in clients])
    ytr = np.concatenate([c["ytr"] for c in clients])
    Xte = np.concatenate([c["Xte"] for c in clients])
    yte = np.concatenate([c["yte"] for c in clients])
    return clients, (Xtr, ytr), (Xte, yte), scaler


def class_weights(y, n=None):
    """Softened inverse-frequency weights (so rare activities aren't ignored)."""
    n = n or len(CLASS_LIST)
    counts = np.bincount(y, minlength=n)
    w = np.sqrt(counts.sum() / (n * np.maximum(counts, 1)))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def evaluate(model, X, y):
    """Return (macro_F1, weighted_F1) on a dataset."""
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
    return (f1_score(y, p, average="macro", zero_division=0),
            f1_score(y, p, average="weighted", zero_division=0))
