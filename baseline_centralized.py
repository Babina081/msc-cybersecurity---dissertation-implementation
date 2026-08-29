"""
baseline_centralized.py  -  Step 3, Stage 1: the centralised (no-privacy) baseline.

Pools every home's windows and trains ONE neural network to recognise the
activity. This is the 'best case' reference all later stages compare against.

Run:  python baseline_centralized.py
"""
import json, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

SEED, EPOCHS, BATCH, LR = 42, 60, 128, 1e-3
torch.manual_seed(SEED); np.random.seed(SEED)

# --- 1. Load and pool every processed home ---------------------------------
meta = json.loads(open("data/processed/meta.json").read())
CLASS_LIST = meta["class_list"]
X_parts, y_parts = [], []
for f in sorted(glob.glob("data/processed/hh*.npz")):
    d = np.load(f); X_parts.append(d["X"]); y_parts.append(d["y"])
X = np.concatenate(X_parts); y = np.concatenate(y_parts)
print(f"Pooled dataset: {X.shape[0]} windows, {X.shape[1]} features, {len(np.unique(y))} classes")

# --- 2. Split, scale (fit on train only), clip outliers --------------------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
scaler = StandardScaler().fit(X_tr)
X_tr = np.clip(scaler.transform(X_tr), -5, 5)     # clip tames extreme outliers
X_te = np.clip(scaler.transform(X_te), -5, 5)

# --- 3. Model:  features -> 64 -> 32 -> classes ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes))
    def forward(self, x): return self.net(x)

model = MLP(X.shape[1], len(CLASS_LIST))

# softened class weights (square-root of inverse frequency, averaged to ~1)
counts = np.bincount(y_tr, minlength=len(CLASS_LIST))
w = np.sqrt(counts.sum() / (len(CLASS_LIST) * np.maximum(counts, 1)))
w = w / w.mean()
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # Adam = stable

train_dl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                    torch.tensor(y_tr, dtype=torch.long)),
                      batch_size=BATCH, shuffle=True)

def macro_f1_on(Xs, ys):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(Xs, dtype=torch.float32)).argmax(1).numpy()
    return f1_score(ys, p, average="macro", zero_division=0)

# --- 4. Train (watch the loss go DOWN this time) ---------------------------
for epoch in range(EPOCHS):
    model.train(); total = 0.0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb); loss.backward(); optimizer.step()
        total += loss.item() * len(xb)
    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:2d}/{EPOCHS}  avg loss {total/len(X_tr):.3f}"
              f"  train macroF1 {macro_f1_on(X_tr, y_tr):.3f}")

# --- 5. Evaluate -----------------------------------------------------------
model.eval()
with torch.no_grad():
    preds = model(torch.tensor(X_te, dtype=torch.float32)).argmax(1).numpy()
macro = f1_score(y_te, preds, average="macro", zero_division=0)
weighted = f1_score(y_te, preds, average="weighted", zero_division=0)
print("\n================ CENTRALISED BASELINE ================")
print(f"Macro F1    : {macro:.3f}   (fair metric - averages classes equally)")
print(f"Weighted F1 : {weighted:.3f}   (rewards common classes)")
print("======================================================\n")
print(classification_report(y_te, preds, labels=range(len(CLASS_LIST)),
                            target_names=CLASS_LIST, zero_division=0))
