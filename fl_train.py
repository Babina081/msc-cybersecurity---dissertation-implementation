"""
fl_train.py  -  Step 3, Stage 2: Federated Learning with FedAvg (no privacy yet).

Each home trains the model on its OWN data and shares only weights; a server
averages those weights (FedAvg). Raw data never leaves the home. For a fair
comparison the script also trains a centralised model on the SAME split, so the
only difference is federation itself.

Run:  python fl_train.py
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import fl_utils as U

SEED, ROUNDS, LOCAL_EPOCHS, BATCH, LR = 42, 40, 2, 128, 1e-3
CENTRAL_EPOCHS = 60
torch.manual_seed(SEED); np.random.seed(SEED)

clients, (Xtr, ytr), (Xte, yte), _ = U.load_clients()
in_dim, K = Xtr.shape[1], len(U.CLASS_LIST)
print("Clients:", [c["name"] for c in clients])
print(f"Pooled train {len(Xtr)} windows | shared test {len(Xte)} windows\n")


def train_local(model, X, y, epochs):
    """Train a model in place on one dataset for a number of epochs."""
    model.train()
    loss_fn = nn.CrossEntropyLoss(weight=U.class_weights(y))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                  torch.tensor(y, dtype=torch.long)),
                    batch_size=BATCH, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    return model


def fedavg(states, sizes):
    """Weighted average of client weights (weight = #samples at that client)."""
    total = sum(sizes)
    return {k: sum(s[k] * n for s, n in zip(states, sizes)) / total
            for k in states[0]}


# ---- centralised reference (same data, same test) -------------------------
central = U.MLP(in_dim, K)
train_local(central, Xtr, ytr, CENTRAL_EPOCHS)
c_macro, c_weight = U.evaluate(central, Xte, yte)

# ---- federated training ---------------------------------------------------
global_model = U.MLP(in_dim, K)
for rnd in range(ROUNDS):
    states, sizes = [], []
    for c in clients:
        local = copy.deepcopy(global_model)             # start from global
        train_local(local, c["Xtr"], c["ytr"], LOCAL_EPOCHS)
        states.append(local.state_dict()); sizes.append(len(c["Xtr"]))
    global_model.load_state_dict(fedavg(states, sizes)) # average back together
    if (rnd + 1) % 10 == 0:
        m, _ = U.evaluate(global_model, Xte, yte)
        print(f"  round {rnd+1:2d}/{ROUNDS}  federated macroF1 {m:.3f}")

f_macro, f_weight = U.evaluate(global_model, Xte, yte)

print("\n======= FEDERATED vs CENTRALISED (identical data & test) =======")
print(f"Centralised (no privacy)  macroF1 {c_macro:.3f}   weightedF1 {c_weight:.3f}")
print(f"Federated   (no privacy)  macroF1 {f_macro:.3f}   weightedF1 {f_weight:.3f}")
print("================================================================")
print("\nThe gap between these two is the 'cost of federation'.")
print("Next stage adds Differential Privacy on top of the federated model.")
