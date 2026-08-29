"""
fl_dp_experiment.py  -  Step 3, Stage 3: Federated Learning + Differential Privacy.

Runs your full privacy sweep and produces the headline results:
  * for each privacy budget eps in {0.1, 0.5, 1.0, 5.0}, plus a no-DP anchor,
  * across several random seeds,
  * measuring macro-F1, weighted-F1, the ACHIEVED (verified) epsilon, and time.

Each client trains locally with DP-SGD (per-sample gradient clipping + calibrated
Gaussian noise, Opacus); only weights are shared and averaged (FedAvg). Raw data
never leaves the home, and each home's data carries a formal (eps, delta) guarantee.

Results are saved to  results.csv  for analysis and plotting.

Run:  python fl_dp_experiment.py
"""
import warnings; warnings.filterwarnings("ignore")
import copy, time, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import RDPAccountant
import fl_utils as U

# ---- experiment configuration ---------------------------------------------
SEEDS       = [0, 1, 2, 3, 4]          # 5 repeats -> mean and std dev
EPS_LIST    = [0.1, 0.5, 1.0, 5.0]     # your four privacy budgets
DELTA       = 1e-4                     # valid: delta < 1/N for every client
ROUNDS      = 15                       # federated rounds (= local epochs here)
BATCH       = 256
LR          = 1.0                      # high lr suits clipped DP gradients
CLIP        = 1.0                      # max per-sample gradient norm
OUT_CSV     = "results.csv"

clients, (Xtr, ytr), (Xte, yte), _ = U.load_clients()
IN_DIM, K = Xtr.shape[1], len(U.CLASS_LIST)
print(f"Clients {[c['name'] for c in clients]} | test windows {len(Xte)}\n")


def loader(X, y):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                    torch.tensor(y, dtype=torch.long)),
                      batch_size=BATCH, shuffle=True)


def fedavg(states, sizes):
    total = sum(sizes)
    return {k: sum(s[k] * n for s, n in zip(states, sizes)) / total
            for k in states[0]}


def train_client_dp(global_model, c, sigma, accountant):
    """One client, DP-SGD for one round. Returns clean weights."""
    local = copy.deepcopy(global_model)
    opt = torch.optim.SGD(local.parameters(), lr=LR, momentum=0.9)
    dl = loader(c["Xtr"], c["ytr"])
    pe = PrivacyEngine(accountant="rdp")
    dpm, dpo, dpl = pe.make_private(module=local, optimizer=opt, data_loader=dl,
                                    noise_multiplier=sigma, max_grad_norm=CLIP)
    lf = nn.CrossEntropyLoss(weight=U.class_weights(c["ytr"]))
    dpm.train()
    q = BATCH / len(c["Xtr"])
    for xb, yb in dpl:
        dpo.zero_grad(); lf(dpm(xb), yb).backward(); dpo.step()
        if accountant is not None:
            accountant.step(noise_multiplier=sigma, sample_rate=q)
    return {k.replace("_module.", ""): v
            for k, v in dpm.state_dict().items() if "_module." in k}


def run(eps, seed):
    """Run one full federated training with (or without) DP.
    eps=None => no-DP anchor: the SAME DP-SGD pipeline (gradient clipping) with
    the noise switched off (sigma=0), so it isolates the cost of the noise."""
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    gm = U.MLP(IN_DIM, K)

    if eps is None:
        sig = [0.0] * len(clients)              # noise off, clipping still on
        accts = None
    else:
        sig = [get_noise_multiplier(target_epsilon=eps, target_delta=DELTA,
                                    sample_rate=BATCH / len(c["Xtr"]),
                                    epochs=ROUNDS, accountant="rdp") for c in clients]
        accts = [RDPAccountant() for _ in clients]

    for rnd in range(ROUNDS):
        states, sizes = [], []
        for i, c in enumerate(clients):
            acc = None if accts is None else accts[i]
            states.append(train_client_dp(gm, c, sig[i], acc))
            sizes.append(len(c["Xtr"]))
        gm.load_state_dict(fedavg(states, sizes))

    macro, weighted = U.evaluate(gm, Xte, yte)
    emp = float("inf") if accts is None else max(a.get_epsilon(delta=DELTA) for a in accts)
    return macro, weighted, emp, time.time() - t0


# ---- run the whole grid ---------------------------------------------------
rows = []
conditions = [("no-DP", None)] + [(f"eps={e}", e) for e in EPS_LIST]
for name, eps in conditions:
    for seed in SEEDS:
        macro, weighted, emp, secs = run(eps, seed)
        rows.append({"condition": name, "target_eps": eps if eps else "inf",
                     "seed": seed, "macro_f1": round(macro, 4),
                     "weighted_f1": round(weighted, 4),
                     "achieved_eps": round(emp, 3) if eps else "inf",
                     "time_s": round(secs, 1)})
        print(f"  {name:8s} seed {seed}: macroF1 {macro:.3f} | "
              f"weightedF1 {weighted:.3f} | eps~{emp if eps else float('inf'):.2f} | {secs:.1f}s")

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- summary table (mean +/- sd) ------------------------------------------
print("\n================ SUMMARY (mean +/- sd over seeds) ================")
print(f"{'condition':10s} {'macroF1':>16s} {'weightedF1':>16s} {'achieved eps':>14s}")
for name, eps in conditions:
    sub = [r for r in rows if r["condition"] == name]
    mac = np.array([r["macro_f1"] for r in sub]); wt = np.array([r["weighted_f1"] for r in sub])
    e = "inf" if eps is None else f"{np.mean([r['achieved_eps'] for r in sub]):.2f}"
    print(f"{name:10s} {mac.mean():.3f} +/- {mac.std():.3f}   "
          f"{wt.mean():.3f} +/- {wt.std():.3f}   {e:>14s}")
print("==================================================================")
print(f"\nSaved {len(rows)} rows to {OUT_CSV}")
print("Reference (from fl_train.py): centralised macroF1 0.674, federated no-DP 0.639")
