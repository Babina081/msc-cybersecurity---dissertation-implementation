"""
plot_results.py  -  turn results.csv into the headline privacy-utility figure.

Produces  figures/tradeoff_curve.png : macro- and weighted-F1 against the
privacy budget epsilon, with the no-privacy ceiling and the 0.75 hypothesis
target drawn in for reference.

Run:  python plot_results.py
"""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

Path("figures").mkdir(exist_ok=True)

# --- read results.csv, group scores by condition ---------------------------
macro, weight = defaultdict(list), defaultdict(list)
for r in csv.DictReader(open("results.csv")):
    macro[r["condition"]].append(float(r["macro_f1"]))
    weight[r["condition"]].append(float(r["weighted_f1"]))

def stats(d, cond):
    a = np.array(d[cond]); return a.mean(), a.std()

# DP points sorted by epsilon
eps_conds = sorted([c for c in macro if c.startswith("eps=")],
                   key=lambda c: float(c.split("=")[1]))
eps_vals = [float(c.split("=")[1]) for c in eps_conds]
m_mean = [stats(macro, c)[0] for c in eps_conds]
m_sd   = [stats(macro, c)[1] for c in eps_conds]
w_mean = [stats(weight, c)[0] for c in eps_conds]
w_sd   = [stats(weight, c)[1] for c in eps_conds]

# --- plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(eps_vals, m_mean, yerr=m_sd, marker="o", capsize=4,
            linewidth=2, label="Macro-F1 (FL+DP)")
ax.errorbar(eps_vals, w_mean, yerr=w_sd, marker="s", capsize=4,
            linewidth=2, label="Weighted-F1 (FL+DP)")

# no-privacy ceilings (dashed) if present
if "no-DP" in macro:
    ax.axhline(stats(macro, "no-DP")[0], ls="--", color="tab:blue", alpha=0.6,
               label=f"Macro-F1 no-DP ceiling ({stats(macro,'no-DP')[0]:.2f})")
    ax.axhline(stats(weight, "no-DP")[0], ls="--", color="tab:orange", alpha=0.6,
               label=f"Weighted-F1 no-DP ceiling ({stats(weight,'no-DP')[0]:.2f})")

# hypothesis target
ax.axhline(0.75, ls=":", color="red", alpha=0.8, label="Hypothesis target (0.75)")

ax.set_xscale("log")
ax.set_xticks(eps_vals); ax.set_xticklabels([str(e) for e in eps_vals])
ax.set_xlabel("Privacy budget  \u03b5  (log scale; lower = stronger privacy)")
ax.set_ylabel("F1-score")
ax.set_ylim(0, 1)
ax.set_title("Privacy\u2013utility trade-off: FL + Differential Privacy\n"
             "smart-home activity recognition (CASAS)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/tradeoff_curve.png", dpi=150)
print("Saved figures/tradeoff_curve.png")

# also print a clean table for your results chapter
print("\n eps      macroF1        weightedF1")
for c, e in zip(eps_conds, eps_vals):
    print(f" {e:<5} {stats(macro,c)[0]:.3f}+/-{stats(macro,c)[1]:.3f}   "
          f"{stats(weight,c)[0]:.3f}+/-{stats(weight,c)[1]:.3f}")
if "no-DP" in macro:
    print(f" no-DP {stats(macro,'no-DP')[0]:.3f}+/-{stats(macro,'no-DP')[1]:.3f}   "
          f"{stats(weight,'no-DP')[0]:.3f}+/-{stats(weight,'no-DP')[1]:.3f}")
