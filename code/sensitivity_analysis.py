"""
Sensitivity analysis for the NIDS-EFS composite ranking (Section 5.7.1, Table 5.8).

Usage:  python sensitivity_analysis.py ahp_scores.csv

Stage 1: each empirical criterion weight is raised and lowered by 10% (relative),
         the other weights are rescaled proportionally, and the top-ranked model is recorded.
Stage 2: 10,000 weight vectors are drawn from a Dirichlet distribution centered on the
         empirical weights, for concentrations 20, 50 (primary) and 100, using one
         numpy Generator seeded with 42 and called in the order 20, 50, 100.
A recommendation is robust if the top model holds in at least 90% of samples.
"""
import sys
import numpy as np
import pandas as pd

ORDER = ["XGBoost", "DecisionTree", "RandomForest", "Ensemble_RF_XGB", "SVM", "CNN", "NaiveBayes", "LSTM"]
W = np.array([0.374, 0.154, 0.173, 0.104, 0.195])   # empirical weights, Section 5.4
CRIT = ["C1", "C2", "C3", "C4", "C5"]

df = pd.read_csv(sys.argv[1])
C = df.groupby("model")[CRIT].mean().loc[ORDER].values      # mean criterion scores across datasets

print("Stage 1: deterministic +/-10% perturbations")
wins = 0
for i in range(5):
    for f in (0.9, 1.1):
        w = W.copy(); w[i] *= f
        rest = [j for j in range(5) if j != i]
        w[rest] *= (1 - w[i]) / w[rest].sum()
        top = ORDER[int(np.argmax(C @ w))]
        wins += (top == "XGBoost")
        print(f"  {CRIT[i]} x{f}: top = {top}")
print(f"  XGBoost first in {wins} of 10 perturbations\n")

print("Stage 2: Monte Carlo (Dirichlet, 10,000 draws, seed 42)")
rng = np.random.default_rng(42)
for conc in (20, 50, 100):
    draws = rng.dirichlet(W * conc, size=10000)
    top = np.argmax(draws @ C.T, axis=1)
    share = pd.Series([ORDER[i] for i in top]).value_counts(normalize=True) * 100
    print(f"  concentration {conc}: " + ", ".join(f"{m} {v:.1f}%" for m, v in share.items()))