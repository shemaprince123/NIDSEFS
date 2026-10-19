import numpy as np
import pandas as pd

# ── AHP Criteria Weights ───────────────────────────────────────────────────
# C1: Detection Performance, C2: Computational Efficiency,
# C3: Deployment Feasibility, C4: Resource Consumption, C5: TCO
CRITERIA = ["Detection_Performance", "Computational_Efficiency",
            "Deployment_Feasibility", "Resource_Consumption", "TCO"]

# FIX (was: provisional placeholder never updated after the expert survey).
# These are the empirically derived weights from Section 4.4, full-panel AIJ
# aggregation of 16 expert pairwise-comparison matrices (lambda_max=5.057,
# CR=0.013). Source of truth: Table 4.3 / Chapter 4.
#   C1=0.374  C2=0.154  C3=0.173  C4=0.104  C5=0.195
WEIGHTS = np.array([0.374, 0.154, 0.173, 0.104, 0.195])

# Keep the old provisional weights available for comparison/reproducibility
# (e.g. to regenerate Table 4.6's "Provisional Score" column if needed).
PROVISIONAL_WEIGHTS = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

# Saaty RI for n=5
RI = 1.12

def compute_ahp_weights(pairwise_matrix):
    """Compute AHP weights and consistency ratio from a 5x5 pairwise matrix."""
    n = pairwise_matrix.shape[0]
    col_sums = pairwise_matrix.sum(axis=0)
    norm = pairwise_matrix / col_sums
    weights = norm.mean(axis=1)
    weighted_sum = pairwise_matrix @ weights
    lambda_max = (weighted_sum / weights).mean()
    CI = (lambda_max - n) / (n - 1)
    CR = CI / RI
    return weights, CR

def normalize_metrics(df):
    """Normalize all metrics to 0-1 scale with direction of benefit applied."""
    norm = df.copy()

    # Higher is better
    for col in ["accuracy", "f1_score", "throughput"]:
        min_v, max_v = df[col].min(), df[col].max()
        norm[col] = (df[col] - min_v) / (max_v - min_v + 1e-9)

    # Lower is better
    for col in ["latency_ms", "training_time", "memory_mb"]:
        min_v, max_v = df[col].min(), df[col].max()
        norm[col] = (max_v - df[col]) / (max_v - min_v + 1e-9)

    return norm

def compute_criterion_scores(norm_df):
    """Map normalized metrics to 5 AHP criteria scores.

    NOTE / KNOWN LIMITATION (flag this in thesis Section 6.6 if not already):
    C3 and C5 are proxy measures, not direct measurements of what they claim
    to represent.
      - C3 "Deployment Feasibility" is defined in the survey instrument
        (Appendix D) as hardware compatibility, configuration complexity, and
        staff expertise requirements. This code instead reuses normalized
        training_time as a stand-in. It is a measurable proxy, not a direct
        measurement of feasibility.
      - C5 "TCO" is defined as full lifecycle cost (hardware, energy,
        training, maintenance). This code computes a 0-1 composite of
        training_time and memory_mb, not a dollar-denominated cost model.
    Both are left as-is here since changing the formula would change every
    downstream AHP score and require re-deriving Table 4.6/4.7. If you
    revise these, regenerate ahp_scores.csv and recompute the composite
    rankings before updating the thesis tables.
    """
    scores = pd.DataFrame()
    scores["model"]   = norm_df["model"]
    scores["dataset"] = norm_df["dataset"]

    # C1: Detection Performance = accuracy + f1 average
    scores["C1"] = (norm_df["accuracy"] + norm_df["f1_score"]) / 2

    # C2: Computational Efficiency = latency + throughput average
    scores["C2"] = (norm_df["latency_ms"] + norm_df["throughput"]) / 2

    # C3: Deployment Feasibility = proxy (see docstring above)
    scores["C3"] = norm_df["training_time"]

    # C4: Resource Consumption = inverse of memory_mb
    scores["C4"] = norm_df["memory_mb"]

    # C5: TCO = proxy composite (see docstring above)
    scores["C5"] = (norm_df["training_time"] + norm_df["memory_mb"]) / 2

    return scores

def compute_ahp_score(scores_df, weights=WEIGHTS):
    """Compute final AHP composite score for each model-dataset combination."""
    criterion_cols = ["C1", "C2", "C3", "C4", "C5"]
    scores_df["AHP_Score"] = scores_df[criterion_cols].values @ weights
    return scores_df

def rank_models(results_csv, output_csv=None, weights=WEIGHTS, weight_label="Empirical (Section 4.4)"):
    """Load benchmark results, compute AHP scores, and rank models."""
    df = pd.read_csv(results_csv)

    print(f"\nLoaded {len(df)} benchmark records from {results_csv}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")

    # Normalize per dataset
    all_scores = []
    for ds in df["dataset"].unique():
        subset = df[df["dataset"] == ds].copy().reset_index(drop=True)
        norm = normalize_metrics(subset)
        scores = compute_criterion_scores(norm)
        scores = compute_ahp_score(scores, weights=weights)
        all_scores.append(scores)

    final = pd.concat(all_scores, ignore_index=True)

    # Overall ranking — average AHP score across all datasets
    overall = final.groupby("model")["AHP_Score"].mean().reset_index()
    overall.columns = ["model", "Mean_AHP_Score"]
    overall = overall.sort_values("Mean_AHP_Score", ascending=False).reset_index(drop=True)
    overall["Rank"] = overall.index + 1

    print("\n" + "="*55)
    print(f"NIDS-EFS — AHP RANKINGS ({weight_label} Weights)")
    print("="*55)
    print(f"Weights: Detection={weights[0]} | Efficiency={weights[1]} | "
          f"Feasibility={weights[2]} | Resources={weights[3]} | TCO={weights[4]}")
    print("-"*55)
    for _, row in overall.iterrows():
        print(f"  Rank {int(row['Rank'])}: {row['model']:<20} Score: {row['Mean_AHP_Score']:.4f}")
    print("="*55)

    # Per-dataset rankings
    print("\nPer-Dataset Rankings:")
    for ds in final["dataset"].unique():
        subset = final[final["dataset"] == ds].sort_values(
            "AHP_Score", ascending=False).reset_index(drop=True)
        print(f"\n  {ds}:")
        for _, row in subset.iterrows():
            print(f"    {row['model']:<20} AHP: {row['AHP_Score']:.4f}")

    if output_csv:
        final.to_csv(output_csv, index=False)
        overall.to_csv(output_csv.replace(".csv", "_overall.csv"), index=False)
        print(f"\nDetailed scores saved to {output_csv}")

    return final, overall

if __name__ == "__main__":
    import os
    RESULTS_PATH = "C:/Users/user/Desktop/NIDS_EFS_Project/results/"

    # Merge all benchmark results into one file
    dfs = []
    for f in ["benchmark_results_v2.csv", "cnn_lstm_results.csv"]:
        path = RESULTS_PATH + f
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
            print(f"Loaded: {f}")

    if not dfs:
        print("No result files found.")
    else:
        combined = pd.concat(dfs, ignore_index=True)
        combined_path = RESULTS_PATH + "all_benchmark_results.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Combined: {len(combined)} records saved to all_benchmark_results.csv")

        # Empirical weights (default, matches current thesis findings)
        final, overall = rank_models(
            combined_path,
            output_csv=RESULTS_PATH + "ahp_scores.csv",
            weights=WEIGHTS,
            weight_label="Empirical (Section 4.4)"
        )

        # Also regenerate the provisional-weight version, for Table 4.6's
        # side-by-side comparison column, so both stay reproducible from
        # one script run.
        final_prov, overall_prov = rank_models(
            combined_path,
            output_csv=RESULTS_PATH + "ahp_scores_provisional.csv",
            weights=PROVISIONAL_WEIGHTS,
            weight_label="Provisional (Chapter 3)"
        )