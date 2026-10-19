import argparse
import pandas as pd
import numpy as np
import os
from tco_calculator import compute_annual_tco  # ADDED: wires in the real TCO formula

RESULTS_PATH = "C:/Users/user/Desktop/NIDS_EFS_Project/results/"

# Model sizes and the old provisional weights array were removed here in an
# earlier fix (model_size_mb is now read from the real benchmark data;
# weights live in ahp_scorer.py).

# MODEL_TCO dict REMOVED (was: the hardcoded guess values $800/$1,500/$400/etc
# that never matched the real formula). Annual TCO is now computed per-model
# via compute_annual_tco() from tco_calculator.py, using each model's real
# measured training_time and throughput. See tco_calculator.py for the full
# documented cost assumptions (hardware cost, electricity rate, staffing).

def load_scores():
    path = RESULTS_PATH + "ahp_scores.csv"
    if not os.path.exists(path):
        print("ERROR: ahp_scores.csv not found. Run ahp_scorer.py first.")
        exit(1)
    return pd.read_csv(path)

def load_benchmarks():
    path = RESULTS_PATH + "all_benchmark_results.csv"
    return pd.read_csv(path)

def filter_to_dataset(df, dataset):
    """dataset == "ALL" means mean across all four datasets (explicit,
    honest default). Any other value filters down to that one dataset
    before anything downstream aggregates.
    """
    if dataset == "ALL":
        return df
    return df[df["dataset"] == dataset].copy()

def apply_hard_constraints(benchmarks, budget, hardware, min_accuracy, max_latency):
    """Eliminate models that fail non-negotiable constraints.

    Expects `benchmarks` to already be filtered to the requested dataset
    (or left as all datasets if dataset == "ALL") by the caller.
    """
    eliminated = []
    surviving = benchmarks.copy()

    # Constraint 1: Minimum accuracy
    if min_accuracy > 0:
        mask = surviving.groupby("model")["accuracy"].mean() >= min_accuracy
        failed = mask[~mask].index.tolist()
        for m in failed:
            eliminated.append((m, f"Accuracy below {min_accuracy} threshold"))
        surviving = surviving[surviving["model"].isin(mask[mask].index)]

    # Constraint 2: Maximum latency (ms). Both max_latency and latency_ms
    # are in milliseconds; compared directly (no unit conversion bug).
    if max_latency > 0:
        mask = surviving.groupby("model")["latency_ms"].mean() <= max_latency
        failed = mask[~mask].index.tolist()
        for m in failed:
            eliminated.append((m, f"Latency exceeds {max_latency}ms threshold"))
        surviving = surviving[surviving["model"].isin(mask[mask].index)]

    # Constraint 3: Budget — eliminate models whose real computed TCO
    # (from tco_calculator.py) exceeds 30% of budget, when budget is tight.
    # CHANGED: this now uses the real per-model TCO formula instead of the
    # old hardcoded MODEL_TCO dict.
    if budget < 10000:
        avg_by_model = surviving.groupby("model").agg(
            training_time=("training_time", "mean"),
            throughput=("throughput", "mean")
        ).reset_index()
        high_tco = []
        for _, row in avg_by_model.iterrows():
            tco = compute_annual_tco(row["model"], row["training_time"], row["throughput"])
            if tco is not None and tco > budget * 0.3:
                high_tco.append(row["model"])
        for m in high_tco:
            eliminated.append((m, f"TCO exceeds 30% of budget (${budget})"))
        surviving = surviving[~surviving["model"].isin(high_tco)]

    # Constraint 4: Hardware — eliminate GPU-dependent models on CPU-only.
    # KNOWN LIMITATION: all 8 models in this study were benchmarked on
    # CPU-only hardware (Section 3.5.1), so there is no measured GPU-vs-CPU
    # comparison to filter on. The heuristic below (excluding LSTM under a
    # tight latency ceiling) is a judgment call, not a measured constraint.
    if hardware == "cpu-only":
        if max_latency > 0 and max_latency < 500:
            if "LSTM" in surviving["model"].values:
                eliminated.append(("LSTM", "Impractical latency on CPU-only hardware (heuristic, not measured)"))
                surviving = surviving[surviving["model"] != "LSTM"]

    return surviving, eliminated

def get_top_recommendations(scores_df, benchmarks, surviving_models, dataset, top_n=3):
    """Return top N models by AHP score among survivors, with real
    measured metrics and a real computed TCO per model.
    """
    scores_df = filter_to_dataset(scores_df, dataset)

    if dataset == "ALL":
        overall = scores_df.groupby("model")["AHP_Score"].mean().reset_index()
    else:
        overall = scores_df[["model", "AHP_Score"]].copy()

    overall.columns = ["model", "AHP_Score"]
    overall = overall[overall["model"].isin(surviving_models)]
    overall = overall.sort_values("AHP_Score", ascending=False).head(top_n)

    if dataset == "ALL":
        avg_metrics = benchmarks.groupby("model").agg({
            "accuracy": "mean", "f1_score": "mean",
            "latency_ms": "mean", "training_time": "mean",
            "memory_mb": "mean", "throughput": "mean",
            "model_size_mb": "mean",
        }).reset_index()
    else:
        avg_metrics = benchmarks[["model", "accuracy", "f1_score", "latency_ms",
                                   "training_time", "memory_mb", "throughput",
                                   "model_size_mb"]].copy()

    result = overall.merge(avg_metrics, on="model", how="left")

    # CHANGED: annual_tco_usd is now computed per-model from real measured
    # training_time and throughput, via the documented formula in
    # tco_calculator.py, instead of a hardcoded lookup dict.
    result["annual_tco_usd"] = result.apply(
        lambda row: compute_annual_tco(row["model"], row["training_time"], row["throughput"]),
        axis=1
    )
    return result

def print_report(args, recommendations, eliminated):
    """Print formatted CLI report."""
    print("\n" + "="*60)
    print("  NIDS-EFS — DEPLOYMENT RECOMMENDATION REPORT")
    print("="*60)
    print(f"  Scenario:        {args.scenario.upper()}")
    print(f"  Budget:          ${args.budget:,}")
    print(f"  Hardware:        {args.hardware}")
    print(f"  Min Accuracy:    {args.min_accuracy}")
    print(f"  Max Latency:     {args.max_latency}ms")
    print(f"  Dataset Profile: {args.dataset}")
    print("="*60)

    if eliminated:
        print("\n  ELIMINATED BY HARD CONSTRAINTS:")
        for model, reason in eliminated:
            print(f"    - {model}: {reason}")

    if recommendations.empty:
        print("\n  No models survived the hard constraints.")
        print("  Recommendation: Relax constraints or upgrade hardware.")
        return

    print(f"\n  TOP {len(recommendations)} RECOMMENDED MODELS:\n")
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        print(f"  {'='*50}")
        print(f"  RANK {i}: {row['model']}")
        print(f"  {'='*50}")
        print(f"    AHP Score:       {row['AHP_Score']:.4f}")
        print(f"    Accuracy:        {row['accuracy']:.4f}")
        print(f"    F1-Score:        {row['f1_score']:.4f}")
        print(f"    Avg Latency:     {row['latency_ms']:.0f}ms")
        print(f"    Training Time:   {row['training_time']:.1f}s")
        print(f"    Memory:          {row['memory_mb']:.0f}MB")
        print(f"    Model Size:      {row['model_size_mb']:.3f}MB")
        # FIX: pandas silently turns None into NaN inside a float column,
        # so `is not None` never caught the Ensemble's missing TCO -- it
        # printed "$nan" instead of the intended message. pd.isna() catches
        # both None and NaN.
        tco_display = (f"${row['annual_tco_usd']:,.2f}  (estimate, see tco_calculator.py)"
                        if not pd.isna(row['annual_tco_usd'])
                        else "N/A — pending re-benchmark (Section 5.6.1 data gap)")
        print(f"    Annual TCO:      {tco_display}")
        print(f"    Throughput:      {row['throughput']:.0f} samples/sec")

    print("\n  DEPLOYMENT SPECIFICATIONS:")
    top = recommendations.iloc[0]
    print(f"    Primary:  {top['model']} — deploy on {args.hardware} server")
    print(f"    RAM:      Minimum {top['memory_mb']:.0f}MB dedicated")
    print(f"    Storage:  {top['model_size_mb']:.3f}MB model file")
    top_tco_display = (f"~${top['annual_tco_usd']:,.2f}/year (estimate)"
                        if not pd.isna(top['annual_tco_usd']) else "N/A")
    print(f"    TCO:      {top_tco_display}")

    print("\n  TRADE-OFF ANALYSIS:")
    if len(recommendations) >= 2:
        r1 = recommendations.iloc[0]
        r2 = recommendations.iloc[1]
        print(f"    {r1['model']} vs {r2['model']}:")
        acc_diff = (r1['accuracy'] - r2['accuracy']) * 100
        lat_diff = r2['latency_ms'] - r1['latency_ms']
        print(f"    Accuracy difference: {abs(acc_diff):.2f}%")
        print(f"    Latency difference:  {abs(lat_diff):.0f}ms")
        if not pd.isna(r1['annual_tco_usd']) and not pd.isna(r2['annual_tco_usd']):
            print(f"    TCO difference:      ${abs(r1['annual_tco_usd'] - r2['annual_tco_usd']):,.2f}/year")
        else:
            print("    TCO difference:      N/A — one model's TCO pending re-benchmark")

    print("\n  GAP REMEDIATION:")
    if args.budget < 5000:
        print("    Budget < $5,000: Consider Decision Tree as lowest-TCO option")
    if args.hardware == "cpu-only":
        print("    CPU-only: Avoid LSTM for real-time detection under tight latency limits")
        print("    Recommendation: XGBoost or Decision Tree for best CPU performance")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="NIDS-EFS: Network Intrusion Detection System Selection Tool")
    parser.add_argument("--budget",       type=float, default=10000,
                        help="Annual security budget in USD")
    parser.add_argument("--hardware",     type=str,   default="cpu-only",
                        choices=["cpu-only", "gpu"],
                        help="Hardware environment")
    parser.add_argument("--min-accuracy", type=float, default=0.85,
                        help="Minimum acceptable F1-score (0-1)")
    parser.add_argument("--max-latency",  type=float, default=0,
                        help="Maximum tolerable latency in ms (0=no limit)")
    parser.add_argument("--dataset",      type=str,   default="ALL",
                        choices=["ALL", "NSL-KDD", "UNSW-NB15", "CICIDS2017", "CICIoT2023"],
                        help="Target dataset profile, or ALL for mean across all four")
    parser.add_argument("--scenario",     type=str,   default="B",
                        choices=["A", "B", "C"],
                        help="A=Small fintech, B=Mid-size bank, C=Enterprise")
    args = parser.parse_args()

    scores    = load_scores()
    benchmarks = load_benchmarks()

    benchmarks_scoped = filter_to_dataset(benchmarks, args.dataset)

    surviving, eliminated = apply_hard_constraints(
        benchmarks_scoped, args.budget, args.hardware,
        args.min_accuracy, args.max_latency)

    surviving_models = surviving["model"].unique().tolist()
    recommendations  = get_top_recommendations(
        scores, benchmarks_scoped, surviving_models, args.dataset, top_n=3)

    print_report(args, recommendations, eliminated)

if __name__ == "__main__":
    main()