"""
TCO calculation for NIDS-EFS.
Add this function to benchmark_runner.py (or a shared utils file), then
call it per model after training_time is known.

Documented assumptions (state these in Chapter 3 / Appendix):
  - Hardware: entry-level CPU-only server comparable to the study's
    benchmarking machine (Intel Core i7-class, 16GB RAM), ~$1,200 USD,
    amortized over a 3-year lifespan. All 8 models share this same
    hardware cost since they ran on identical infrastructure.
  - Power draw: ~65W average for a CPU-only server under typical ML
    training/inference load (standard range for this server class).
  - Electricity rate: KES 22-25/kWh commercial tariff (~$0.17/kWh),
    per EPRA/KPLC 2025-2026 commercial rates (source: globalpetrolprices.com,
    Dec 2025; stimatracker.com) -- representative of the East African
    financial institution context this study targets.
  - Maintenance: assumes quarterly retraining (4x/year), each retrain
    costing one training_time run plus an estimated analyst labor cost.
"""

# ── Documented cost assumptions (cite these in Chapter 3 / Appendix) ──────
HARDWARE_COST_USD = 1200          # entry-level CPU-only server
HARDWARE_LIFESPAN_YEARS = 3
SERVER_POWER_DRAW_WATTS = 65      # average under ML training/inference load
ELECTRICITY_RATE_USD_PER_KWH = 0.17   # Kenya commercial tariff (EPRA/KPLC)
RETRAINS_PER_YEAR = 4             # assumed quarterly retraining
ANALYST_HOURLY_RATE_USD = 15      # assumed local analyst/engineer rate
RETRAIN_LABOR_HOURS = 2           # assumed oversight time per retrain, beyond compute

# Annual traffic volume assumption, tied to this study's own scenario
# definitions (Appendix D): mid-size bank profile, ~50,000 flows/day.
ANNUAL_FLOWS = 50_000 * 365

# Staffing cost tier, tied directly to this study's own C3 (Deployment
# Feasibility) definition in Appendix D: "a deep learning model requiring
# specialised ML engineers to maintain scores poorly if the organisation
# has only general IT staff." Traditional ML models (tree-based, linear,
# probabilistic) can be maintained by a generalist IT/security analyst;
# deep learning models require a specialised ML engineer, a real and
# well-documented salary premium.
GENERALIST_ANNUAL_SALARY_USD = 9_000    # Kenya-market generalist IT/security analyst
SPECIALIST_ANNUAL_SALARY_USD = 22_000   # Kenya-market ML/data engineer premium
MAINTENANCE_TIME_FRACTION = 0.10        # assumed 10% of a staff member's year

DEEP_LEARNING_MODELS = {"CNN", "LSTM"}


def compute_annual_tco(model_name, training_time_seconds, throughput_samples_per_sec):
    """
    Returns estimated annual TCO in USD for one model, or None if the
    inputs are unreliable (see guard below).

    model_name: used only to determine whether the deep-learning staffing
        premium applies (see DEEP_LEARNING_MODELS above)
    training_time_seconds: the model's measured training_time (one run)
    throughput_samples_per_sec: the model's measured throughput
    """
    # GUARD: Ensemble_RF_XGB has a known data gap (Section 4.6.1/5.6.1) --
    # its training_time, latency, throughput, and memory are all recorded
    # as exactly 0.0, not genuine measurements. Dividing ANNUAL_FLOWS by a
    # near-zero throughput would otherwise produce a nonsensical TCO
    # (this surfaced as $56 million in testing). Return None here instead
    # of a fabricated number; the caller should display this as "N/A --
    # pending re-benchmark" rather than a dollar figure.
    if throughput_samples_per_sec is None or throughput_samples_per_sec < 1:
        return None

    # 1. Hardware cost, amortized across all models (same shared server)
    hardware_annual = HARDWARE_COST_USD / HARDWARE_LIFESPAN_YEARS

    # 2. Training energy: one retrain's compute time x retrains/year
    training_hours_per_retrain = training_time_seconds / 3600
    training_energy_kwh_per_year = (
        training_hours_per_retrain * (SERVER_POWER_DRAW_WATTS / 1000) * RETRAINS_PER_YEAR
    )

    # 3. Inference energy: driven by real throughput, not a fixed guess
    inference_seconds_per_year = ANNUAL_FLOWS / max(throughput_samples_per_sec, 1e-6)
    inference_hours_per_year = inference_seconds_per_year / 3600
    inference_energy_kwh_per_year = inference_hours_per_year * (SERVER_POWER_DRAW_WATTS / 1000)

    energy_annual = (
        (training_energy_kwh_per_year + inference_energy_kwh_per_year)
        * ELECTRICITY_RATE_USD_PER_KWH
    )

    # 4. Staffing cost: the real differentiator (see comment above)
    salary = (SPECIALIST_ANNUAL_SALARY_USD if model_name in DEEP_LEARNING_MODELS
              else GENERALIST_ANNUAL_SALARY_USD)
    staffing_annual = salary * MAINTENANCE_TIME_FRACTION

    total = hardware_annual + energy_annual + staffing_annual
    return round(total, 2)


if __name__ == "__main__":
    # Sanity check using real measured training_time + throughput (Table 4.5)
    examples = {
        # model: (training_time_s, throughput_samples_per_sec)
        "XGBoost":      (17.0,   1_429_984),
        "DecisionTree": (49.8,   4_604_266),
        "RandomForest": (140.3,    429_832),
        "CNN":          (940.6,     12_645),
        "LSTM":         (4342.5,     4_872),
    }
    print(f"{'Model':<14}{'Training(s)':>12}{'Throughput':>14}{'Est. Annual TCO':>18}")
    for model, (t, thr) in examples.items():
        tco = compute_annual_tco(model, t, thr)
        print(f"{model:<14}{t:>12.1f}{thr:>14,}{'$'+format(tco, ',.2f'):>18}")