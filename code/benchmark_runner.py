import time
import tracemalloc
import numpy as np
import mlflow
import os
import joblib
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from data_loader import load_nslkdd, load_unsw, load_cicids, load_ciciot
import csv

# ── Configuration ─────────────────────────────────────────────────────────────
RESULTS_PATH = "C:/Users/user/Desktop/NIDS_EFS_Project/results/"
MLFLOW_PATH  = "C:/Users/user/Desktop/NIDS_EFS_Project/mlflow_logs"
MODELS_TEMP_PATH = RESULTS_PATH + "temp_models/"
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(MODELS_TEMP_PATH, exist_ok=True)

mlflow.set_tracking_uri(f"file:///{MLFLOW_PATH}")
mlflow.set_experiment("NIDS_EFS_Benchmarking")

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "RandomForest":    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "DecisionTree":    DecisionTreeClassifier(random_state=42),
    "NaiveBayes":      GaussianNB(),
    "XGBoost":         XGBClassifier(random_state=42, n_jobs=-1, verbosity=0),
    "SVM":             make_pipeline(StandardScaler(), LinearSVC(random_state=42, max_iter=1000)),
    "Ensemble_RF_XGB": None,
}

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = {
    "NSL-KDD":    load_nslkdd,
    "UNSW-NB15":  load_unsw,
    "CICIDS2017": load_cicids,
    "CICIoT2023": load_ciciot,
}

def measure_model_size_mb(model, model_name, dataset_name):
    temp_path = f"{MODELS_TEMP_PATH}{model_name}_{dataset_name}.pkl"
    joblib.dump(model, temp_path)
    size_mb = round(os.path.getsize(temp_path) / (1024 ** 2), 2)
    os.remove(temp_path)
    return size_mb

# ── Benchmark function (unchanged, used for the 5 single models) ─────────────
def benchmark(model_name, model, dataset_name, X_train, X_test, y_train, y_test):
    print(f"\n  Running {model_name} on {dataset_name}...")

    tracemalloc.start()
    t_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = round(peak / (1024 ** 2), 2)

    model_size_mb = measure_model_size_mb(model, model_name, dataset_name)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        model.predict(X_test)
        latencies.append(time.perf_counter() - t0)

    median_latency = np.median(latencies) * 1000
    throughput = len(X_test) / np.median(latencies)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")

    results = {
        "model":         model_name,
        "dataset":       dataset_name,
        "accuracy":      round(accuracy, 4),
        "f1_score":      round(f1, 4),
        "training_time": round(training_time, 4),
        "latency_ms":    round(median_latency, 4),
        "throughput":    round(throughput, 2),
        "memory_mb":     round(memory_mb, 2),
        "model_size_mb": model_size_mb,
    }

    with mlflow.start_run(run_name=f"{model_name}_{dataset_name}"):
        mlflow.log_params({"model": model_name, "dataset": dataset_name})
        mlflow.log_metrics({k: v for k, v in results.items()
                            if isinstance(v, (int, float))})

    print(f"  OK Accuracy: {accuracy:.4f} | F1: {f1:.4f} | "
          f"Train: {training_time:.2f}s | Latency: {median_latency:.2f}ms | "
          f"Size: {model_size_mb}MB")
    return results

# ── ADDED: dedicated benchmark function for the ensemble ──────────────────────
# FIX (was: training_time, latency_ms, throughput, and memory_mb were all
# hardcoded to 0 for Ensemble_RF_XGB -- not measured, not estimated, just
# literal zeros, documented as a known limitation in Section 5.6.1/4.6.1.
# This function actually measures the ensemble the same way every other
# model is measured: real training time (fitting both constituent models),
# real memory (tracemalloc across both fits), real inference latency over
# 50 trials of the actual averaging-and-argmax prediction path, and real
# throughput derived from that measured latency.
def benchmark_ensemble(dataset_name, X_train, X_test, y_train, y_test):
    print(f"\n  Running Ensemble_RF_XGB on {dataset_name}...")

    tracemalloc.start()
    t_start = time.perf_counter()
    rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(random_state=42, n_jobs=-1, verbosity=0)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    training_time = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = round(peak / (1024 ** 2), 2)

    rf_size  = measure_model_size_mb(rf, "RF_part", dataset_name)
    xgb_size = measure_model_size_mb(xgb, "XGB_part", dataset_name)
    model_size_mb = round(rf_size + xgb_size, 2)

    # Real inference timing: the actual predict path the ensemble uses --
    # both models' probabilities, averaged, then argmax -- run 50 times,
    # exactly like every other model's benchmark() does.
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        rf_proba  = rf.predict_proba(X_test)
        xgb_proba = xgb.predict_proba(X_test)
        avg_proba = (rf_proba + xgb_proba) / 2
        _ = np.argmax(avg_proba, axis=1)
        latencies.append(time.perf_counter() - t0)

    median_latency = np.median(latencies) * 1000
    throughput = len(X_test) / np.median(latencies)

    # Final prediction for accuracy/F1 (same logic as the timed loop above)
    rf_proba  = rf.predict_proba(X_test)
    xgb_proba = xgb.predict_proba(X_test)
    avg_proba = (rf_proba + xgb_proba) / 2
    y_pred    = np.argmax(avg_proba, axis=1)
    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average="weighted")

    results = {
        "model":         "Ensemble_RF_XGB",
        "dataset":       dataset_name,
        "accuracy":      round(accuracy, 4),
        "f1_score":      round(f1, 4),
        "training_time": round(training_time, 4),
        "latency_ms":    round(median_latency, 4),
        "throughput":    round(throughput, 2),
        "memory_mb":     round(memory_mb, 2),
        "model_size_mb": model_size_mb,
    }

    with mlflow.start_run(run_name=f"Ensemble_RF_XGB_{dataset_name}"):
        mlflow.log_params({"model": "Ensemble_RF_XGB", "dataset": dataset_name})
        mlflow.log_metrics({k: v for k, v in results.items()
                            if isinstance(v, (int, float))})

    print(f"  OK Accuracy: {accuracy:.4f} | F1: {f1:.4f} | "
          f"Train: {training_time:.2f}s | Latency: {median_latency:.2f}ms | "
          f"Size: {model_size_mb}MB (RF {rf_size}MB + XGB {xgb_size}MB)")
    return results

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    all_results = []

    for ds_name, loader in DATASETS.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*50}")
        X_train, X_test, y_train, y_test = loader()

        for m_name, model in MODELS.items():
            if m_name == "Ensemble_RF_XGB":
                # CHANGED: calls the new real-measurement function instead
                # of hardcoding zeros.
                result = benchmark_ensemble(ds_name, X_train, X_test, y_train, y_test)
                all_results.append(result)
            else:
                result = benchmark(m_name, model, ds_name,
                                   X_train, X_test, y_train, y_test)
                all_results.append(result)

    # Save results to CSV
    csv_file = RESULTS_PATH + "benchmark_results_v2.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print("Results saved to " + csv_file)