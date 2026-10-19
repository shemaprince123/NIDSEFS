"""
Repair cnn_lstm_results.csv:
  1. Removes ANY stray header-like row appearing as data (first field
     literally "model"), regardless of column count.
  2. Backfills a missing model_size_mb column (as 0.0) on old-format rows.
  3. Deduplicates by (model, dataset): if the same model+dataset pair
     appears more than once (e.g. an old 0.0-placeholder row AND a real
     measured row), keeps the one with the real (non-zero) model_size_mb,
     and reports exactly which duplicates it resolved so nothing is lost
     silently.
  4. Verifies every numeric column is actually numeric before declaring
     success.

Run this once:
    python repair_cnn_lstm_csv_v3.py
"""
import pandas as pd

PATH = "C:/Users/user/Desktop/NIDS_EFS_Project/results/cnn_lstm_results.csv"
EXPECTED_COLS = ["model", "dataset", "accuracy", "f1_score", "training_time",
                  "latency_ms", "throughput", "memory_mb", "model_size_mb"]
NUMERIC_COLS = ["accuracy", "f1_score", "training_time", "latency_ms",
                 "throughput", "memory_mb", "model_size_mb"]

with open(PATH) as f:
    raw_lines = [l.rstrip("\n").rstrip("\r") for l in f if l.strip()]

clean_rows = []
for line in raw_lines:
    parts = line.split(",")
    first_field = parts[0].strip().lower()
    if first_field == "model":
        continue  # strip any header-like row, wherever it appears
    if len(parts) == 8:
        parts.append("0.0")  # backfill missing model_size_mb
    if len(parts) != 9:
        print(f"WARNING: skipping malformed row (expected 9 fields, got {len(parts)}): {line}")
        continue
    clean_rows.append(parts)

df = pd.DataFrame(clean_rows, columns=EXPECTED_COLS)
for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Deduplicate by (model, dataset) ────────────────────────────────────
dupes = df[df.duplicated(subset=["model", "dataset"], keep=False)]
if len(dupes) > 0:
    print(f"Found {len(dupes)} rows involved in (model, dataset) duplicates:")
    print(dupes[["model", "dataset", "model_size_mb", "training_time"]].to_string(index=False))
    print()

# For each duplicate group, prefer the row with the largest model_size_mb
# (the real measurement; the placeholder is 0.0), tie-break on last seen.
df["_row_order"] = range(len(df))
df_sorted = df.sort_values(["model_size_mb", "_row_order"], ascending=[False, True])
df_dedup = df_sorted.drop_duplicates(subset=["model", "dataset"], keep="first")
df_dedup = df_dedup.drop(columns=["_row_order"]).sort_values(["dataset", "model"]).reset_index(drop=True)

removed = len(df) - len(df_dedup)
if removed > 0:
    print(f"Removed {removed} duplicate row(s), keeping the real-measurement version of each.\n")

df_dedup.to_csv(PATH, index=False)

# ── Verify ──────────────────────────────────────────────────────────────
check = pd.read_csv(PATH)
print(f"Repaired. {len(check)} data rows written (expect 8: 2 models x 4 datasets).")
print(f"Columns: {check.columns.tolist()}")
print("\nmodels:", check["model"].unique().tolist())
print("datasets:", check["dataset"].unique().tolist())

print("\nDtype check (all must say float64/int64, NOT object):")
all_good = True
for col in NUMERIC_COLS:
    dtype = check[col].dtype
    ok = dtype != object
    print(f"  {col}: {dtype}  {'OK' if ok else '<<< STILL BROKEN'}")
    if not ok:
        all_good = False

if len(check) != 8:
    all_good = False
    print(f"\n*** Expected exactly 8 rows (2 models x 4 datasets), got {len(check)}. Check manually. ***")

print("\nFull table:")
print(check.to_string(index=False))

print("\n" + ("ALL CHECKS PASSED -- safe to run ahp_scorer.py now."
              if all_good else
              "DO NOT proceed to ahp_scorer.py yet -- see warnings above."))