import time
import tracemalloc
import numpy as np
import csv
import os
import gc  # ADDED: for memory cleanup between models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, accuracy_score

RESULTS_PATH = "C:/Users/user/Desktop/NIDS_EFS_Project/results/"
MODELS_TEMP_PATH = RESULTS_PATH + "temp_models/"
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(MODELS_TEMP_PATH, exist_ok=True)

def load_data(name):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.model_selection import train_test_split

    if name == "NSL-KDD":
        cols = ["duration","protocol_type","service","flag","src_bytes","dst_bytes",
                "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
                "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
                "num_shells","num_access_files","num_outbound_cmds","is_host_login",
                "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
                "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
                "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
                "dst_host_same_srv_rate","dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
                "dst_host_serror_rate","dst_host_srv_serror_rate",
                "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
        BASE = "C:/Users/user/Desktop/NIDS_EFS_Project/Datasets/NSL_KDD-master/NSL_KDD-master/"
        train = pd.read_csv(BASE + "KDDTrain+.txt", names=cols)
        test  = pd.read_csv(BASE + "KDDTest+.txt",  names=cols)
        for df in [train, test]:
            df.drop("difficulty", axis=1, inplace=True)
            df["label"] = df["label"].apply(lambda x: 0 if x=="normal" else 1)
            for col in ["protocol_type","service","flag"]:
                df[col] = LabelEncoder().fit_transform(df[col])
        X_tr = MinMaxScaler().fit_transform(train.drop("label",axis=1))
        X_te = MinMaxScaler().fit_transform(test.drop("label",axis=1))
        return X_tr, X_te, train["label"].values, test["label"].values

    elif name == "UNSW-NB15":
        BASE = "C:/Users/user/Desktop/NIDS_EFS_Project/Datasets/UNSW_NB15/"
        train = pd.read_csv(BASE + "UNSW_NB15_training-set.csv")
        test  = pd.read_csv(BASE + "UNSW_NB15_testing-set.csv")
        for df in [train, test]:
            df.drop(columns=["id","attack_cat"], inplace=True, errors="ignore")
            for col in df.select_dtypes(include="object").columns:
                df[col] = LabelEncoder().fit_transform(df[col])
        X_tr = MinMaxScaler().fit_transform(train.drop("label",axis=1))
        X_te = MinMaxScaler().fit_transform(test.drop("label",axis=1))
        return X_tr, X_te, train["label"].values, test["label"].values

    elif name == "CICIDS2017":
        import glob
        BASE = "C:/Users/user/Desktop/NIDS_EFS_Project/Datasets/CIC-IDS- 2017/"
        files = glob.glob(BASE + "*.csv")
        dfs = [pd.read_csv(f, low_memory=False) for f in files]
        data = pd.concat(dfs, ignore_index=True)
        data.columns = data.columns.str.strip()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        data["label"] = data["Label"].apply(lambda x: 0 if x=="BENIGN" else 1)
        data.drop(columns=["Label"], inplace=True)
        X = MinMaxScaler().fit_transform(data.drop("label",axis=1))
        y = data["label"].values
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    elif name == "CICIoT2023":
        import glob
        BASE = "C:/Users/user/Desktop/NIDS_EFS_Project/Datasets/CIC_IOT_Dataset2023/MERGED_CSV/"
        files = sorted(glob.glob(BASE + "*.csv"))[:10]
        dfs = []
        for f in files:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)
        data.columns = data.columns.str.strip()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        label_col = [c for c in data.columns if "label" in c.lower() or "attack" in c.lower()][-1]
        data["label"] = data[label_col].apply(lambda x: 0 if str(x).lower() in ["benign","normal","0"] else 1)
        data.drop(columns=[label_col], inplace=True)
        for col in data.select_dtypes(include="object").columns:
            data[col] = LabelEncoder().fit_transform(data[col])
        # ADDED: float32 instead of float64 -- halves memory for this large
        # dataset (7.3M rows), fixing the ArrayMemoryError on this dataset.
        data = data.astype('float32')
        X = MinMaxScaler().fit_transform(data.drop("label",axis=1))
        y = data["label"].values
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def build_cnn(input_dim):
    m = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=(input_dim, 1)),
        Conv1D(32, kernel_size=3, activation="relu"),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

def build_lstm(input_dim):
    m = Sequential([
        LSTM(64, input_shape=(input_dim, 1), return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

def measure_keras_model_size_mb(model, model_name, dataset_name):
    temp_path = f"{MODELS_TEMP_PATH}{model_name}_{dataset_name}.keras"
    model.save(temp_path)
    size_mb = round(os.path.getsize(temp_path) / (1024 ** 2), 2)
    os.remove(temp_path)
    return size_mb

def benchmark(model_name, model_fn, ds_name, X_train, X_test, y_train, y_test):
    print(f"\n  Running {model_name} on {ds_name}...")
    X_tr = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_te = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = model_fn(X_train.shape[1])

    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit(X_tr, y_train, epochs=5, batch_size=256, verbose=0)
    training_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = round(peak / (1024**2), 2)

    model_size_mb = measure_keras_model_size_mb(model, model_name, ds_name)

    latencies = []
    for _ in range(50):
        t1 = time.perf_counter()
        model.predict(X_te, verbose=0)
        latencies.append(time.perf_counter() - t1)

    median_latency = np.median(latencies) * 1000
    throughput = len(X_te) / np.median(latencies)
    y_pred = (model.predict(X_te, verbose=0) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"  OK Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Train: {training_time:.2f}s | "
          f"Latency: {median_latency:.2f}ms | Size: {model_size_mb}MB")

    result = {
        "model": model_name, "dataset": ds_name,
        "accuracy": round(accuracy,4), "f1_score": round(f1,4),
        "training_time": round(training_time,4),
        "latency_ms": round(median_latency,4),
        "throughput": round(throughput,2),
        "memory_mb": memory_mb,
        "model_size_mb": model_size_mb,
    }

    # ADDED: release TF/Keras session and force garbage collection so
    # memory from this model doesn't accumulate into the next one. This
    # is what caused the CICIoT2023 crash -- by the 7th of 8 model runs,
    # leftover TF graph memory from earlier runs had not been freed.
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return result

DATASETS = ["CICIoT2023"]  # CHANGED: only re-running the dataset that crashed;
                            # NSL-KDD, UNSW-NB15, CICIDS2017 already completed
                            # and are saved in cnn_lstm_results.csv
MODELS   = {"CNN": build_cnn, "LSTM": build_lstm}

if __name__ == "__main__":
    all_results = []
    csv_file = RESULTS_PATH + "cnn_lstm_results.csv"
    # CHANGED: detect whether the file already has content (from the
    # earlier successful run), so we don't write a duplicate header row
    # that would corrupt the CSV when it's read back later.
    headers_written = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0

    for ds in DATASETS:
        print(f"\n{'='*50}\nDataset: {ds}\n{'='*50}")
        X_train, X_test, y_train, y_test = load_data(ds)
        for mname, mfn in MODELS.items():
            r = benchmark(mname, mfn, ds, X_train, X_test, y_train, y_test)
            all_results.append(r)

            with open(csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=r.keys())
                if not headers_written:
                    writer.writeheader()
                    headers_written = True
                writer.writerow(r)
            print("Saved: " + mname + " on " + ds)

    print("All results saved to " + csv_file)