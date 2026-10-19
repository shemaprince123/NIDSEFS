import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

DATASETS_PATH = "C:/Users/user/Desktop/NIDS_EFS_Project/Datasets/NSL_KDD-master/NSL_KDD-master/"

def load_nslkdd():
    cols = ["duration","protocol_type","service","flag","src_bytes","dst_bytes",
            "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
            "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
            "num_shells","num_access_files","num_outbound_cmds","is_host_login",
            "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
            "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
            "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

    train = pd.read_csv(DATASETS_PATH + "KDDTrain+.txt", names=cols)
    test  = pd.read_csv(DATASETS_PATH + "KDDTest+.txt",  names=cols)

    for df in [train, test]:
        df.drop("difficulty", axis=1, inplace=True)
        df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)
        for col in ["protocol_type","service","flag"]:
            df[col] = LabelEncoder().fit_transform(df[col])

    X_train = MinMaxScaler().fit_transform(train.drop("label", axis=1))
    X_test  = MinMaxScaler().fit_transform(test.drop("label",  axis=1))
    y_train = train["label"].values
    y_test  = test["label"].values

    print(f"NSL-KDD  Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_nslkdd()



def load_unsw():
    UNSW_PATH = "C:/Users/user/Desktop/NIDS_EFS_Project/Datasets/UNSW_NB15/"
    
    train = pd.read_csv(UNSW_PATH + "UNSW_NB15_training-set.csv")
    test  = pd.read_csv(UNSW_PATH + "UNSW_NB15_testing-set.csv")

    for df in [train, test]:
        df.drop(columns=["id","attack_cat"], inplace=True, errors="ignore")
        for col in df.select_dtypes(include="object").columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    X_train = MinMaxScaler().fit_transform(train.drop("label", axis=1))
    X_test  = MinMaxScaler().fit_transform(test.drop("label",  axis=1))
    y_train = train["label"].values
    y_test  = test["label"].values

    print(f"UNSW-NB15  Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_nslkdd()
    load_unsw()
    
    
    
def load_cicids():
    import os
    CICIDS_PATH = "C:/Users/user/Desktop/NIDS_EFS_Project/Datasets/CIC-IDS- 2017/"
    
    files = [f for f in os.listdir(CICIDS_PATH) if f.endswith(".csv")]
    dfs = []
    for f in files:
        df = pd.read_csv(CICIDS_PATH + f, encoding="utf-8", low_memory=False)
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    data.columns = data.columns.str.strip()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    
    data["label"] = data["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    data.drop(columns=["Label"], inplace=True)
    
    for col in data.select_dtypes(include="object").columns:
        data[col] = LabelEncoder().fit_transform(data[col])
    
    X = MinMaxScaler().fit_transform(data.drop("label", axis=1))
    y = data["label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"CICIDS2017  Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_nslkdd()
    load_unsw()
    load_cicids()
    
    
    
def load_ciciot():
    import os
    CICIOT_PATH = "C:/Users/user/Desktop/NIDS_EFS_Project/Datasets/CIC_IOT_Dataset2023/MERGED_CSV/"
    
    files = sorted([f for f in os.listdir(CICIOT_PATH) if f.endswith(".csv")])[:10]
    dfs = []
    for f in files:
        df = pd.read_csv(CICIOT_PATH + f, low_memory=False)
        dfs.append(df)
        print(f"  Loaded {f} — {df.shape}")
    
    data = pd.concat(dfs, ignore_index=True)
    data.columns = data.columns.str.strip()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    
    label_col = [c for c in data.columns if "label" in c.lower() or "attack" in c.lower()][-1]
    data["label"] = data[label_col].apply(lambda x: 0 if str(x).lower() in ["benign","normal","0"] else 1)
    data.drop(columns=[label_col], inplace=True)
    
    for col in data.select_dtypes(include="object").columns:
        data[col] = LabelEncoder().fit_transform(data[col])
    
    X = MinMaxScaler().fit_transform(data.drop("label", axis=1))
    y = data["label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"CICIoT2023  Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_nslkdd()
    load_unsw()
    load_cicids()
    load_ciciot()