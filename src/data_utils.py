import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import load_task as _load_raw, get_split


DATA_DIR = "data"
SEED = 42

def load_task(task):
    X, y, _, _ = _load_raw(task)                
    train_idx, val_idx = get_split(task, y)     
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]
    # -> X_train, X_val, y_train, y_val

def load_test(task, data_dir=DATA_DIR):
    df = pd.read_csv(f"{data_dir}/{task}_test.csv")
    ids = df["id"].to_numpy()
    X = df.drop(columns=["id"]).to_numpy(dtype=np.float32)
    return ids, X

def load_full_train(task, data_dir=DATA_DIR):
    """All labeled data, for the final retrain."""
    df = pd.read_csv(f"{data_dir}/{task}_train.csv")
    return (df.drop(columns=["id", "label"]).to_numpy(dtype=np.float32),
            df["label"].to_numpy(dtype=np.int64))

def standardize(X_train, *others):
    """Fit mean/std on X_train only (no leakage), apply to the rest."""
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-8
    return [(X_train - mu) / sd] + [(X - mu) / sd for X in others]