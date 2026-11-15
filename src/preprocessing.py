import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = Path("data")
SPLIT_DIR = Path("splits")
TASKS = {"image": (500, 50), "sentiment": (768, 3)}


def load_task(task):
    dims, n_classes = TASKS[task]
    cols = [f"feature_{i}" for i in range(dims)]
    train = pd.read_csv(DATA_DIR / f"{task}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{task}_test.csv")
    y = train["label"].to_numpy()
    assert y.min() >= 0 and y.max() < n_classes
    return train[cols].to_numpy(np.float32), y, test[cols].to_numpy(np.float32), test["id"].to_numpy()


def get_split(task, y, val_fraction=0.2):
    path = SPLIT_DIR / f"{task}_split.npz"
    if not path.exists():
        rng = np.random.default_rng(SEED)
        val = np.concatenate([rng.permutation(np.flatnonzero(y == c))[: round(val_fraction * (y == c).sum())] for c in np.unique(y)])
        SPLIT_DIR.mkdir(exist_ok=True)
        np.savez(path, train=np.setdiff1d(np.arange(len(y)), val), val=np.sort(val))
    with np.load(path) as f:
        train, val = f["train"], f["val"]
    assert len(train) + len(val) == len(y)
    return train, val


class Preprocessor:
    def __init__(self, scaling="none", pca=0):
        self.scaling, self.pca = scaling, pca

    def fit(self, X):
        if self.scaling == "sqrt":
            X = np.sqrt(X)
        if self.scaling == "standardize":
            self.mean, self.std = X.mean(0), X.std(0) + 1e-8
            X = (X - self.mean) / self.std
        if self.pca:
            self.center = X.mean(0)
            self.components = np.linalg.svd(X - self.center, full_matrices=False)[2][: self.pca]
        return self

    def transform(self, X):
        if self.scaling == "sqrt":
            X = np.sqrt(X)
        if self.scaling == "standardize":
            X = (X - self.mean) / self.std
        if self.pca:
            X = (X - self.center) @ self.components.T
        return X.astype(np.float32)


def explore(task, X, y, X_test, fig_dir):
    centered = X - X.mean(0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    variance = np.cumsum(s**2) / (s**2).sum()
    counts = np.bincount(y)
    print(f"\n{task}: {counts.min()}-{counts.max()} samples per class")
    print(f"values: [{X.min():.3f}, {X.max():.3f}], zeros: {(X == 0).mean():.1%}")
    print(f"feature std: [{X.std(0).min():.3f}, {X.std(0).max():.3f}]")
    print(f"row L2 norm: train {np.linalg.norm(X, axis=1).mean():.2f}, test {np.linalg.norm(X_test, axis=1).mean():.2f}")
    print(f"PCA components for 90% variance: {np.searchsorted(variance, 0.9) + 1}")

    sample = np.random.default_rng(SEED).choice(len(X), 4000, replace=False)
    projected = centered[sample] @ vt[:2].T
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    left.plot(range(1, len(variance) + 1), variance)
    left.set(title="PCA cumulative explained variance", xlabel="components", ylabel="fraction")
    right.scatter(projected[:, 0], projected[:, 1], c=y[sample], s=3)
    right.set(title="First two principal components", xlabel="PC1", ylabel="PC2")
    fig.suptitle(task)
    fig.savefig(fig_dir / f"{task}_exploration.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        X, y, X_test, _ = load_task(task)
        explore(task, X, y, X_test, args.fig_dir)
        train, val = get_split(task, y)
        print(f"split: {len(train)} train / {len(val)} validation")