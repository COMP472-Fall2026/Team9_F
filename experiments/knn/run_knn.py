import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from knn import KNN, vote
from preprocessing import TASKS, Preprocessor, get_split, load_task

RESULTS = Path("results")
K_VALUES = [1, 3, 5, 10, 20, 50, 100]
SENTIMENT = ["Negative", "Neutral", "Positive"]


def configs(task, manhattan):
    metrics = ["euclidean", "cosine"] + ["manhattan"] * manhattan
    scalings = ["none", "standardize"] + ["sqrt"] * (task == "image")
    return [(s, 0, m) for s in scalings for m in metrics] + [("standardize", 50, "cosine")]


def predict(cfg, X_fit, y_fit, X_query):
    prep = Preprocessor(cfg["scaling"], cfg["pca"]).fit(X_fit)
    model = KNN(cfg["k"], cfg["metric"], cfg["weighting"]).fit(prep.transform(X_fit), y_fit)
    return model.predict(prep.transform(X_query))


def plot_k(task, results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, weighting in zip(axes, ("uniform", "distance")):
        for name, g in results[results.weighting == weighting].groupby(["scaling", "pca", "metric"]):
            ax.plot(g.k, g.val_acc, marker="o", label=str(name))
        ax.set(xscale="log", title=f"{task}: {weighting} voting", xlabel="k", ylabel="validation accuracy")
    axes[0].legend()
    fig.savefig(RESULTS / f"knn_{task}_acc_vs_k.png")
    plt.close(fig)


def plot_confusion(task, cm):
    fig, ax = plt.subplots()
    ax.imshow(cm)
    if task == "sentiment":
        ax.set_xticks(range(3), SENTIMENT)
        ax.set_yticks(range(3), SENTIMENT)
        for (i, j), n in np.ndenumerate(cm):
            ax.text(j, i, n)
    ax.set(title=f"{task} k-NN confusion matrix", xlabel="predicted", ylabel="true")
    fig.savefig(RESULTS / f"knn_{task}_confusion.png")
    plt.close(fig)


def sweep(task, X, y, manhattan):
    train, val = get_split(task, y)
    rows = []
    for scaling, pca, metric in configs(task, manhattan):
        prep = Preprocessor(scaling, pca).fit(X[train])
        start = time.perf_counter()
        dist, idx = KNN(max(K_VALUES), metric).fit(prep.transform(X[train]), y[train]).kneighbors(prep.transform(X[val]))
        seconds = round(time.perf_counter() - start, 2)
        for k in K_VALUES:
            for weighting in ("uniform", "distance"):
                pred = vote(y[train][idx[:, :k]], dist[:, :k], TASKS[task][1], weighting)
                rows.append(dict(scaling=scaling, pca=pca, metric=metric, k=k, weighting=weighting, val_acc=(pred == y[val]).mean(), seconds=seconds))

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / f"knn_{task}_sweep.csv", index=False)
    plot_k(task, results)
    best = json.loads(results.loc[results.val_acc.idxmax(), ["scaling", "pca", "metric", "k", "weighting", "val_acc"]].to_json())
    (RESULTS / f"knn_{task}_best.json").write_text(json.dumps(best, indent=2))
    print(f"[{task}] best: {best}")

    pred = predict(best, X[train], y[train], X[val])
    pd.DataFrame({"row": val, "true": y[val], "pred": pred}).to_csv(RESULTS / f"knn_{task}_val_predictions.csv", index=False)
    cm = np.zeros((TASKS[task][1], TASKS[task][1]), dtype=int)
    np.add.at(cm, (y[val], pred), 1)
    recall = cm.diagonal() / cm.sum(1)
    if task == "sentiment":
        print("  recall:", dict(zip(SENTIMENT, recall.round(3).tolist())))
    else:
        print("  hardest classes:", np.argsort(recall)[:5])
        mistakes = cm.copy()
        np.fill_diagonal(mistakes, 0)
        top = np.argsort(-mistakes, axis=None)[:5]
        print("  most confused [true, pred]:", np.column_stack(np.unravel_index(top, cm.shape)).tolist())
    plot_confusion(task, cm)


def final(task, X, y, X_test, ids):
    best = json.loads((RESULTS / f"knn_{task}_best.json").read_text())
    pd.DataFrame({"id": ids, "label": predict(best, X, y, X_test)}).to_csv(RESULTS / f"knn_{task}_test_predictions.csv", index=False)
    print(f"[{task}] wrote test predictions with {best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["image", "sentiment", "all"], default="all")
    parser.add_argument("--manhattan", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    for task in TASKS if args.task == "all" else [args.task]:
        X, y, X_test, ids = load_task(task)
        final(task, X, y, X_test, ids) if args.final else sweep(task, X, y, args.manhattan)