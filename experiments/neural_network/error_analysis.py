import os
import sys

sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data_utils import load_task, standardize
from src.nn import train_model
from src.nn_config import FINAL, N_CLASSES

SENTIMENT = ["Negative", "Neutral", "Positive"]
os.makedirs("results", exist_ok=True)

for task, cfg in FINAL.items():
    hp, n_classes = cfg["hp"], N_CLASSES[task]
    X_tr, X_val, y_tr, y_val = load_task(task)
    pred = np.load(f"results/nn_{task}_val_predictions.npy")
    assert len(pred) == len(y_val), "val predictions do not match the split"
    print(f"\n===== {task}: NN validation accuracy {(pred == y_val).mean():.4f} =====")

    #  per-class metrics 
    rep = classification_report(y_val, pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(f"results/nn_{task}_class_report.csv")
    if task == "sentiment":
        print(classification_report(y_val, pred, target_names=SENTIMENT, digits=3))
    else:
        per_class = pd.DataFrame(rep).T.iloc[:n_classes]
        print("5 weakest classes by F1:")
        print(per_class.sort_values("f1-score").head(5).round(3).to_string())

    # confusion matrix 
    cm = confusion_matrix(y_val, pred, labels=range(n_classes))
    fig, ax = plt.subplots(figsize=(5, 4) if task == "sentiment" else (9, 8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    if task == "sentiment":
        ax.set_xticks(range(3), SENTIMENT)
        ax.set_yticks(range(3), SENTIMENT)
        for (i, j), n in np.ndenumerate(cm):
            ax.text(j, i, n, ha="center", va="center")
    ax.set(title=f"NN confusion matrix ({task}, validation)",
           xlabel="predicted", ylabel="true")
    fig.tight_layout()
    fig.savefig(f"results/nn_{task}_confusion.png", dpi=150)
    plt.close(fig)
    if task == "image":
        off = cm.copy()
        np.fill_diagonal(off, 0)
        print("Most confused (true -> predicted : count):")
        for flat in np.argsort(-off, axis=None)[:10]:
            t, p = np.unravel_index(flat, off.shape)
            print(f"  {t} -> {p} : {off[t, p]}")


    # train vs validation curves 
    X_tr, X_val = standardize(X_tr, X_val)
    _, h = train_model(X_tr, y_tr, X_val, y_val, n_classes=n_classes, **hp)
    best = h["best_epoch"] + 1
    epochs = range(1, len(h["val_acc"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(epochs, h["train_acc"], label="train")
    ax[0].plot(epochs, h["val_acc"], label="validation")
    ax[0].set(xlabel="epoch", ylabel="accuracy")
    ax[1].plot(epochs, h["train_loss"], label="train")
    ax[1].plot(epochs, h["val_loss"], label="validation")
    ax[1].set(xlabel="epoch", ylabel="cross-entropy loss")
    for a in ax:
        a.axvline(best, color="gray", linestyle="--", label=f"best epoch ({best})")
        a.legend()
    fig.suptitle(f"NN training curves ({task})")
    fig.tight_layout()
    fig.savefig(f"results/nn_{task}_curves.png", dpi=150)
    plt.close(fig)

print("\nSaved reports and figures to results/")