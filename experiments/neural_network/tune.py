import os
import sys

sys.path.insert(0, ".")
import pandas as pd

from src.data_utils import load_task, standardize
from src.neural_network import count_params, train_model

TASK = "sentiment"                       # task
N_CLASSES = {"image": 50, "sentiment": 3}[TASK]
os.makedirs("results", exist_ok=True)

X_tr, X_val, y_tr, y_val = load_task(TASK)
X_tr, X_val = standardize(X_tr, X_val)     # mean/std from train only

# Current best settings
BEST = dict(hidden_sizes=(256, 64), activation="relu", dropout=0.0,
            lr=1e-4, weight_decay=0.0, batch_size=64, optimizer="adam")        

"""
 IMAGE FINAL: hidden_sizes=(512, 256), activation="relu", dropout=0.0,
              lr=1e-3, weight_decay=1e-3, batch_size=64, optimizer="adam", best epoch 17
"""

def sweep(name, variants):
    rows = []
    for ov in variants:
        cfg = {**BEST, **ov}
        print(f"\n--- running {ov} ---")
        model, h = train_model(X_tr, y_tr, X_val, y_val,
                               n_classes=N_CLASSES, verbose=True, **cfg)
        b = h["best_epoch"]
        rows.append({"setting": ", ".join(f"{k}={v}" for k, v in ov.items()),
                     "train_acc": round(h["train_acc"][b], 4),
                     "val_acc": round(h["val_acc"][b], 4),
                     "gap": round(h["train_acc"][b] - h["val_acc"][b], 4),
                     "best_epoch": b + 1,
                     "train_time_s": round(h["train_time"], 1),
                     "params": count_params(model)})
        print(rows[-1])
    df = pd.DataFrame(rows)
    print(f"\n=== {TASK}: sweep over {name} ===")
    print(df.to_string(index=False))
    df.to_csv(f"results/nn_{TASK}_{name}.csv", index=False)
    return df


# sweep("lr", [dict(lr=v) for v in (1e-2, 1e-3, 1e-4)])

"""
sweep("hidden_sizes", [dict(hidden_sizes=h) for h in
      ((), (64,), (256,), (256, 64), (512, 256))])
"""

# sweep("dropout", [dict(dropout=p) for p in (0.0, 0.2, 0.5)])
# sweep("weight_decay", [dict(weight_decay=w) for w in (0.0, 1e-5, 1e-4, 1e-3, 1e-2)])
#sweep("batch_size", [dict(batch_size=b) for b in (32, 64, 128, 512)])
#sweep("activation", [dict(activation=a) for a in ("relu", "tanh", "leaky_relu")])
#sweep("optimizer", [dict(optimizer="adam", lr=1e-4), dict(optimizer="sgd", lr=1e-2)])
"""
sweep("round2", [
    dict(weight_decay=5e-4), dict(weight_decay=2e-3),
    dict(dropout=0.2, weight_decay=1e-3),
    dict(dropout=0.2, weight_decay=5e-4),
    dict(lr=5e-4), dict(lr=2e-3),
])
"""