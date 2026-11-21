import os
import sys

sys.path.insert(0, ".")
import pandas as pd

from src.data_utils import load_task, standardize
from src.nn import count_params, train_model
from src.nn_config import DEFAULTS, N_CLASSES

import numpy as np

TASK = "image"  
RUN = "dropout"                     #  sweep


BEST = {**DEFAULTS,
       # "weight_decay": 1e-3,
       # "hidden_sizes": (512, 256),
        }

SWEEPS = {
    "lr": [dict(lr=v) for v in (1e-2, 1e-3, 1e-4)],
    "dropout": [dict(dropout=p) for p in (0.0, 0.2, 0.5)],
    "weight_decay": [dict(weight_decay=w) for w in (0.0, 1e-5, 1e-4, 1e-3, 1e-2)],
    "hidden_sizes": [dict(hidden_sizes=h)
                     for h in ((), (64,), (256,), (256, 64), (512, 256))],
    "batch_size": [dict(batch_size=b) for b in (32, 64, 128, 512)],
    "activation": [dict(activation=a) for a in ("relu", "tanh", "leaky_relu")],
    "optimizer": [dict(optimizer="adam", lr=BEST["lr"]),
                  dict(optimizer="sgd", lr=1e-2)],
}

os.makedirs("results", exist_ok=True)
X_tr, X_val, y_tr, y_val = load_task(TASK)

#X_tr, X_val = np.sqrt(X_tr), np.sqrt(X_val)    TEST ONLY
X_tr, X_val = standardize(X_tr, X_val)     # mean/std from train only

def sweep(name, variants):
    rows = []
    print(f"\n=== {TASK}: sweep over {name} ===")
    for ov in variants:
        model, h = train_model(X_tr, y_tr, X_val, y_val,
                               n_classes=N_CLASSES[TASK], **{**BEST, **ov})
        b = h["best_epoch"]
        rows.append({"setting": ", ".join(f"{k}={v}" for k, v in ov.items()),
                     "train_acc": round(h["train_acc"][b], 4),
                     "val_acc": round(h["val_acc"][b], 4),
                     "gap": round(h["train_acc"][b] - h["val_acc"][b], 4),
                     "best_epoch": b + 1,
                     "train_time_s": round(h["train_time"], 1),
                     "params": count_params(model)})
        print(rows[-1]["setting"], "-> val_acc", rows[-1]["val_acc"])
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    df.to_csv(f"results/nn_{TASK}_{name}.csv", index=False)

if RUN != "all" and RUN not in SWEEPS:
    raise SystemExit(f"RUN must be 'all' or one of: {list(SWEEPS)}")
for name in (list(SWEEPS) if RUN == "all" else [RUN]):
    sweep(name, SWEEPS[name])