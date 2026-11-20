import os
import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.data_utils import load_full_train, load_task, load_test, standardize
from src.nn import predict, train_model
from src.nn_config import FINAL, N_CLASSES

RETRAIN_ON_ALL = True      

os.makedirs("results", exist_ok=True)

for task, cfg in FINAL.items():
    n_classes, hp = N_CLASSES[task], cfg["hp"]
    ids, X_test = load_test(task)

    # validated model (train split only)
    X_tr, X_val, y_tr, y_val = load_task(task)
    X_tr, X_val, X_test_s = standardize(X_tr, X_val, X_test)   # stats from train split
    model, h = train_model(X_tr, y_tr, X_val, y_val, n_classes=n_classes, **hp)
    val_acc = h["val_acc"][h["best_epoch"]]
    print(f"{task}: validation accuracy {val_acc:.4f} "
          f"(best epoch {h['best_epoch'] + 1})")
    np.save(f"results/nn_{task}_val_predictions.npy", predict(model, X_val))
    pred = predict(model, X_test_s)

    # retrain on all labeled data 
    if RETRAIN_ON_ALL:
        X_full, y_full = load_full_train(task)
        X_full, X_test_f = standardize(X_full, X_test)         # stats from full train
        model, _ = train_model(X_full, y_full, n_classes=n_classes,
                               epochs=cfg["epochs"], **hp)
        pred = predict(model, X_test_f)

    # save predictions 
    assert len(pred) == len(ids) == 5000
    assert pred.min() >= 0 and pred.max() < n_classes
    pd.DataFrame({"id": ids, "label": pred.astype(int)}).to_csv(
        f"results/nn_{task}_test_predictions.csv", index=False)
    print(f"{task}: saved results/nn_test_pred_{task}.csv "
          f"(retrained on all data: {RETRAIN_ON_ALL})")
    print("  predicted class counts:", np.bincount(pred, minlength=n_classes).tolist())
