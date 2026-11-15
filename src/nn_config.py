
N_CLASSES = {"image": 50, "sentiment": 3}

# Starting point for tuning (used by tune.py).
DEFAULTS = dict(hidden_sizes=(256, 64), activation="relu", dropout=0.0,
                lr=1e-3, weight_decay=0.0, batch_size=64, optimizer="adam")

# Final settings. UPDATE after tuning.
#   hp     = the hyperparameters chosen by tuning
#   epochs = best epoch from tuning (used when retraining on all labeled data)
FINAL = {
    "image": dict(
        hp=dict(hidden_sizes=(512, 256), activation="relu", dropout=0.0,
                lr=1e-3, weight_decay=1e-3, batch_size=64, optimizer="adam"),
        epochs=44),
    "sentiment": dict(
        hp=dict(hidden_sizes=(256, 64), activation="relu", dropout=0.0,
                lr=1e-4, weight_decay=0.0, batch_size=64, optimizer="adam"),
        epochs=5),
}