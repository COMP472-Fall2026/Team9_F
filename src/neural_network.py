import copy
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_sizes=(256, 64),
                 activation="relu", dropout=0.0):
        super().__init__()
        acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}
        layers, prev = [], input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), acts[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))   # logits, no softmax
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    logits = model(torch.as_tensor(X, dtype=torch.float32))
    return torch.softmax(logits, dim=1).numpy()


def predict(model, X):
    return predict_proba(model, X).argmax(axis=1)


@torch.no_grad()
def evaluate(model, X, y):
    """Returns (cross-entropy loss, accuracy)."""
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    logits = model(X)
    loss = nn.functional.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


def train_model(X_tr, y_tr, X_val=None, y_val=None, n_classes=None,
                hidden_sizes=(256, 64), activation="relu", dropout=0.0,
                lr=1e-3, weight_decay=0.0, optimizer="adam",
                batch_size=64, epochs=50, patience=10, seed=42,
                verbose=False):
   
    set_seed(seed)                       # before the model is created
    n_classes = n_classes or int(np.max(y_tr)) + 1
    model = MLP(X_tr.shape[1], n_classes, hidden_sizes, activation, dropout)

    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(torch.as_tensor(X_tr, dtype=torch.float32),
                       torch.as_tensor(y_tr, dtype=torch.long))
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

    has_val = X_val is not None
    history: dict[str, Any] = {"train_loss": [], "train_acc": [],
                               "val_loss": [], "val_acc": []}
    best_acc, best_epoch, bad = -1.0, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()                    # 1. clear old gradients
            loss = loss_fn(model(xb), yb)      # 2. forward pass + loss
            loss.backward()                    # 3. backward pass (backprop)
            opt.step()                         # 4. update the weights

        tr_loss, tr_acc = evaluate(model, X_tr, y_tr)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)

        if has_val:
            va_loss, va_acc = evaluate(model, X_val, y_val)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            if verbose:
                print(f"epoch {epoch + 1:3d}  loss {tr_loss:.4f}  "
                      f"acc {tr_acc:.3f}  val_acc {va_acc:.3f}")
            if va_acc > best_acc:              # new best: remember weights
                best_acc, best_epoch, bad = va_acc, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:                              # early stopping
                bad += 1
                if patience is not None and bad >= patience:
                    break

    if has_val:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch if has_val else epochs - 1
    history["train_time"] = time.time() - t0
    return model, history