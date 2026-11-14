
import os
import sys

sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.clustering import KMeans
from src.data_utils import load_full_train, standardize

TASK = "image"                                  # task
N_CLASSES = {"image": 50, "sentiment": 3}[TASK]
KS = {"sentiment": [2, 3, 4, 5, 6, 8, 10],
      "image": [10, 20, 30, 50, 75, 100]}[TASK]
SEEDS = (0, 1, 2, 3, 4)
os.makedirs("results", exist_ok=True)

X, y = load_full_train(TASK)
X = standardize(X)[0]                               # preprocessing


def purity(y, c):
    return sum(np.bincount(y[c == j]).max() for j in np.unique(c)) / len(y)


def scores(X, y, km):
    return dict(inertia=round(km.inertia_, 1), n_iter=km.n_iter_,
                ARI=round(adjusted_rand_score(y, km.labels_), 4),
                purity=round(purity(y, km.labels_), 4),
                silhouette=round(silhouette_score(
                    X, km.labels_, sample_size=3000, random_state=0), 4))


def best_of(X, k, seeds=SEEDS):
    runs = [KMeans(k=k, seed=s).fit(X) for s in seeds]
    return min(runs, key=lambda m: m.inertia_)


# 1-> k = number of true classes, several random starts
print(f"\n=== {TASK}: k = {N_CLASSES}, different seeds ===")
rows = []
for s in SEEDS:
    km = KMeans(k=N_CLASSES, seed=s).fit(X)
    rows.append({"seed": s, **scores(X, y, km)})
df1 = pd.DataFrame(rows)
print(df1.to_string(index=False))
df1.to_csv(f"results/kmeans_{TASK}_seeds.csv", index=False)

# 2 -> vary the number of clusters
print(f"\n=== {TASK}: varying k (best of {len(SEEDS)} seeds) ===")
rows = []
for k in KS:
    km = best_of(X, k)
    rows.append({"k": k, **scores(X, y, km)})
    print(rows[-1])
df2 = pd.DataFrame(rows)
df2.to_csv(f"results/kmeans_{TASK}_vary_k.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
for a, col in zip(ax, ["inertia", "silhouette", "ARI"]):
    a.plot(df2["k"], df2[col], marker="o")
    a.set_xlabel("k"); a.set_ylabel(col)
fig.suptitle(f"k-means on {TASK}: effect of k")
plt.tight_layout(); plt.savefig(f"results/kmeans_{TASK}_vary_k.png", dpi=150)
plt.close(fig)

# 3 -> which classes end up in which cluster? 
km = best_of(X, N_CLASSES)
table = pd.crosstab(km.labels_, y, rownames=["cluster"], colnames=["label"])
table.to_csv(f"results/kmeans_{TASK}_cluster_vs_label.csv")
print(f"\n=== {TASK}: share of each cluster made of each true label ===")
share = table.div(table.sum(axis=1), axis=0).round(2)
if TASK == "sentiment":
    print(share.to_string())          # label 0=Neg, 1=Neu, 2=Pos
else:
    top = pd.DataFrame({"size": table.sum(axis=1),
                        "top_label": table.idxmax(axis=1),
                        "top_share": share.max(axis=1)})
    print(top.sort_values("top_share").to_string())

print("\nSaved tables and plots to results/")