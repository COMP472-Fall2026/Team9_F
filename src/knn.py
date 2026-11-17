import numpy as np


def vote(labels, dist, n_classes, weighting="uniform"):
    weights = 1 / (dist + 1e-8) if weighting == "distance" else np.ones(labels.shape)
    votes = np.zeros((len(labels), n_classes))
    np.add.at(votes, (np.arange(len(labels))[:, None], labels), weights)
    return votes.argmax(1)


class KNN:
    def __init__(self, k=5, metric="euclidean", weighting="uniform"):
        self.k, self.metric, self.weighting = k, metric, weighting

    def fit(self, X, y):
        self.X, self.y, self.n_classes = X.astype(np.float32), y, y.max() + 1
        return self

    def distances(self, A):
        if self.metric == "euclidean":
            squared = (A**2).sum(1)[:, None] + (self.X**2).sum(1) - 2 * A @ self.X.T
            return np.sqrt(np.maximum(squared, 0))
        if self.metric == "cosine":
            unit = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
            return 1 - unit(A) @ unit(self.X).T
        return np.stack([np.abs(a - self.X).sum(1) for a in A])

    def kneighbors(self, X, k=None, batch_size=500):
        k = k or self.k
        dists, indices = [], []
        for start in range(0, len(X), batch_size):
            d = self.distances(X[start : start + batch_size].astype(np.float32))
            idx = np.argpartition(d, k - 1, axis=1)[:, :k]
            idx = np.take_along_axis(idx, np.argsort(np.take_along_axis(d, idx, 1), 1), 1)
            dists.append(np.take_along_axis(d, idx, 1))
            indices.append(idx)
        return np.vstack(dists), np.vstack(indices)

    def predict(self, X):
        dist, idx = self.kneighbors(X)
        return vote(self.y[idx], dist, self.n_classes, self.weighting)
