"""k-NN classifier from scratch (numpy only)"""
import numpy as np


def vote(labels, dist, n_classes, weighting="uniform"):
    """Majority vote over the k neighbours, uniform or weighted by 1/distance"""
    # small rank penalty so ties go to the closest neighbour's class
    # (argmax would otherwise just pick the lowest class index)
    rank_penalty = 1e-6 * np.arange(labels.shape[1])
    weights = 1 / (dist + 1e-8) if weighting == "distance" else np.ones(labels.shape) - rank_penalty
    votes = np.zeros((len(labels), n_classes))
    np.add.at(votes, (np.arange(len(labels))[:, None], labels), weights)
    return votes.argmax(1)


class KNN:
    def __init__(self, k=5, metric="euclidean", weighting="uniform"):
        self.k, self.metric, self.weighting = k, metric, weighting

    def fit(self, X, y):
        # nothing to train, we just keep the data
        self.X, self.y, self.n_classes = X.astype(np.float32), y, y.max() + 1
        return self

    def distances(self, A):
        """Distances from each query in A to every training point"""
        if self.metric == "euclidean":
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab, so one matrix product instead of a loop
            squared = (A**2).sum(1)[:, None] + (self.X**2).sum(1) - 2 * A @ self.X.T
            return np.sqrt(np.maximum(squared, 0))  # clip tiny negatives from rounding
        if self.metric == "cosine":
            unit = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
            return 1 - unit(A) @ unit(self.X).T
        # manhattan, slow since there's no matrix trick for it
        return np.stack([np.abs(a - self.X).sum(1) for a in A])

    def kneighbors(self, X, k=None, batch_size=500):
        """Distances and indices of the k nearest neighbours, closest first"""
        k = k or self.k
        dists, indices = [], []
        # batches so the distance matrix fits in memory
        for start in range(0, len(X), batch_size):
            d = self.distances(X[start : start + batch_size].astype(np.float32))
            # argpartition grabs the k smallest without sorting everything, then we sort just those k
            idx = np.argpartition(d, k - 1, axis=1)[:, :k]
            idx = np.take_along_axis(idx, np.argsort(np.take_along_axis(d, idx, 1), 1), 1)
            dists.append(np.take_along_axis(d, idx, 1))
            indices.append(idx)
        return np.vstack(dists), np.vstack(indices)

    def predict(self, X):
        dist, idx = self.kneighbors(X)
        return vote(self.y[idx], dist, self.n_classes, self.weighting)
