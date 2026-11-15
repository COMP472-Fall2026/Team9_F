import numpy as np

def initialize_centroids(X, k, seed=42):
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), k, replace=False)].copy()

def assign_clusters(X, centroids):
    d = (X**2).sum(1)[:, None] - 2 * X @ centroids.T + (centroids**2).sum(1)[None, :]
    return d.argmin(1)

def update_centroids(X, assignments, k, rng):
    centroids = np.empty((k, X.shape[1]))
    for j in range(k):
        pts = X[assignments == j]
        centroids[j] = pts.mean(0) if len(pts) else X[rng.integers(len(X))]
    return centroids

def compute_inertia(X, assignments, centroids):
    return ((X - centroids[assignments]) ** 2).sum()

class KMeans:
    def __init__(self, k=3, max_iter=300, tol=1e-4, seed=42):
        self.k, self.max_iter, self.tol, self.seed = k, max_iter, tol, seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        self.centroids_ = initialize_centroids(X, self.k, self.seed)
        self.n_iter_ = 0
        for _ in range(self.max_iter):
            assignments = assign_clusters(X, self.centroids_)
            new_centroids = update_centroids(X, assignments, self.k, rng)
            shift = np.linalg.norm(new_centroids - self.centroids_, axis=1).max()
            self.centroids_ = new_centroids
            self.n_iter_ += 1
            if shift < self.tol:
                break
        self.labels_ = assign_clusters(X, self.centroids_)
        self.inertia_ = compute_inertia(X, self.labels_, self.centroids_)
        return self

    def predict(self, X):
        return assign_clusters(X, self.centroids_)