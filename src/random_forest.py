import numpy as np

from src.decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', criterion='gini',
                 max_thresholds=None, bootstrap=True, max_samples=None,
                 random_state=None):
        if n_estimators < 1:
            raise ValueError('n_estimators must be positive')
        if max_samples is not None and (not isinstance(max_samples, (int, float)) or
                                        isinstance(max_samples, bool) or max_samples <= 0):
            raise ValueError('max_samples must be a positive integer, fraction, or None')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.max_thresholds = max_thresholds
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
            raise ValueError('Expected nonempty X (n, d) and y (n,)')
        if not np.isfinite(X).all():
            raise ValueError('X must contain only finite numbers')
        if not self.bootstrap and self.max_samples is not None:
            raise ValueError('max_samples requires bootstrap=True')
        n = len(y)
        if self.max_samples is None:
            sample_size = n
        elif isinstance(self.max_samples, (float, np.floating)):
            if self.max_samples > 1:
                raise ValueError('Fractional max_samples must be <= 1')
            sample_size = max(1, round(n * self.max_samples))
        else:
            sample_size = int(self.max_samples)
            if sample_size > n:
                raise ValueError('Integer max_samples cannot exceed n')
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.estimators_ = []
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_estimators):
            indices = rng.integers(0, n, size=sample_size) if self.bootstrap else np.arange(n)
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                max_features=self.max_features,
                                criterion=self.criterion,
                                max_thresholds=self.max_thresholds,
                                random_state=seed)
            self.estimators_.append(tree.fit(X[indices], y[indices]))
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0)
        return self

    def predict_proba(self, X):
        if not hasattr(self, 'estimators_'):
            raise RuntimeError('Call fit before predict')
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError('X has the wrong shape')
        scores = np.zeros((len(X), len(self.classes_)), dtype=float)
        for tree in self.estimators_:
            columns = np.searchsorted(self.classes_, tree.classes_)
            scores[:, columns] += tree.predict_proba(X)
        return scores / len(self.estimators_)

    def predict(self, X):
        if not hasattr(self, 'estimators_'):
            raise RuntimeError('Call fit before predict')
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError('X has the wrong shape')
        votes = np.zeros((len(X), len(self.classes_)), dtype=np.int32)
        rows = np.arange(len(X))
        for tree in self.estimators_:
            cols = np.searchsorted(self.classes_, tree.predict(X))
            np.add.at(votes, (rows, cols), 1)
        return self.classes_[votes.argmax(axis=1)]
