from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class _Node:
    probabilities: np.ndarray
    feature: int = -1
    threshold: float = 0.0
    left: '_Node | None' = None
    right: '_Node | None' = None


class DecisionTree:

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, criterion='gini', max_thresholds=None,
                 random_state=None):
        if max_depth is not None and max_depth < 0:
            raise ValueError('max_depth must be nonnegative or None')
        if min_samples_split < 2 or min_samples_leaf < 1:
            raise ValueError('Invalid minimum sample counts')
        if criterion not in ('gini', 'entropy'):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if max_thresholds is not None and max_thresholds < 1:
            raise ValueError('max_thresholds must be positive or None')
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.max_thresholds = max_thresholds
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
            raise ValueError('Expected nonempty X (n, d) and y (n,)')
        if not np.isfinite(X).all():
            raise ValueError('X must contain only finite numbers')
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        if self.n_features_in_ == 0:
            raise ValueError('X must have at least one feature')
        self.n_classes_ = len(self.classes_)
        self._feature_count = self._resolve_features(self.n_features_in_)
        self._rng = np.random.default_rng(self.random_state)
        self._X = X
        self._y = encoded
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)
        self.root_ = self._grow(np.arange(len(y)), depth=0)
        total = self.feature_importances_.sum()
        if total:
            self.feature_importances_ /= total
        del self._X, self._y
        return self

    def _resolve_features(self, n):
        m = self.max_features
        if m is None:
            return n
        if m == 'sqrt':
            return max(1, int(np.sqrt(n)))
        if m == 'log2':
            return max(1, int(np.log2(n)))
        if isinstance(m, (int, np.integer)) and not isinstance(m, (bool, np.bool_)) and 1 <= m <= n:
            return int(m)
        if isinstance(m, (float, np.floating)) and 0 < m <= 1:
            return max(1, int(m * n))
        raise ValueError('max_features must be None, sqrt, log2, an integer in [1, d], or a fraction in (0, 1]')

    def _impurity(self, counts):
        p = counts / np.maximum(counts.sum(axis=-1, keepdims=True), 1)
        if self.criterion == 'gini':
            return 1 - np.sum(p * p, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            logp = np.zeros_like(p, dtype=float)
            np.log2(p, out=logp, where=p > 0)
        return -np.sum(p * logp, axis=-1)

    def _grow(self, indices, depth):
        n = len(indices)
        counts = np.bincount(self._y[indices], minlength=self.n_classes_)
        node = _Node(counts / n)
        if (counts.max() == n or n < self.min_samples_split or
                n < 2 * self.min_samples_leaf or
                (self.max_depth is not None and depth >= self.max_depth)):
            return node
        parent_impurity = float(self._impurity(counts))
        features = self._rng.choice(self.n_features_in_, self._feature_count, replace=False)
        best_gain = 0.0
        best_feature = -1
        best_cut = -1
        best_order = None
        for feature in features:
            values = self._X[indices, feature]
            order = np.argsort(values, kind='stable')
            sorted_values = values[order]
            valid = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
            valid = valid[(valid + 1 >= self.min_samples_leaf) &
                          (n - valid - 1 >= self.min_samples_leaf)]
            if not len(valid):
                continue
            if self.max_thresholds is not None and len(valid) > self.max_thresholds:
                positions = np.linspace(0, len(valid) - 1, self.max_thresholds).astype(int)
                valid = valid[positions]
            labels = self._y[indices[order]]
            # Count labels in the intervals between candidate split positions.
            # This avoids allocating an (n_samples, n_classes) one-hot matrix.
            buckets = np.searchsorted(valid, np.arange(n), side='left')
            codes = buckets * self.n_classes_ + labels
            interval_counts = np.bincount(
                codes, minlength=(len(valid) + 1) * self.n_classes_
            ).reshape(len(valid) + 1, self.n_classes_)
            left_counts = np.cumsum(interval_counts[:-1], axis=0)
            right_counts = counts - left_counts
            left_n = valid + 1
            right_n = n - left_n
            weighted = (left_n * self._impurity(left_counts) +
                        right_n * self._impurity(right_counts)) / n
            gains = parent_impurity - weighted
            best_at = int(np.argmax(gains))
            if gains[best_at] > best_gain + 1e-12:
                best_gain = float(gains[best_at])
                best_feature = int(feature)
                best_cut = int(valid[best_at])
                best_order = order
        if best_feature < 0:
            return node
        ordered_indices = indices[best_order]
        left_indices = ordered_indices[:best_cut + 1]
        right_indices = ordered_indices[best_cut + 1:]
        a = float(self._X[ordered_indices[best_cut], best_feature])
        b = float(self._X[ordered_indices[best_cut + 1], best_feature])
        node.feature = best_feature
        node.threshold = a + (b - a) / 2
        self.feature_importances_[best_feature] += n * best_gain
        node.left = self._grow(left_indices, depth + 1)
        node.right = self._grow(right_indices, depth + 1)
        return node

    def predict_proba(self, X):
        if not hasattr(self, 'root_'):
            raise RuntimeError('Call fit before predict')
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError('X has the wrong shape')
        if not np.isfinite(X).all():
            raise ValueError('X must contain only finite numbers')
        result = np.empty((len(X), self.n_classes_), dtype=float)
        stack = [(self.root_, np.arange(len(X)))]
        while stack:
            node, rows = stack.pop()
            if not len(rows):
                continue
            if node.feature == -1:
                result[rows] = node.probabilities
            else:
                left_mask = X[rows, node.feature] <= node.threshold
                stack.append((node.left, rows[left_mask]))
                stack.append((node.right, rows[~left_mask]))
        return result

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]
