import unittest

import numpy as np

from src.decision_tree import DecisionTree
from src.random_forest import RandomForest


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2]], dtype=float)
        self.y = np.array([10, 10, 10, 20, 20, 20])

        # Both impurity criteria should recover the known split at 2.5.
    def test_tree_learns_threshold(self):
        for criterion in ('gini', 'entropy'):
            tree = DecisionTree(criterion=criterion, random_state=3).fit(self.X, self.y)
            np.testing.assert_array_equal(tree.predict(self.X), self.y)
            self.assertEqual(tree.root_.feature, 0)
            self.assertAlmostEqual(tree.root_.threshold, 2.5)
            np.testing.assert_allclose(tree.predict_proba(self.X).sum(axis=1), 1)

        # A depth-zero tree predicts its root majority class for every sample.
    def test_depth_zero_majority(self):
        tree = DecisionTree(max_depth=0).fit(self.X, self.y)
        np.testing.assert_array_equal(tree.predict(self.X), np.repeat(10, len(self.y)))

        # Constant features prevent splitting; a single-class dataset stays pure.
    def test_constant_features_and_one_class(self):
        X = np.ones((6, 3))
        y = np.array([1, 1, 2, 2, 2, 1])
        tree = DecisionTree().fit(X, y)
        self.assertEqual(tree.root_.feature, -1)
        np.testing.assert_allclose(tree.predict_proba(X)[0], [0.5, 0.5])
        single = DecisionTree().fit(X, np.repeat(7, 6))
        np.testing.assert_array_equal(single.predict(X), np.repeat(7, 6))

        # A split is disallowed if either child would violate the minimum leaf size.
    def test_leaf_constraint(self):
        tree = DecisionTree(min_samples_leaf=4).fit(self.X, self.y)
        self.assertEqual(tree.root_.feature, -1)

        # Fixed random seeds should reproduce votes and class probabilities.
    def test_forest_reproducible(self):
        params = dict(n_estimators=11, max_depth=3, random_state=5)
        first = RandomForest(**params).fit(self.X, self.y)
        second = RandomForest(**params).fit(self.X, self.y)
        np.testing.assert_array_equal(first.predict(self.X), second.predict(self.X))
        np.testing.assert_allclose(first.predict_proba(self.X), second.predict_proba(self.X))
        np.testing.assert_allclose(first.predict_proba(self.X).sum(axis=1), 1)

        # Some bootstrap trees may omit a class; forest probabilities must still align.
    def test_forest_class_missing_in_bootstrap(self):
        X = np.arange(12, dtype=float).reshape(-1, 1)
        y = np.array([0] * 10 + [1] * 2)
        model = RandomForest(n_estimators=15, random_state=7).fit(X, y)
        self.assertEqual(model.predict_proba(X).shape, (12, 2))

        # Invalid inputs and prediction before fitting must raise clear exceptions.
    def test_errors(self):
        with self.assertRaises(RuntimeError):
            DecisionTree().predict(self.X)
        with self.assertRaises(ValueError):
            DecisionTree().fit(self.X, self.y[:-1])
        with self.assertRaises(ValueError):
            DecisionTree(max_features=3).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RandomForest(n_estimators=0)


if __name__ == '__main__':
    unittest.main()
