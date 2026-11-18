import unittest
from pathlib import Path
import importlib.util
import numpy as np
from src.decision_tree import DecisionTree
from src.random_forest import RandomForest


def load_original():
    path = Path(__file__).resolve().parent / 'reference_decision_tree.py'
    spec = importlib.util.spec_from_file_location('original_decision_tree', path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DecisionTree


class OptimizationTests(unittest.TestCase):
    def test_original_vs_optimized(self):
        Original = load_original()
        rng = np.random.default_rng(101)
        for n, d, classes in [(80, 7, 3), (140, 13, 50)]:
            X = rng.normal(size=(n, d)).astype(np.float32)
            X[:, 0] = np.round(X[:, 0], 1)  # repeated feature values
            y = rng.integers(0, classes, size=n)
            for criterion in ('gini', 'entropy'):
                for thresholds in (None, 8, 32):
                    params = dict(max_depth=5, min_samples_leaf=3,
                                  max_features='sqrt', max_thresholds=thresholds,
                                  criterion=criterion, random_state=42)
                    original = Original(**params).fit(X, y)
                    optimized = DecisionTree(**params).fit(X, y)
                    np.testing.assert_array_equal(original.predict(X), optimized.predict(X))
                    np.testing.assert_allclose(original.predict_proba(X), optimized.predict_proba(X))
                    np.testing.assert_allclose(original.feature_importances_, optimized.feature_importances_)

    def test_forest_soft_and_hard_votes_valid(self):
        rng = np.random.default_rng(17)
        X = rng.normal(size=(100, 9))
        y = rng.integers(0, 5, size=100)
        forest = RandomForest(n_estimators=9, max_depth=6,
                              max_thresholds=8, random_state=42).fit(X, y)
        hard = forest.predict(X)
        soft = forest.classes_[forest.predict_proba(X).argmax(axis=1)]
        self.assertEqual(hard.shape, (100,))
        self.assertEqual(soft.shape, (100,))
        self.assertTrue(np.isin(soft, forest.classes_).all())

if __name__ == '__main__':
    unittest.main()
