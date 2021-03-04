import pytest
from unittest import TestCase

import numpy as np
from ..stratified_group_kfold import StratifiedGroupKFold, RepeatedStratifiedGroupKFold


class TestStratifiedGroupKfold(TestCase):
    def test_split(self):
        seed = 42
        np.random.seed(seed)

        def _test(n_class, n_samples, n_splits):
            x = np.zeros((n_samples, n_class))  # This is not used in the split
            y = np.random.randint(0, n_class, size=n_samples)
            g = np.random.randint(0, 200, size=n_samples)
            sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=seed, split_groups=g)

            for train_index, val_index in sgkf.split(x, y):
                self.assertEqual(len(set(train_index) & set(val_index)), 0)
                train_y = y[train_index]
                val_y = y[val_index]
                t_unique, t_count = np.unique(train_y, return_counts=True)
                v_unique, v_count = np.unique(val_y, return_counts=True)
                for i in range(1, n_class):
                    self.assertAlmostEqual(
                        100 * t_count[i] / len(y), 100 * v_count[i] / len(y), delta=30)
                train_groups = g[train_index]
                val_groups = g[val_index]
                self.assertEqual(len(np.intersect1d(train_groups, val_groups)), 0)

        _test(2, 2000, 2)
        _test(2, 2000, 3)
        _test(2, 2000, 4)
        _test(3, 2000, 2)
        _test(3, 2000, 3)
        _test(3, 2000, 4)

    def test_repeated_multilabel_split(self):
        seed = 0
        np.random.seed(seed)

        def _test(n_class, n_samples, n_splits):
            x = np.zeros((n_samples, n_class))  # This is not used in the split
            y = np.random.randint(0, 2, size=(n_samples, n_class))
            g = np.random.randint(0, 200, size=n_samples)
            sgkf = RepeatedStratifiedGroupKFold(n_splits=n_splits, random_state=seed,
                                                split_groups=g, concat=True, n_repeats=5)

            for train_index, val_index in sgkf.split(x, y):
                self.assertEqual(len(set(train_index) & set(val_index)), 0)
                train_y = y[train_index]
                val_y = y[val_index]
                for i in range(n_class):
                    t_unique, t_count = np.unique(train_y[:, i], return_counts=True)
                    v_unique, v_count = np.unique(val_y[:, i], return_counts=True)
                    self.assertAlmostEqual(
                        100 * t_count[0] / len(y), 100 * v_count[0] / len(y), delta=30)
                    self.assertAlmostEqual(
                        100 * t_count[1] / len(y), 100 * v_count[1] / len(y), delta=30)
                train_groups = g[train_index]
                val_groups = g[val_index]
                self.assertEqual(len(np.intersect1d(train_groups, val_groups)), 0)

        _test(2, 2000, 2)
        _test(2, 2000, 3)
        _test(2, 2000, 4)
        _test(3, 2000, 2)
        _test(3, 2000, 3)
        _test(3, 2000, 4)
