import pytest
from unittest import TestCase

import numpy as np
from ..stratified_group_kfold import StratifiedGroupKfold


class TestStratifiedGroupKfold(TestCase):
    def test_split(self):
        seed = 0
        np.random.seed(seed)

        def _test(n_class, n_samples, n_splits):
            x = np.zeros((n_samples, n_class))  # This is not used in the split
            y = np.random.randint(0, n_class, size=n_samples)
            g = np.random.randint(0, 200, size=n_samples)
            sgkf = StratifiedGroupKfold(n_splits=n_splits, random_state=seed, split_groups=g)

            for train_index, val_index in sgkf.split(x, y):
                self.assertEqual(len(set(train_index) & set(val_index)), 0)
                train_y = y[train_index]
                val_y = y[val_index]
                t_unique, t_count = np.unique(train_y, return_counts=True)
                v_unique, v_count = np.unique(val_y, return_counts=True)
                for i in range(1, n_class):
                    self.assertAlmostEqual(
                        100 * t_count[0] / v_count[0], 100 * t_count[i] / v_count[i], delta=20)
                train_groups = g[train_index]
                val_groups = g[val_index]
                self.assertEqual(len(np.intersect1d(train_groups, val_groups)), 0)

        _test(2, 500, 2)
        _test(2, 500, 3)
        _test(2, 500, 5)
        _test(3, 500, 2)
        _test(3, 500, 3)
        _test(3, 500, 5)
