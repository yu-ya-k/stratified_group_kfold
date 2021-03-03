import random
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection._split import _BaseKFold


class StratifiedGroupKfold(_BaseKFold):
    def __init__(self, n_splits, shuffle=True, random_state=None, split_groups=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        assert isinstance(split_groups, np.ndarray) and split_groups.ndim == 1
        self.split_groups = split_groups

    def eval_y_counts_per_fold(self, y_counts, fold, labels_num, y_counts_per_fold, y_distr):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(self.n_splits)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        assert isinstance(y, np.ndarray) and y.ndim == 1

        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, self.split_groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random().shuffle(groups_and_y_counts)
        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                fold_eval = self.eval_y_counts_per_fold(y_counts, i, labels_num, y_counts_per_fold, y_distr)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        for i in range(self.n_splits):
            test_groups = groups_per_fold[i]
            test_indices = np.asarray([i for i, g in enumerate(self.split_groups) if g in test_groups])
            yield test_indices

    def split(self, X, y=None, groups=None):
        return super().split(X, y, groups)
