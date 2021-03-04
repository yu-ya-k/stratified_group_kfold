import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection._split import _BaseKFold


class BaseStratifiedGroupKFold(_BaseKFold):
    def __init__(self, n_splits, shuffle=True, random_state=None, split_groups=None, concat=False):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        assert isinstance(split_groups, np.ndarray) and split_groups.ndim == 1
        self.split_groups = split_groups
        self.concat = concat

    def eval_y_counts_per_fold(self, y_counts, fold, labels_num, y_counts_per_fold, y_distr):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(self.n_splits)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    def concat_multilabel(self, y):
        y = np.asarray(y).astype(str)
        y_concat = pd.Series(['']*y.shape[0])
        for i in range(y.shape[1]):
            y_concat += y[:, i]
        le = LabelEncoder()
        y_concat = le.fit_transform(y_concat)
        return y_concat


class StratifiedGroupKFold(BaseStratifiedGroupKFold):
    def __init__(self, n_splits, shuffle=True, random_state=None, split_groups=None, concat=False):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state,
                         split_groups=split_groups, concat=concat)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        if not self.concat:
            y = np.asarray(y)
        else:
            y = self.concat_multilabel(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, self.split_groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1
        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)
        groups_and_y_counts = list(y_counts_per_group.items())

        if self.shuffle:
            rnd = check_random_state(self.random_state)
            rnd.shuffle(groups_and_y_counts)

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


class RepeatedStratifiedGroupKFold(BaseStratifiedGroupKFold):
    def __init__(self, n_splits, shuffle=True, random_state=None, split_groups=None, concat=False, n_repeats=2):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state,
                         split_groups=split_groups, concat=concat)
        self.n_repeats = n_repeats

    def _iter_test_indices(self, X=None, y=None, groups=None):
        if not self.concat:
            y = np.asarray(y)
        else:
            y = self.concat_multilabel(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        rnd = check_random_state(self.random_state)
        labels_num = np.max(y) + 1
        for repeat in range(self.n_repeats):
            y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
            y_distr = Counter()
            for label, g in zip(y, self.split_groups):
                y_counts_per_group[g][label] += 1
                y_distr[label] += 1
            y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
            groups_per_fold = defaultdict(set)
            groups_and_y_counts = list(y_counts_per_group.items())

            if self.shuffle:
                rnd.shuffle(groups_and_y_counts)

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
