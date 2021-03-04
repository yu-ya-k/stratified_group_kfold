# stratified_group_kfold
Stratified Group K-fold cross validation inheriting scikit-learn base class

## Background
K-fold cross validation can be performed in several manners like [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold) 
and [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold), 
but a method combining these two has not been officially implemented in [scikit-learn](https://scikit-learn.org/stable/index.html).

This repository implements Stratified Group K-fold cross validation with inheriting scikit-learn base K-fold class for easier application with other scikit-lean classes. Repeated option is available. 

Multilabel dataset can be dealt with [concatenating label columns](https://github.com/yu-ya-k/stratified_group_kfold/blob/3e4f9740ee596ef827e5ade8eff7b5bbe92a6f96/sgkfold/stratified_group_kfold.py#L27).

Detailed explanation of difference between each methods can be found [here](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py).

## Ref
This repository is based on implementation introduced in Kaggle notebooks by [Jakub Wasikowski](https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation) and [sin](https://www.kaggle.com/underwearfitting/how-to-properly-split-folds)
