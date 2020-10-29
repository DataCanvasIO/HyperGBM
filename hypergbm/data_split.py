# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from .hyper_gbm import HyperGBM
from .search_space import search_space_general
from hypernets.searchers.evolution_searcher import EvolutionSearcher


def split_by_model(X_train, y_train, X_test, eval_size=0.3, max_test_samples=None, random_state=9527, cache_dir=None,
                   max_trails=25):
    """Split data by model which trained to estimate whether a sample in train set is
    similar with samples in test set.

    :param X_train:
    :param y_train:
    :param X_test:
    :param eval_size:
    :return:
    """
    assert X_train.shape[1] == X_test.shape[1]
    assert len(set(X_train.columns.to_list()) - set(X_test.columns.to_list())) == 0
    if isinstance(eval_size, float):
        assert eval_size < 1.0
        eval_size = int(X_train.shape[0] * eval_size)

    target_col = '__split_by_model__target__'

    if max_test_samples is not None and max_test_samples < X_test.shape[0]:
        X_test, _ = train_test_split(X_test, train_size=max_test_samples, shuffle=True, random_state=random_state)

    train_size = np.min([X_test.shape[0], int(X_train.shape[0] * 0.2)])
    X_train_train, X_train_remained, y_train_train, y_train_remained = train_test_split(X_train, y_train, train_size=train_size,
                                                                            shuffle=True,
                                                                            random_state=random_state)

    X_train_train[target_col] = 0
    X_test[target_col] = 1

    X_merge = pd.concat([X_train_train, X_test], axis=0)
    y = X_merge.pop(target_col)

    X_train, X_eval, y_train, y_eval = train_test_split(X_merge, y, train_size=0.7, shuffle=True, stratify=y,
                                                        random_state=random_state)
    searcher = EvolutionSearcher(search_space_general, 20, 10, regularized=True, optimize_direction='max')

    hypermodel = HyperGBM(searcher, task='classification', reward_metric='auc', cache_dir=cache_dir)

    hypermodel.search(X_train, y_train, X_eval, y_eval, max_trails=max_trails)
    estimator = hypermodel.final_train(hypermodel.get_best_trail().space_sample, X_merge, y)

    proba = estimator.predict_proba(X_train_remained)[:, 1]

    order = np.argsort(proba)

    X_train_train.pop(target_col)
    X_eval = X_train_remained.iloc[order[-eval_size:]]
    y_eval = y_train_remained.iloc[order[-eval_size:]]
    X_train = pd.concat([X_train_train, X_train_remained.iloc[order[:-eval_size]]], axis=0)
    y_train = pd.concat([y_train_train, y_train_remained.iloc[order[:-eval_size]]], axis=0)
    return X_train, X_eval, y_train, y_eval
