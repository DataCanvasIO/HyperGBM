# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from sklearn.utils import Bunch
from sklearn.inspection import permutation_importance
import numpy as np


def feature_importance_batch(estimators, X, y, scoring=None, n_repeats=5,
                             n_jobs=None, random_state=None):
    """Evaluate the importance of features of a set of estimators

    Parameters
    ----------
    estimator : list
        A set of estimators that has already been :term:`fitted` and is compatible
        with :term:`scorer`.

    X : ndarray or DataFrame, shape (n_samples, n_features)
        Data on which permutation importance will be computed.

    y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
        Targets for supervised or `None` for unsupervised.

    scoring : string, callable or None, default=None
        Scorer to use. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer is used.

    n_repeats : int, default=5
        Number of times to permute a feature.

    n_jobs : int or None, default=None
        The number of jobs to use for the computation.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, or None, default=None
        Pseudo-random number generator to control the permutations of each
        feature. See :term:`random_state`.

    Returns
    -------
    result : Bunch
        Dictionary-like object, with attributes:

        importances_mean : ndarray, shape (n_features, )
            Mean of feature importance over `n_repeats`.
        importances_std : ndarray, shape (n_features, )
            Standard deviation over `n_repeats`.
        importances : ndarray, shape (n_features, n_repeats)
            Raw permutation importance scores.
    """
    importances = []

    for est in estimators:
        importance = permutation_importance(est, X, y, scoring, n_repeats, n_jobs, random_state)
        importances.append(importance.importances)

    importances = np.reshape(np.stack(importances, axis=2), (X.shape[1], -1), 'F')
    bunch = Bunch(importances_mean=np.mean(importances, axis=1),
                  importances_std=np.std(importances, axis=1),
                  importances=importances)
    return bunch
