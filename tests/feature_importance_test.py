# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer

from hypergbm.feature_importance import feature_importance_batch
from hypergbm.hyper_gbm import HyperGBM
from hypergbm.search_space import search_space_general
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers.random_searcher import RandomSearcher
from tabular_toolbox.datasets import dsutils
from tests import test_output_dir


class Test_FeatureImportance():

    def test_basic(self):
        rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)
        hk = HyperGBM(rs, task='classification', reward_metric='accuracy',
                      cache_dir=f'{test_output_dir}/hypergbm_cache')

        df = dsutils.load_bank().head(10000)
        y = df.pop('y')
        df.drop(['id'], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

        hk.search(X_train, y_train, X_test, y_test, max_trails=10, use_cache=True)
        best_trials = hk.get_top_trails(3)
        estimators = []
        for trail in best_trials:
            estimators.append(hk.load_estimator(trail.model_file))

        importances = feature_importance_batch(estimators, X_test, y_test, get_scorer('roc_auc_ovr'), n_repeats=2)
        assert importances
