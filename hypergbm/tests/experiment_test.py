# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer

from hypergbm import HyperGBM, CompeteExperiment
from hypergbm.search_space import search_space_general
from hypernets.core.searcher import OptimizeDirection
from hypernets.experiment import GeneralExperiment
from hypernets.searchers import RandomSearcher
from tabular_toolbox.datasets import dsutils


class Test_HyperGBM():

    def test_exp(self):
        rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)
        hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
                      cache_dir=f'hypergbm_cache',
                      callbacks=[])

        df = dsutils.load_bank().head(1000)
        df.drop(['id'], axis=1, inplace=True)
        y = df.pop('y')

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=9527)

        experiment = GeneralExperiment(hk, X_train, y_train, X_test)
        experiment.run(use_cache=True, max_trails=5)
        best_trial = experiment.hyper_model.get_best_trail()
        estimator = experiment.hyper_model.final_train(best_trial.space_sample, X_train, y_train)
        score = estimator.predict(X_test)
        result = estimator.evaluate(X_test, y_test)

    def test_compete(self):
        rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)
        hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
                      cache_dir=f'hypergbm_cache',
                      callbacks=[])

        df = dsutils.load_bank().head(1000)
        df.drop(['id'], axis=1, inplace=True)
        y = df.pop('y')

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=9527)

        experiment = CompeteExperiment('binary', hk, X_train, y_train, X_test,
                                       scorer=get_scorer('roc_auc_ovr'),
                                       drop_feature_with_collinearity=True,
                                       drift_detection=True,
                                       mode='two-stage',
                                       n_est_feature_importance=5,
                                       importance_threshold=1e-5,
                                       ensemble_size=5
                                       )
        experiment.run(use_cache=True, max_trails=10)
        # scores = experiment.estimator.predict(X_test)
