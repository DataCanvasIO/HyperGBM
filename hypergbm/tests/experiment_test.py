# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

from hypergbm import HyperGBM, CompeteExperiment
from hypergbm.search_space import search_space_general
from hypernets.core import OptimizeDirection, EarlyStoppingCallback
from hypernets.experiment import GeneralExperiment, ExperimentCallback, ConsoleCallback
from hypernets.searchers import RandomSearcher
from tabular_toolbox.datasets import dsutils
import numpy as np
import pandas as pd


class LogCallback(ExperimentCallback):
    def __init__(self, output_elapsed=False):
        self.logs = []
        self.experiment_elapsed = None
        self.output_elapsed = output_elapsed

    def experiment_start(self, exp):
        self.logs.append('experiment start')

    def experiment_end(self, exp, elapsed):
        self.logs.append(f'experiment end')
        if self.output_elapsed:
            self.logs.append(f'   elapsed:{elapsed}')
        self.experiment_elapsed = elapsed

    def experiment_break(self, exp, error):
        self.logs.append(f'experiment break, error:{error}')

    def step_start(self, exp, step):
        self.logs.append(f'   step start, step:{step}')

    def step_progress(self, exp, step, progress, elapsed, eta=None):
        self.logs.append(f'      progress:{progress}')
        if self.output_elapsed:
            self.logs.append(f'         elapsed:{elapsed}')

    def step_end(self, exp, step, output, elapsed):
        self.logs.append(f'   step end, step:{step}, output:{output.keys() if output is not None else ""}')
        if self.output_elapsed:
            self.logs.append(f'      elapsed:{elapsed}')

    def step_break(self, exp, step, error):
        self.logs.append(f'step break, step:{step}, error:{error}')


class Test_HyperGBM():
    def test_regression_cv(self):
        self.run_regression(cv=True)

    def test_regression_feature_reselection(self):
        self.run_regression(feature_reselection=True)

    def test_regression_pseudo_labeling(self):
        self.run_regression(pseudo_labeling=True)

    def test_regression_adversarial_validation(self):
        self.run_regression(train_test_split_strategy='adversarial_validation')

    def run_regression(self, train_test_split_strategy=None, cv=False, feature_reselection=False, pseudo_labeling=False,
                       collinearity_detection=False, drift_detection=True, max_trials=3):
        df = dsutils.load_Bike_Sharing()
        y = df.pop('count')

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=9527)

        log_callback = LogCallback(output_elapsed=True)
        rs = RandomSearcher(lambda: search_space_general(early_stopping_rounds=5, ),
                            optimize_direction='min')
        hk = HyperGBM(rs, task='regression', reward_metric='mse', cache_dir=f'hypergbm_cache', callbacks=[])
        experiment = CompeteExperiment(hk, X_train, y_train, X_test=X_test,
                                       callbacks=[log_callback],
                                       train_test_split_strategy=train_test_split_strategy,
                                       cv=cv, num_folds=3,
                                       pseudo_labeling=pseudo_labeling,
                                       scorer=get_scorer('neg_root_mean_squared_error'),
                                       collinearity_detection=collinearity_detection,
                                       drift_detection=drift_detection,
                                       feature_reselection=feature_reselection,
                                       feature_reselection_estimator_size=5,
                                       feature_reselection_threshold=1e-5,
                                       ensemble_size=10
                                       )
        pipeline = experiment.run(use_cache=True, max_trials=max_trials)
        rmse_scorer = get_scorer('neg_root_mean_squared_error')
        rmse = rmse_scorer(pipeline, X_test, y_test)
        assert rmse

    def test_multiclass_cv(self):
        self.run_multiclass(cv=True)

    def test_multiclass_pseudo_labeling(self):
        self.run_multiclass(pseudo_labeling=True)

    def test_multiclass_feature_reselection(self):
        self.run_multiclass(feature_reselection=True)

    def test_multiclass_adversarial_validation(self):
        self.run_multiclass(train_test_split_strategy='adversarial_validation')

    def run_multiclass(self, train_test_split_strategy=None, cv=False, feature_reselection=False, pseudo_labeling=False,
                       collinearity_detection=False, drift_detection=True, max_trials=3):
        df = dsutils.load_glass_uci()
        df.columns = [f'x_{c}' for c in df.columns.to_list()]
        df.pop('x_0')
        y = df.pop('x_10')
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1, stratify=y)

        rs = RandomSearcher(lambda: search_space_general(early_stopping_rounds=20, verbose=0),
                            optimize_direction=OptimizeDirection.Maximize)
        es = EarlyStoppingCallback(20, 'max')
        hk = HyperGBM(rs, reward_metric='auc', cache_dir=f'hypergbm_cache', callbacks=[es])

        log_callback = ConsoleCallback()
        experiment = CompeteExperiment(hk, X_train, y_train, X_test=X_test,
                                       callbacks=[log_callback],
                                       train_test_split_strategy=train_test_split_strategy,
                                       cv=cv, num_folds=3,
                                       pseudo_labeling=pseudo_labeling,
                                       scorer=get_scorer('roc_auc_ovr'),
                                       collinearity_detection=collinearity_detection,
                                       drift_detection=drift_detection,
                                       feature_reselection=feature_reselection,
                                       feature_reselection_estimator_size=5,
                                       feature_reselection_threshold=1e-5,
                                       ensemble_size=10
                                       )
        pipeline = experiment.run(use_cache=True, max_trials=max_trials)
        acc_scorer = get_scorer('accuracy')
        acc = acc_scorer(pipeline, X_test, y_test)
        assert acc
        auc_scorer = get_scorer('roc_auc_ovo')
        auc = auc_scorer(pipeline, X_test, y_test)
        assert auc

    def test_general_exp(self):
        rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)
        hk = HyperGBM(rs, reward_metric='accuracy', cache_dir=f'hypergbm_cache', callbacks=[])

        df = dsutils.load_bank().head(1000)
        df.drop(['id'], axis=1, inplace=True)
        y = df.pop('y')

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=9527)
        log_callback = LogCallback()
        experiment = GeneralExperiment(hk, X_train, y_train, X_test=X_test, callbacks=[log_callback])
        experiment.run(use_cache=True, max_trials=5)
        assert log_callback.logs == ['experiment start',
                                     '   step start, step:data split',
                                     "   step end, step:data split, output:dict_keys(['X_train.shape', "
                                     "'y_train.shape', 'X_eval.shape', 'y_eval.shape', 'X_test.shape'])",
                                     '   step start, step:search',
                                     "   step end, step:search, output:dict_keys(['best_trial'])",
                                     '   step start, step:load estimator',
                                     "   step end, step:load estimator, output:dict_keys(['estimator'])",
                                     'experiment end']

    def run_binary(self, train_test_split_strategy=None, cv=False, pseudo_labeling=False,
                   feature_reselection=False,
                   collinearity_detection=False, drift_detection=True, max_trials=3, scoring='roc_auc_ovr'):
        rs = RandomSearcher(lambda: search_space_general(early_stopping_rounds=20, verbose=0),
                            optimize_direction=OptimizeDirection.Maximize)
        hk = HyperGBM(rs, reward_metric='auc', cache_dir=f'hypergbm_cache', callbacks=[])
        df = dsutils.load_bank().head(1000)
        df.drop(['id'], axis=1, inplace=True)
        y = df.pop('y')

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=9527)
        log_callback = LogCallback(output_elapsed=True)
        experiment = CompeteExperiment(hk, X_train, y_train, X_test=X_test,
                                       train_test_split_strategy=train_test_split_strategy,
                                       callbacks=[log_callback],
                                       scorer=get_scorer(scoring),
                                       collinearity_detection=collinearity_detection,
                                       drift_detection=drift_detection,
                                       cv=cv,
                                       pseudo_labeling=pseudo_labeling,
                                       feature_reselection=feature_reselection,
                                       feature_reselection_estimator_size=5,
                                       feature_reselection_threshold=1e-5,
                                       ensemble_size=5
                                       )
        pipeline = experiment.run(use_cache=True, max_trials=max_trials)
        auc_scorer = get_scorer('roc_auc_ovo')
        acc_scorer = get_scorer('accuracy')
        auc = auc_scorer(pipeline, X_test, y_test)
        acc = acc_scorer(pipeline, X_test, y_test)
        assert auc
        assert acc

    def test_binary_cv(self):
        self.run_binary(cv=True)

    def test_binary_pseudo_labeling(self):
        self.run_binary(pseudo_labeling=True)

    def test_binary_importance_selection(self):
        self.run_binary(feature_reselection=True, cv=True, scoring='accuracy')

    def test_binary_adversarial_validation(self):
        self.run_binary(train_test_split_strategy='adversarial_validation')
