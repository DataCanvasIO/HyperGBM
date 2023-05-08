# -*- coding:utf-8 -*-
__author__ = 'yangjian'

from random import random

from hypergbm.estimators import CatBoostClassifierWrapper

"""

"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

from hypergbm import HyperGBM, CompeteExperiment
from hypergbm.search_space import search_space_general, GeneralSearchSpaceGenerator
from hypernets.core import OptimizeDirection, EarlyStoppingCallback
from hypernets.experiment import GeneralExperiment, ExperimentCallback, ConsoleCallback, StepNames
from hypernets.searchers import RandomSearcher
from hypernets.tabular.feature_generators import is_feature_generator_ready

from hypergbm import make_experiment
from hypergbm.experiment import PipelineKernelExplainer, PipelineTreeExplainer
from hypernets.tabular.datasets import dsutils

from hypergbm.tests.hypergbm_test import TestShapExplainer

try:
    import shap
    import matplotlib.pyplot as plt
    is_shap_installed = True
except:
    is_shap_installed = False

need_shap = pytest.mark.skipif(not is_shap_installed, reason="shap is not installed")


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


class Test_Experiment():
    def test_regression_cv(self):
        self.run_regression(cv=True)

    def test_regression_feature_reselection(self):
        self.run_regression(feature_reselection=True)

    def test_regression_pseudo_labeling(self):
        self.run_regression(pseudo_labeling=True)

    def test_regression_adversarial_validation(self):
        self.run_regression(train_test_split_strategy='adversarial_validation')

    def test_regression_cross_validator(self):
        from hypernets.tabular.lifelong_learning import PrequentialSplit
        preq_split = PrequentialSplit(PrequentialSplit.STRATEGY_PREQ_BLS, n_splits=3)
        self.run_regression(cv=True, cross_validator=preq_split)

    def run_regression(self, train_test_split_strategy=None, cv=False, feature_reselection=False, pseudo_labeling=False,
                       collinearity_detection=False, drift_detection=True, max_trials=3, cross_validator=None):
        df = dsutils.load_Bike_Sharing()
        y = df.pop('count')

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=9527)

        log_callback = LogCallback(output_elapsed=True)
        rs = RandomSearcher(lambda: search_space_general(early_stopping_rounds=5, ),
                            optimize_direction='min')
        hk = HyperGBM(rs, task='regression', reward_metric='mse', callbacks=[])
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
                                       ensemble_size=10,
                                       cross_validator=cross_validator,
                                       random_state=12345,
                                       )
        pipeline = experiment.run(max_trials=max_trials)
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

    def test_multiclass_cross_validator(self):
        from hypernets.tabular.lifelong_learning import PrequentialSplit
        preq_split = PrequentialSplit(PrequentialSplit.STRATEGY_PREQ_BLS, n_splits=3)
        self.run_multiclass(cv=True, cross_validator=preq_split)

    def run_multiclass(self, train_test_split_strategy=None, cv=False, feature_reselection=False, pseudo_labeling=False,
                       collinearity_detection=False, drift_detection=True, max_trials=3, cross_validator=None):
        df = dsutils.load_glass_uci()
        df = pd.concat([df] * 10).sample(frac=1.0).reset_index(drop=True)
        df.columns = [f'x_{c}' for c in df.columns.to_list()]
        df.pop('x_0')
        y = df.pop('x_10')
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=1, stratify=y)

        rs = RandomSearcher(lambda: search_space_general(early_stopping_rounds=20, verbose=0),
                            optimize_direction=OptimizeDirection.Maximize)
        es = EarlyStoppingCallback(20, 'max')
        hk = HyperGBM(rs, reward_metric='auc', callbacks=[es])

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
                                       ensemble_size=10,
                                       cross_validator=cross_validator,
                                       random_state=2345
                                       )
        pipeline = experiment.run(max_trials=max_trials)
        acc_scorer = get_scorer('accuracy')
        acc = acc_scorer(pipeline, X_test, y_test)
        assert acc
        auc_scorer = get_scorer('roc_auc_ovo')
        auc = auc_scorer(pipeline, X_test, y_test)
        assert auc

    def test_general_exp(self):
        rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)
        hk = HyperGBM(rs, reward_metric='accuracy', callbacks=[])

        df = dsutils.load_bank().head(1000)
        df.drop(['id'], axis=1, inplace=True)
        y = df.pop('y')

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=9527)
        log_callback = LogCallback()
        experiment = GeneralExperiment(hk, X_train, y_train, X_test=X_test, callbacks=[log_callback])
        experiment.run(max_trials=5)
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
                   collinearity_detection=False, drift_detection=True, max_trials=3, scoring='roc_auc_ovr',
                   cross_validator=None):
        rs = RandomSearcher(lambda: search_space_general(early_stopping_rounds=20, verbose=0),
                            optimize_direction=OptimizeDirection.Maximize)
        hk = HyperGBM(rs, reward_metric='auc', callbacks=[])
        df = dsutils.load_bank().head(1000)
        df.drop(['id'], axis=1, inplace=True)
        y = df.pop('y')

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=9527)
        log_callback = LogCallback(output_elapsed=True)
        experiment = CompeteExperiment(hk, X_train, y_train, X_test=X_test,
                                       train_test_split_strategy=train_test_split_strategy,
                                       callbacks=[log_callback],
                                       scorer=get_scorer(scoring),
                                       data_adaption=False,
                                       collinearity_detection=collinearity_detection,
                                       drift_detection=drift_detection,
                                       cv=cv,
                                       pseudo_labeling=pseudo_labeling,
                                       feature_reselection=feature_reselection,
                                       feature_reselection_estimator_size=5,
                                       feature_reselection_threshold=1e-5,
                                       ensemble_size=5,
                                       cross_validator=cross_validator,
                                       random_state=12345,
                                       )
        pipeline = experiment.run(max_trials=max_trials)
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

    def test_binary_cross_validator(self):
        from hypernets.tabular.lifelong_learning import PrequentialSplit
        preq_split = PrequentialSplit(PrequentialSplit.STRATEGY_PREQ_BLS, n_splits=3)
        self.run_binary(cv=True, cross_validator=preq_split)

    @pytest.mark.skipif(not is_feature_generator_ready, reason='feature_generator is not ready')
    def test_feature_generation(self):
        from hypernets.tabular.cfg import TabularCfg as tcfg
        tcfg.tfidf_primitive_output_feature_count = 5

        df = dsutils.load_movielens()
        df['genres'] = df['genres'].apply(lambda s: s.replace('|', ' '))
        df['timestamp'] = df['timestamp'].apply(datetime.fromtimestamp)

        experiment = make_experiment(df, target='rating', cv=False, ensemble_size=0,
                                     feature_generation=True,
                                     feature_generation_text_cols=['title', 'genres'],
                                     random_state=2345
                                     )
        assert isinstance(experiment, CompeteExperiment)

        estimator = experiment.run(max_trials=3)
        assert estimator is not None

        step = experiment.get_step(StepNames.FEATURE_GENERATION)
        assert step is not None

        feature_names = step.get_fitted_params()['output_feature_names']
        assert all([c in feature_names for c in ['TFIDF__title____0__', 'TFIDF__genres____0__', 'DAY__timestamp__']])

    def run_cat_boost_multiclass(self, metric, cat_metric):
        # iris
        df = dsutils.load_bank()
        df.drop(['id'], axis=1, inplace=True)
        from random import Random

        df['y'] = df['y'].apply(lambda v: str(Random().randint(0, 2)))
        df = df.sample(1000)

        space = GeneralSearchSpaceGenerator(enable_lightgbm=False, enable_xgb=False,
                                            enable_catboost=True, enable_histgb=False)

        experiment = make_experiment(df, target='y', cv=False, ensemble_size=0, search_space=space,
                                     feature_generation=False,
                                     random_state=2345,
                                     reward_metric=metric)

        assert isinstance(experiment, CompeteExperiment)

        estimator = experiment.run(max_trials=3)
        assert estimator is not None
        assert isinstance(estimator.steps[-1][1].gbm_model, CatBoostClassifierWrapper)
        if cat_metric is not None:
            assert estimator.steps[-1][1].gbm_model._init_params['eval_metric'] == cat_metric
        else:
            assert estimator.steps[-1][1].gbm_model._init_params.get('eval_metric') is None

    def test_cat_boost_multiclass(self):
        self.run_cat_boost_multiclass('f1', 'TotalF1')
        self.run_cat_boost_multiclass('precision', None)
        self.run_cat_boost_multiclass('recall', None)


@need_shap
class TestPipelineExplainer:

    def run_kernel_explainer(self, estimator, df_test, interaction_feature, task):
        kernel_explainer = PipelineKernelExplainer(estimator, data=df_test.sample(10))
        X_exp = df_test.head(n=2)
        kernel_shap_values = kernel_explainer(X_exp)
        if task == 'regression':
            assert kernel_shap_values.shape == X_exp.shape
            TestShapExplainer.run_plot(kernel_shap_values, interaction=interaction_feature)
        elif task == 'binary':
            # for binary task, the shape is (n_classes, n_rows, n_cols)
            assert len(kernel_shap_values) == 2  # 2 outputs
            assert kernel_shap_values[0].shape == X_exp.shape
            TestShapExplainer.run_plot(kernel_shap_values[1], interaction=interaction_feature)
        else:
            assert kernel_shap_values is not None

    def run_tree_explainer(self, estimator, df_test, model_indexes):
        tree_explainer = PipelineTreeExplainer(estimator, model_indexes=model_indexes)
        return tree_explainer(df_test)

    def get_regression_model(self, cv, estimator_type:str, enable_ensemble: bool):
        """
        estimator_type: is one of lightgbm, xgb, catboost
        """

        if cv is True:
            num_folds = 3
        else:
            num_folds = None

        if enable_ensemble:
            ensemble_size = 20
        else:
            ensemble_size = 0

        df = dsutils.load_boston()
        df_train, df_test = train_test_split(df, test_size=0.8, random_state=42)

        search_space_options = dict(enable_lightgbm=False, enable_xgb=False, enable_catboost=False, enable_histgb=False)
        search_space_options[f'enable_{estimator_type}'] = True

        from hypergbm.search_space import GeneralSearchSpaceGenerator
        search_space = GeneralSearchSpaceGenerator(**search_space_options)

        experiment = make_experiment(df_train, target='target',
                                     max_trials=3,
                                     random_state=1234,
                                     search_space=search_space,
                                     log_level='info', cv=cv, ensemble_size=ensemble_size, num_folds=num_folds)

        estimator = experiment.run()
        return estimator, df_test

    def get_binary_model(self, cv, estimator_type, enable_ensemble):
        if cv is True:
            num_folds = 3
        else:
            num_folds = None

        if enable_ensemble:
            ensemble_size = 20
        else:
            ensemble_size = 0

        search_space_options = dict(enable_lightgbm=False, enable_xgb=False, enable_catboost=False, enable_histgb=False)
        search_space_options[f'enable_{estimator_type}'] = True

        from hypergbm.search_space import GeneralSearchSpaceGenerator
        search_space = GeneralSearchSpaceGenerator(**search_space_options)

        df = dsutils.load_bank().sample(1000, random_state=1234)
        df_train, df_test = train_test_split(df, test_size=0.8, random_state=42)
        experiment = make_experiment(df_train.copy(), target='y',
                                     max_trials=3,
                                     search_space=search_space,
                                     reward_metric='accuracy',
                                     optimize_direction='max',
                                     random_state=1234,
                                     log_level='info',
                                     cv=cv,
                                     ensemble_size=ensemble_size,
                                     num_folds=num_folds)

        estimator = experiment.run()
        return estimator, df_test.copy()

    @staticmethod
    def get_max_weight_index(estimator):
        max_weight_index = np.argmax(estimator.steps[-1][1].weights_)
        return max_weight_index

    @pytest.mark.parametrize('estimator_type', ['lightgbm', 'xgb', 'catboost'])
    @pytest.mark.parametrize('enable_ensemble', [True, False])
    @pytest.mark.parametrize('enable_cv', [True, False])
    def test_regression_plot(self, estimator_type: str, enable_ensemble: bool, enable_cv: bool):

        estimator, df_test = self.get_regression_model(cv=enable_cv, enable_ensemble=enable_ensemble,
                                                       estimator_type=estimator_type)

        self.run_kernel_explainer(estimator, df_test, interaction_feature='CRIM', task='regression')

        if enable_ensemble:
            max_weight_index = self.get_max_weight_index(estimator)  # index of 0 maybe None in ensemble
        else:
            max_weight_index = None

        if enable_cv:
            if enable_ensemble:
                tree_values_list = self.run_tree_explainer(estimator, df_test, model_indexes=[max_weight_index])
                assert len(tree_values_list) == 1
                assert len(tree_values_list[0]) == 3  # cv models
                # On regression task it's the same shape of lightgbm or other estimators
                TestShapExplainer.run_plot(tree_values_list[0][0], interaction="CRIM")
            else:
                tree_values_list = self.run_tree_explainer(estimator, df_test, model_indexes=None)
                assert len(tree_values_list) == 3  # cv models
                assert len(tree_values_list[0].shape) == 2
                TestShapExplainer.run_plot(tree_values_list[0], interaction="CRIM")
        else:
            if enable_ensemble:
                tree_values_list = self.run_tree_explainer(estimator, df_test, model_indexes=[max_weight_index])
                assert len(tree_values_list) == 1
                # tree_values_list[0].shape=(3617,16,2)
                TestShapExplainer.run_plot(tree_values_list[0], interaction='CRIM')
            else:
                tree_values_list = self.run_tree_explainer(estimator, df_test, model_indexes=None)
                assert len(tree_values_list.shape) == 2
                # tree_values_list[0].shape=(3617,1)
                TestShapExplainer.run_plot(tree_values_list, interaction='CRIM')

    @pytest.mark.parametrize('estimator_type', ['lightgbm', 'xgb', 'catboost'])
    @pytest.mark.parametrize('enable_ensemble', [True, False])
    @pytest.mark.parametrize('enable_cv', [True, False])
    def test_binary_plot(self, estimator_type: str, enable_ensemble: bool, enable_cv: bool):

        estimator, df_test = self.get_binary_model(cv=enable_cv,
                                                   estimator_type=estimator_type, enable_ensemble=enable_ensemble)

        self.run_kernel_explainer(estimator, df_test, interaction_feature="duration", task='binary')

        def assert_shap_value(shape_values):
            if estimator_type == 'lightgbm':
                assert len(shape_values.shape) == 3
                TestShapExplainer.run_plot(shape_values[:, :, 1],
                                           interaction='duration')  # tree_values_list[0].shape=(3617,16,2)
            else:
                assert len(shape_values.shape) == 2
                # tree_values_list[0].shape=(3617,16)
                TestShapExplainer.run_plot(shape_values, interaction='duration')

        if enable_ensemble:
            max_weight_index = self.get_max_weight_index(estimator)  # index of 0 maybe None in ensemble
        else:
            max_weight_index = None

        if enable_cv:
            if enable_ensemble:

                tree_values_list = self.run_tree_explainer(estimator, df_test, model_indexes=[max_weight_index])
                assert len(tree_values_list) == 1
                assert len(tree_values_list[0]) == 3  # cv models
                assert_shap_value(tree_values_list[0][0])
            else:
                tree_values_list = self.run_tree_explainer(estimator, df_test, model_indexes=None)
                assert len(tree_values_list) == 3  # cv models
                assert_shap_value(tree_values_list[0])
        else:
            if enable_ensemble:
                tree_values_list = self.run_tree_explainer(estimator, df_test, model_indexes=[max_weight_index])
                assert len(tree_values_list) == 1
                assert_shap_value(tree_values_list[0])
            else:
                tree_values_list = self.run_tree_explainer(estimator, df_test, model_indexes=None)
                # tree_values_list[0].shape=(3617,1)
                assert_shap_value(tree_values_list)


class TestObjectives:

    def test_feature_usage(self):

        from hypergbm import make_experiment

        from hypernets.tabular import get_tool_box
        from hypernets.tabular.datasets import dsutils
        from hypernets.core.random_state import get_random_state

        # hyn_logging.set_level(hyn_logging.WARN)
        random_state = get_random_state()

        df = dsutils.load_bank().head(1000)
        tb = get_tool_box(df)
        df_train, df_test = tb.train_test_split(df, test_size=0.2, random_state=9527)
        experiment = make_experiment(df_train,
                                     eval_data=df_test.copy(),
                                     callbacks=[],
                                     cv=False,
                                     num_folds=3,
                                     random_state=1234,
                                     search_callbacks=[],
                                     target='y',
                                     searcher='nsga2',  # available MOO searcher: moead, nsga2, rnsga2
                                     searcher_options={'population_size': 3},
                                     reward_metric='logloss',
                                     objectives=['feature_usage'],
                                     early_stopping_rounds=10)

        estimators = experiment.run(max_trials=5)
        hyper_model = experiment.hyper_model_

        assert hyper_model.history.get_best() is not None

