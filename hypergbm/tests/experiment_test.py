# -*- coding:utf-8 -*-
__author__ = 'yangjian'

from random import random

from hypergbm.estimators import CatBoostClassifierWrapper

"""

"""
import os
from datetime import datetime
import pandas as pd
from sklearn.metrics import get_scorer
import pytest

from hypergbm import HyperGBM, CompeteExperiment
from hypergbm.search_space import search_space_general, GeneralSearchSpaceGenerator
from hypernets.core import OptimizeDirection, EarlyStoppingCallback
from hypernets.experiment import GeneralExperiment, ExperimentCallback, ConsoleCallback, StepNames
from hypernets.searchers import RandomSearcher

from sklearn.model_selection import train_test_split

from hypergbm import make_experiment
from hypergbm.experiment import PipelineSHAPExplainer
from hypernets.tabular.datasets import dsutils

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

    def get_random_path(self):
        import tempfile
        t_fd, t_path = tempfile.mkstemp(prefix='shap_')

        os.close(t_fd)
        return t_path

    def run_plot(self, shap_values):
        # water fall
        shap.plots.waterfall(shap_values[0], show=False)
        plt.savefig(self.get_random_path())

        # beeswarm
        shap.plots.beeswarm(shap_values, show=False)
        plt.savefig(self.get_random_path(), show=False)

        # force
        shap.plots.force(shap_values[0], show=False)
        plt.savefig(self.get_random_path())

        # plot interaction
        shap.plots.scatter(shap_values[:, "duration"], color=shap_values, show=False)
        plt.savefig(self.get_random_path())

    def run_cv_plot_shap_value(self, model_indexes, enable_kernel):
        df = dsutils.load_bank()
        df_train, df_test = train_test_split(df, test_size=0.8, random_state=42)

        experiment = make_experiment(df_train, target='y',
                                     max_trials=3,
                                     log_level='info', cv=True, num_folds=3)

        estimator = experiment.run()

        # test tree explainer
        tree_explainer = PipelineSHAPExplainer(estimator, model_indexes=model_indexes)
        tree_values_list = tree_explainer(df_test)
        assert len(tree_values_list) == len(model_indexes)

        self.run_plot(tree_values_list[0][0])

        if enable_kernel:
            # test kernel explainer
            kernel_explainer = PipelineSHAPExplainer(estimator, data=df_test, method='kernel')
            kernel_values_list = kernel_explainer(df_test.sample(n=10))
            self.run_plot(kernel_values_list[0])

    def test_cv_plot_shap_value(self):
        self.run_cv_plot_shap_value(model_indexes=[0], enable_kernel=True)

    def test_select_cv_models(self):
        self.run_cv_plot_shap_value(model_indexes=[0, 1], enable_kernel=False)

    def test_final_train_plot_shap_value(self):
        df = dsutils.load_bank()
        df_train, df_test = train_test_split(df, test_size=0.8, random_state=42)

        experiment = make_experiment(df_train, target='y',
                                     max_trials=3,
                                     log_level='info', ensemble=None, cv=True, num_folds=3)

        estimator = experiment.run()

        # test tree explainer
        tree_explainer = PipelineSHAPExplainer(estimator)
        tree_values_list = tree_explainer(df_test)
        self.run_plot(tree_values_list[0][0])

        # test kernel explainer
        kernel_explainer = PipelineSHAPExplainer(estimator, data=df_test, method='kernel')
        kernel_values_list = kernel_explainer(df_test.sample(n=5))
        self.run_plot(kernel_values_list[0])
