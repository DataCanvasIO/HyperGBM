# -*- coding:utf-8 -*-
"""

"""
import random
import os

import pytest

from sklearn.model_selection import train_test_split

from hypergbm import HyperGBM
from hypergbm.estimators import LGBMClassifierWrapper
from hypergbm.hyper_gbm import HyperGBMShapExplainer
from hypergbm.objectives import FeatureUsageObjective
from hypergbm.search_space import search_space_general, GeneralSearchSpaceGenerator
from hypergbm.tests import test_output_dir
from hypernets.core import set_random_state
from hypernets.core.callbacks import SummaryCallback, FileLoggingCallback
from hypernets.core.searcher import OptimizeDirection
from hypernets.discriminators import PercentileDiscriminator
from hypernets.model.objectives import PredictionObjective
from hypernets.searchers import NSGAIISearcher
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.tabular.datasets import dsutils
from hypernets.utils import fs

try:
    import shap
    import matplotlib.pyplot as plt
    is_shap_installed = True
except:
    is_shap_installed = False

need_shap = pytest.mark.skipif(not  is_shap_installed, reason="shap is not installed ")


class Test_HyperGBM():

    def get_data(self):
        df = dsutils.load_bank()
        df.drop(['id'], axis=1, inplace=True)
        X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
        y_train = X_train.pop('y')
        y_test = X_test.pop('y')
        return X_train, X_test, y_train, y_test

    def run_search(self, data_partition, cv=False, num_folds=3, discriminator=None, max_trials=3, space_fn=None):
        rs = RandomSearcher(space_fn if space_fn else search_space_general,
                            optimize_direction=OptimizeDirection.Maximize)
        hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
                      callbacks=[SummaryCallback(), FileLoggingCallback(rs, output_dir=f'{test_output_dir}/hyn_logs')],
                      discriminator=discriminator)

        X_train, X_test, y_train, y_test = data_partition()

        hk.search(X_train, y_train, X_test, y_test, cv=cv, num_folds=num_folds, max_trials=max_trials)
        best_trial = hk.get_best_trial()

        estimator = hk.final_train(best_trial.space_sample, X_train, y_train)
        score = estimator.predict(X_test)
        result = estimator.evaluate(X_test, y_test)
        assert len(score) == 200
        return estimator, hk

    def test_cross_validator(self):
        from hypernets.tabular.lifelong_learning import PrequentialSplit
        df = dsutils.load_bank()
        df.drop(['id'], axis=1, inplace=True)
        rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)
        hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
                      callbacks=[SummaryCallback(), FileLoggingCallback(rs, output_dir=f'{test_output_dir}/hyn_logs')])

        X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
        y_train = X_train.pop('y')
        y_test = X_test.pop('y')

        preq_split = PrequentialSplit(PrequentialSplit.STRATEGY_PREQ_BLS, n_splits=3)
        hk.search(X_train, y_train, X_test, y_test, cv=True, max_trials=3, cross_validator=preq_split)
        best_trial = hk.get_best_trial()

        estimator = hk.final_train(best_trial.space_sample, X_train, y_train)
        score = estimator.predict(X_test)
        result = estimator.evaluate(X_test, y_test)
        assert len(score) == 200
        return estimator, hk

    def test_save_load(self):
        df = dsutils.load_bank()
        df.drop(['id'], axis=1, inplace=True)
        X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
        y_train = X_train.pop('y')
        y_test = X_test.pop('y')

        def f():
            return X_train, X_test, y_train, y_test

        est, hypergbm = self.run_search(f)
        fs.mkdirs(test_output_dir, exist_ok=True)
        filepath = test_output_dir + '/hypergbm_model.pkl'
        est.save(filepath)
        assert fs.isfile(filepath)
        model = hypergbm.load_estimator(filepath)
        score = model.evaluate(X_test, y_test, ['AUC'])
        assert score

    def test_model(self):
        self.run_search(self.get_data)

    def test_cv(self):
        self.run_search(self.get_data, cv=True)

    def test_no_categorical(self):
        df = dsutils.load_bank()

        df.drop(['id'], axis=1, inplace=True)
        df = df[['age', 'duration', 'previous', 'y']]

        X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
        y_train = X_train.pop('y')
        y_test = X_test.pop('y')

        def f():
            return X_train, X_test, y_train, y_test

        self.run_search(f)

    def test_no_continuous(self):
        df = dsutils.load_bank()

        df.drop(['id'], axis=1, inplace=True)
        df = df[['job', 'education', 'loan', 'y']]

        X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
        y_train = X_train.pop('y')
        y_test = X_test.pop('y')

        def f():
            return X_train, X_test, y_train, y_test

        self.run_search(f)

    def test_discriminator_cv(self):
        discriminator = PercentileDiscriminator(100, min_trials=3, min_steps=5, stride=1)
        space_fn = GeneralSearchSpaceGenerator(enable_catboost=False, enable_xgb=False)
        _, hk = self.run_search(self.get_data, cv=True, discriminator=discriminator, max_trials=10, space_fn=space_fn)
        broken_trials = [t for t in hk.history.trials if not t.succeeded]
        assert len(broken_trials) > 0

    def test_discriminator(self):
        discriminator = PercentileDiscriminator(100, min_trials=3, min_steps=5, stride=1)
        space_fn = GeneralSearchSpaceGenerator(enable_catboost=False, enable_xgb=False)
        _, hk = self.run_search(self.get_data, cv=True, discriminator=discriminator, max_trials=10, space_fn=space_fn)
        broken_trials = [t for t in hk.history.trials if not t.succeeded]
        assert len(broken_trials) > 0

    def test_discriminator_catboost(self):
        discriminator = PercentileDiscriminator(100, min_trials=3, min_steps=5, stride=1)
        space_fn = GeneralSearchSpaceGenerator(enable_catboost=True, enable_lightgbm=False, enable_xgb=False)
        _, hk = self.run_search(self.get_data, cv=True, discriminator=discriminator, max_trials=10, space_fn=space_fn)
        broken_trials = [t for t in hk.history.trials if not t.succeeded]
        assert len(broken_trials) > 0

    def test_set_random_state(self):
        set_random_state(9527)
        _, hk = self.run_search(self.get_data, cv=False, max_trials=5)
        vectors = [t.space_sample.vectors for t in hk.history.trials]
        # assert vectors == [[0, 0, 1, 0, 30, 0, 0, 0, 4, 0], [1, 3, 1, 0, 3, 1, 2, 1, 0, 0, 4],
        #                    [0, 0, 1, 1, 180, 2, 3, 5, 1, 0], [1, 1, 0, 1, 0, 0, 4, 1, 4, 1], [2, 3, 1, 3, 2, 4, 1]]

        # reset random seed
        set_random_state(9527)
        _, hk2 = self.run_search(self.get_data, cv=False, max_trials=5)
        vectors2 = [t.space_sample.vectors for t in hk2.history.trials]

        assert vectors == vectors2


@need_shap
class TestShapExplainer:

    @classmethod
    def setup_class(cls):
        kwargs_list = [
            dict(enable_lightgbm=True, enable_xgb=False, enable_catboost=False, enable_histgb=False),
            dict(enable_lightgbm=False, enable_xgb=True, enable_catboost=False, enable_histgb=False),
            dict(enable_lightgbm=False, enable_xgb=False, enable_catboost=True, enable_histgb=False)
        ]
        cls.search_spaces = [GeneralSearchSpaceGenerator(**_) for _ in kwargs_list]

    def _train(self, df, target, search_space, task, reward_metric, optimize_direction,  is_cv):
        rs = RandomSearcher(search_space, optimize_direction=optimize_direction)
        # rs = RandomSearcher(search_space, optimize_direction=OptimizeDirection.Maximize)
        # binary
        hk = HyperGBM(rs, task=task,
                      reward_metric=reward_metric,
                      callbacks=[SummaryCallback(), FileLoggingCallback(rs, output_dir=f'{test_output_dir}/hyn_logs')])
        X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
        y_train = X_train.pop(target)
        y_test = X_test.pop(target)
        if is_cv:
            hk.search(X_train, y_train, X_test, y_test, cv=True, num_folds=3, max_trials=2)
        else:
            hk.search(X_train, y_train, X_test, y_test, cv=False, max_trials=2)

        best_trial = hk.get_best_trial()

        best_estimator = best_trial.get_model()

        explainer = HyperGBMShapExplainer(best_estimator)
        return explainer

    @staticmethod
    def get_random_path():
        import tempfile
        t_fd, t_path = tempfile.mkstemp(prefix=TestShapExplainer.__class__.__name__)
        os.close(t_fd)
        return t_path

    @staticmethod
    def run_plot(shap_values, interaction):

        # water fall
        shap.plots.waterfall(shap_values[0], show=False)
        plt.savefig(TestShapExplainer.get_random_path())

        # beeswarm
        shap.plots.beeswarm(shap_values, show=False)
        plt.savefig(TestShapExplainer.get_random_path(), show=False)

        # force
        shap.plots.force(shap_values[0], show=False)
        plt.savefig(TestShapExplainer.get_random_path())

        # plot interaction
        shap.plots.scatter(shap_values[:, interaction], color=shap_values, show=False)
        plt.savefig(TestShapExplainer.get_random_path())

    def _explain_gbm_alg(self, df, target, task, reward_metric, optimize_direction, is_cv):
        data_list = []
        # OptimizeDirection.Maximize
        for search_space in self.search_spaces:
            explainer = self._train(df=df.copy(), target=target,
                                    search_space=search_space,
                                    task=task,
                                    optimize_direction=optimize_direction,
                                    reward_metric=reward_metric,
                                    is_cv=is_cv)
            df_test = df.sample(n=100, random_state=1234)
            shap_values_list = explainer(df_test)
            data_list.append((shap_values_list, explainer))
        return data_list

    def run_binary(self, func_validation, is_cv):
        df = dsutils.load_bank()
        df.drop(['id'], axis=1, inplace=True)
        data_list = self._explain_gbm_alg(df=df, target='y', task='binary',
                                          reward_metric='accuracy',
                                          optimize_direction=OptimizeDirection.Maximize,
                                          is_cv=is_cv)

        for shap_values_list, explainer in data_list:
            func_validation(shap_values_list, explainer, "duration")

    def run_regression(self, func_validation, is_cv):
        df = dsutils.load_boston()
        data_list = self._explain_gbm_alg(df=df.copy(), target='target', task='regression',
                                          reward_metric='rmse',
                                          optimize_direction=OptimizeDirection.Minimize, is_cv=is_cv)

        for shap_values_list, explainer in data_list:
            func_validation(shap_values_list, explainer, interaction="CRIM")

    def validate_train_test_split_model(self, shap_values_list, explainer, interaction):
        if isinstance(explainer.hypergbm_estimator.model, LGBMClassifierWrapper):
            # LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray
            # LightGBM output shape: (n_rows, n_cols, n_classes)
            # other gbm alg output shape: (n_rows, n_cols)
            assert len(shap_values_list.shape) == 3
            shap_values = shap_values_list[:, :, 1]  # shap values of positive label
        else:
            shap_values = shap_values_list
        assert len(shap_values.shape) == 2
        self.run_plot(shap_values, interaction)

    def validate_cv_models(self, shap_values_list, explainer, interaction):
        assert len(shap_values_list) == 3
        assert isinstance(shap_values_list, list)
        if isinstance(explainer.hypergbm_estimator.model, LGBMClassifierWrapper):
            # LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray
            # LightGBM output shape: (n_rows, n_cols, n_classes)
            # other gbm alg output shape: (n_rows, n_cols)
            assert len(shap_values_list[0].shape) == 3
            shap_values = shap_values_list[0][:, :, 1]  # shap values of positive label
        else:
            shap_values = shap_values_list[0]
            assert len(shap_values.shape) == 2
        self.run_plot(shap_values, interaction)

    def test_binary_cv_models(self):
        self.run_binary(self.validate_cv_models, True)

    def test_binary_train_test_split_model(self):
        self.run_binary(self.validate_train_test_split_model, is_cv=False)

    def test_regression_cv_models(self):
        self.run_regression(self.validate_cv_models, is_cv=True)

    def test_regression_train_test_split_model(self):
        self.run_regression(self.validate_train_test_split_model, is_cv=False)


class TestObjectivesInHyperGBM:

    def test_feature_usage(self):
        from hypernets.tabular.datasets import dsutils
        df = dsutils.load_bank().head(1000)

        rs = NSGAIISearcher(search_space_general, objectives=[PredictionObjective.create('accuracy'),
                                                              FeatureUsageObjective()])

        hk = HyperGBM(rs, task='binary',
                      reward_metric='accuracy',
                      callbacks=[SummaryCallback(), FileLoggingCallback(rs, output_dir=f'{test_output_dir}/hyn_logs')])
        X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
        target = 'y'
        y_train = X_train.pop(target)
        y_test = X_test.pop(target)
        is_cv = True

        if is_cv:
            hk.search(X_train, y_train, X_test, y_test, cv=True, num_folds=3, max_trials=2)
        else:
            hk.search(X_train, y_train, X_test, y_test, cv=False, max_trials=2)

        best_trials = hk.get_best_trial()
        assert best_trials
        best_estimator = best_trials[0].get_model()
        assert best_estimator.predict(X_test.copy()) is not None

