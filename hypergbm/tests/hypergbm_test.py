# -*- coding:utf-8 -*-
"""

"""

from sklearn.model_selection import train_test_split

from hypergbm import HyperGBM
from hypergbm.search_space import search_space_general, GeneralSearchSpaceGenerator
from hypernets.core.callbacks import SummaryCallback, FileLoggingCallback
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.utils import fs
from hypernets.tabular.datasets import dsutils
from hypergbm.tests import test_output_dir
from hypernets.discriminators import PercentileDiscriminator


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
                      cache_dir=f'{test_output_dir}/hypergbm_cache',
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
                      cache_dir=f'{test_output_dir}/hypergbm_cache',
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
        filepath = test_output_dir + '/hypergbm_model.pkl'
        est.save(filepath)
        assert fs.isfile(filepath) == True
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
        discriminator = PercentileDiscriminator(0, min_trials=3, min_steps=5, stride=1)
        space_fn = GeneralSearchSpaceGenerator(enable_catboost=False, enable_xgb=False)
        _, hk = self.run_search(self.get_data, cv=True, discriminator=discriminator, max_trials=10, space_fn=space_fn)
        broken_trials = [t for t in hk.history.trials if not t.succeeded]
        assert len(broken_trials) > 0

    def test_discriminator(self):
        discriminator = PercentileDiscriminator(0, min_trials=3, min_steps=5, stride=1)
        space_fn = GeneralSearchSpaceGenerator(enable_catboost=False, enable_xgb=False)
        _, hk = self.run_search(self.get_data, cv=True, discriminator=discriminator, max_trials=10, space_fn=space_fn)
        broken_trials = [t for t in hk.history.trials if not t.succeeded]
        assert len(broken_trials) > 0