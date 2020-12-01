# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

from hypergbm import HyperGBM, CompeteExperiment
from hypergbm.search_space import search_space_general
from hypernets.core.searcher import OptimizeDirection
from hypernets.experiment import GeneralExperiment, ExperimentCallback
from hypernets.searchers import RandomSearcher
from tabular_toolbox.datasets import dsutils


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

    def test_exp(self):
        rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)
        hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
                      cache_dir=f'hypergbm_cache',
                      callbacks=[])

        df = dsutils.load_bank().head(1000)
        df.drop(['id'], axis=1, inplace=True)
        y = df.pop('y')

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=9527)
        log_callback = LogCallback()
        experiment = GeneralExperiment(hk, X_train, y_train, X_test, callbacks=[log_callback])
        experiment.run(use_cache=True, max_trails=5)
        assert log_callback.logs == ['experiment start',
                                     '   step start, step:data split',
                                     '   step end, step:data split, output:',
                                     '   step start, step:search',
                                     "   step end, step:search, output:dict_keys(['best_trail'])",
                                     '   step start, step:load estimator',
                                     "   step end, step:load estimator, output:dict_keys(['estimator'])",
                                     'experiment end']

    def test_compete_one_stage(self):
        rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)
        hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
                      cache_dir=f'hypergbm_cache',
                      callbacks=[])

        df = dsutils.load_bank().head(1000)
        df.drop(['id'], axis=1, inplace=True)
        y = df.pop('y')

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=9527)
        log_callback = LogCallback(output_elapsed=True)
        experiment = CompeteExperiment('binary', hk, X_train, y_train, X_test,
                                       callbacks=[log_callback],
                                       scorer=get_scorer('roc_auc_ovr'),
                                       drop_feature_with_collinearity=True,
                                       drift_detection=True,
                                       mode='one-stage',
                                       n_est_feature_importance=5,
                                       importance_threshold=1e-5,
                                       ensemble_size=5
                                       )
        pipeline = experiment.run(use_cache=True, max_trails=10)
        # X_test, _ = experiment.data_cleaner.transform(X_test)
        #
        # scores = experiment.estimator.predict(X_test)
        assert len(log_callback.logs) == len(
            ['experiment start', '   step start, step:clean and split data',
             '      progress:fit_transform train set',
             '         elapsed:0.13164687156677246',
             '      progress:split into train set and eval set',
             '         elapsed:0.13884496688842773',
             '      progress:transform X_test',
             '         elapsed:0.19460487365722656',
             '   step end, step:clean and split data, output:',
             '      elapsed:0.19462800025939941',
             '   step start, step:drop features with multicollinearity',
             '      progress:calc correlation',
             '         elapsed:0.010833263397216797',
             '      progress:drop features',
             '         elapsed:0.02186417579650879',
             "   step end, step:drop features with multicollinearity, output:dict_keys(['corr_linkage', 'selected', 'unselected'])",
             '      elapsed:0.021885156631469727',
             '   step start, step:detect drifting',
             "   step end, step:detect drifting, output:dict_keys(['no_drift_features', 'history'])",
             '      elapsed:3.533931016921997',
             '   step start, step:first stage search',
             "   step end, step:first stage search, output:dict_keys(['best_reward'])",
             '      elapsed:4.484910011291504',
             '   step start, step:ensemble',
             "   step end, step:ensemble, output:dict_keys(['ensemble'])",
             '      elapsed:0.5766010284423828',
             '   step start, step:compose pipeline',
             '   step end, step:compose pipeline, output:',
             '      elapsed:0.00015687942504882812',
             'experiment end',
             '   elapsed:8.81291389465332'])

    def test_compete_two_stage(self):
        rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)
        hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
                      cache_dir=f'hypergbm_cache',
                      callbacks=[])

        df = dsutils.load_bank().head(1000)
        df.drop(['id'], axis=1, inplace=True)
        y = df.pop('y')

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=9527)
        log_callback = LogCallback(output_elapsed=True)
        experiment = CompeteExperiment('binary', hk, X_train, y_train, X_test,
                                       callbacks=[log_callback],
                                       scorer=get_scorer('roc_auc_ovr'),
                                       drop_feature_with_collinearity=True,
                                       drift_detection=True,
                                       mode='two-stage',
                                       n_est_feature_importance=5,
                                       importance_threshold=1e-5,
                                       ensemble_size=5
                                       )
        pipeline = experiment.run(use_cache=True, max_trails=10)
        # X_test, _ = experiment.data_cleaner.transform(X_test)
        #
        # scores = experiment.estimator.predict(X_test)
        assert len(log_callback.logs) == len(['experiment start',
                                              '   step start, step:clean and split data',
                                              '      progress:fit_transform train set',
                                              '         elapsed:0.07546687126159668',
                                              '      progress:split into train set and eval set',
                                              '         elapsed:0.07970499992370605',
                                              '      progress:transform X_test',
                                              '         elapsed:0.1262950897216797',
                                              '   step end, step:clean and split data, output:',
                                              '      elapsed:0.12631487846374512',
                                              '   step start, step:drop features with multicollinearity',
                                              '      progress:calc correlation',
                                              '         elapsed:0.00865626335144043',
                                              '      progress:drop features',
                                              '         elapsed:0.014214277267456055',
                                              "   step end, step:drop features with multicollinearity, output:dict_keys(['corr_linkage', 'selected', 'unselected'])",
                                              '      elapsed:0.014228105545043945',
                                              '   step start, step:detect drifting',
                                              "   step end, step:detect drifting, output:dict_keys(['no_drift_features', 'history'])",
                                              '      elapsed:1.293889045715332',
                                              '   step start, step:first stage search',
                                              "   step end, step:first stage search, output:dict_keys(['best_reward'])",
                                              '      elapsed:5.078425168991089',
                                              '   step start, step:evaluate feature importance',
                                              '      progress:load estimators',
                                              '         elapsed:0.01919388771057129',
                                              '      progress:calc importance',
                                              '         elapsed:40.845674991607666',
                                              '      progress:drop features',
                                              '         elapsed:40.85099387168884',
                                              "   step end, step:evaluate feature importance, output:dict_keys(['importances', 'selected_features', 'unselected_features'])",
                                              '      elapsed:40.851008892059326',
                                              '   step start, step:two stage search',
                                              "   step end, step:two stage search, output:dict_keys(['best_reward'])",
                                              '      elapsed:3.4388532638549805',
                                              '   step start, step:ensemble',
                                              "   step end, step:ensemble, output:dict_keys(['ensemble'])",
                                              '      elapsed:0.5035500526428223',
                                              '   step start, step:compose pipeline',
                                              '   step end, step:compose pipeline, output:',
                                              '      elapsed:0.0001652240753173828',
                                              'experiment end',
                                              '   elapsed:51.307193756103516'])
