# -*- coding:utf-8 -*-
"""

"""

from sklearn.model_selection import train_test_split

from hypergbm import HyperGBM
from hypergbm.search_space import search_space_general
from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers.random_searcher import RandomSearcher
from tabular_toolbox.datasets import dsutils
from hypergbm.tests import test_output_dir


def main():
    rs = RandomSearcher(search_space_general, optimize_direction=OptimizeDirection.Maximize)
    hk = HyperGBM(rs, task='classification', reward_metric='auc',
                  cache_dir=f'{test_output_dir}/hypergbm_cache',
                  callbacks=[SummaryCallback(), FileLoggingCallback(rs, output_dir=f'{test_output_dir}/hyn_logs')])

    df = dsutils.load_bank()
    df.drop(['id'], axis=1, inplace=True)
    X_train, X_test = train_test_split(df, test_size=0.8, random_state=42)
    y_train = X_train.pop('y')
    y_test = X_test.pop('y')

    hk.search(X_train, y_train, X_test, y_test, max_trails=500)
    best_trial = hk.get_best_trail()
    print(f'best_train:{best_trial}')
    estimator = hk.final_train(best_trial.space_sample, X_train, y_train)
    score = estimator.predict(X_test)
    result = estimator.evaluate(X_test, y_test, metrics=['auc', 'accuracy'])
    print(f'final result:{result}')


if __name__ == '__main__':
    main()
