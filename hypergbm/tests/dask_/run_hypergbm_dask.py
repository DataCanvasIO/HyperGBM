# -*- coding:utf-8 -*-
"""

"""
from dask.distributed import Client
from dask_ml import preprocessing as dm_pre
from dask_ml.model_selection import train_test_split
from tabular_toolbox.datasets import dsutils

from hypergbm.dask.dask_ops import get_space_num_cat_pipeline_complex
from hypergbm import HyperGBM
from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers.random_searcher import RandomSearcher
from hypergbm.tests import test_output_dir


def main():
    # client = Client("tcp://127.0.0.1:64958")
    client = Client(processes=False, threads_per_worker=2, n_workers=1, memory_limit='4GB')
    print(client)

    rs = RandomSearcher(get_space_num_cat_pipeline_complex, optimize_direction=OptimizeDirection.Maximize)
    hk = HyperGBM(rs, task='classification', reward_metric='accuracy',
                  cache_dir=f'{test_output_dir}/hypergbm_cache',
                  callbacks=[SummaryCallback(), FileLoggingCallback(rs, output_dir=f'{test_output_dir}/hyn_logs')])

    df = dsutils.load_bank_by_dask()
    df.drop(['id'], axis=1)
    df['y'] = dm_pre.LabelEncoder().fit_transform(df['y'])
    # df = df.sample(frac=0.1)

    # object_columns = [i for i, v in df.dtypes.items() if v == 'object']
    # for c in object_columns:
    #     df[c] = df[c].astype('category')
    # df = df.categorize(object_columns)

    X_train, X_test = train_test_split(df, test_size=0.8, random_state=42)
    y_train = X_train.pop('y')
    y_test = X_test.pop('y')

    hk.search(X_train, y_train, X_test, y_test, max_trails=50)
    print('-' * 30)

    best_trial = hk.get_best_trail()
    print(f'best_train:{best_trial}')
    estimator = hk.final_train(best_trial.space_sample, X_train, y_train)
    score = estimator.predict(X_test)
    result = estimator.evaluate(X_test, y_test, metrics=['accuracy', 'auc', 'logloss'])
    print(f'final result:{result}')


if __name__ == '__main__':
    main()
