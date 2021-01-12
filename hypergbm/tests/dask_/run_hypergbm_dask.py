# -*- coding:utf-8 -*-
"""

"""
from dask.distributed import Client
from dask_ml import preprocessing as dm_pre
from dask_ml.model_selection import train_test_split

from hypergbm import HyperGBM
from hypergbm.dask.search_space import search_space_general
from hypergbm.tests import test_output_dir
from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers.random_searcher import RandomSearcher
from tabular_toolbox.datasets import dsutils


def main():
    # client = Client("tcp://127.0.0.1:64958")
    client = Client(processes=False, threads_per_worker=2, n_workers=1, memory_limit='4GB')
    # client = Client(processes=True, threads_per_worker=4, n_workers=30, memory_limit='20GB')
    print(client)

    def search_space():
        return search_space_general(lightgbm_init_kwargs={'n_jobs': 4},
                                    xgb_init_kwargs={'n_jobs': 4})

    rs = RandomSearcher(search_space, optimize_direction=OptimizeDirection.Maximize)
    hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
                  cache_dir=f'hypergbm_cache',
                  callbacks=[SummaryCallback(),
                             FileStorageLoggingCallback(rs, output_dir=f'{test_output_dir}/hyn_logs')])

    target_name = 'y'
    df = dsutils.load_bank_by_dask()
    df.drop(['id'], axis=1)
    df[target_name] = dm_pre.LabelEncoder().fit_transform(df['y'])
    # df = df.sample(frac=0.1)

    worker_count = len(client.ncores())
    df = df.repartition(npartitions=worker_count)

    # df.repartition(npartitions=6)
    # object_columns = [i for i, v in df.dtypes.items() if v == 'object']
    # for c in object_columns:
    #     df[c] = df[c].astype('category')
    # df = df.categorize(object_columns)

    X_train, X_test = train_test_split(df, test_size=0.5, random_state=42, shuffle=False)
    y_train = X_train.pop(target_name)
    y_test = X_test.pop(target_name)

    # X_train, X_test, y_train, y_test =  X_train.persist(), X_test.persist(), y_train.persist(), y_test.persist()
    X_train, X_test, y_train, y_test = client.persist([X_train, X_test, y_train, y_test])

    hk.search(X_train, y_train, X_test, y_test, max_trials=200, use_cache=False, verbose=1)
    print('-' * 30)

    best_trial = hk.get_best_trial()
    print(f'best_train:{best_trial}')

    estimator = hk.final_train(best_trial.space_sample, X_train, y_train)
    # score = estimator.predict(X_test)

    result = estimator.evaluate(X_test, y_test, metrics=['accuracy', 'auc', 'logloss'])
    print(f'final result:{result}')


if __name__ == '__main__':
    main()
