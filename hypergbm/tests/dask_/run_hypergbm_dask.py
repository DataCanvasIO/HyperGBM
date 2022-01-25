# -*- coding:utf-8 -*-
"""

"""
from dask.distributed import default_client

from hypergbm import make_experiment
from hypernets.tabular.dask_ex import DaskToolBox
from hypernets.tabular.datasets import dsutils
from hypernets.tests.tabular.tb_dask import setup_dask


def main(max_trials=10, log_level='info'):
    # setup Dask cluster
    # client = Client("tcp://127.0.0.1:64958")
    # client = Client(processes=False, threads_per_worker=2, n_workers=1, memory_limit='4GB')
    # client = Client(processes=True, threads_per_worker=4, n_workers=2, memory_limit='10GB')
    setup_dask(None)
    client = default_client()
    worker_count = len(client.ncores())
    print(client)

    # prepare data
    target_name = 'y'
    df = dsutils.load_bank_by_dask()
    # df = df.sample(frac=0.1)
    df = df.repartition(npartitions=2)

    df_train, df_test = DaskToolBox.train_test_split(df, test_size=0.5, random_state=42, shuffle=False)
    df_train, df_test = client.persist([df_train, df_test])

    # make experiment and run it
    experiment = make_experiment(df_train, target=target_name,
                                 down_sample_search=False, cv=False,
                                 early_stopping_rounds=0, class_balancing=True,
                                 max_trials=max_trials, log_level=log_level, )
    estimator = experiment.run()
    print(estimator)

    best_trial = experiment.hyper_model_.get_best_trial()
    print(f'best_trial: {best_trial}')

    # import pickle
    # with open('/tmp/model.pkl', 'wb') as f:
    #     pickle.dump(estimator, f, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('/tmp/model.pkl', 'rb') as f:
    #     estimator = pickle.load(f)

    # evaluate the trained estimator
    X_test = df_test.copy()
    y_test = X_test.pop(target_name)
    X_test, y_test = client.persist([X_test, y_test])
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)
    result = DaskToolBox.metrics.calc_score(y_test, y_pred, y_proba,
                                            metrics=['accuracy', 'auc', 'logloss', 'f1', 'recall', 'precision'],
                                            pos_label='yes')
    print(f'final result: {result}')

    return estimator


if __name__ == '__main__':
    main(max_trials=3)
