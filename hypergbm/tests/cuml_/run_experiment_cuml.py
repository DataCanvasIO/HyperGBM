# -*- coding:utf-8 -*-
"""

"""

import cudf

from hypergbm import make_experiment
from hypernets.tabular import get_tool_box
from hypernets.tabular.datasets import dsutils


def main(target='y', dtype=None, max_trials=3, drift_detection=False, clear_cache=True, **kwargs):
    tb = get_tool_box(cudf.DataFrame)
    assert isinstance(tb, type) and tb.__name__ == 'CumlToolBox'

    print("preparing...")
    df = dsutils.load_bank()
    if dtype is not None:
        df[target] = df[target].astype(dtype)
    df, = tb.from_local(df)
    assert isinstance(df, cudf.DataFrame)

    df_train, df_test = tb.train_test_split(df, test_size=0.5, random_state=123)
    X_test = df_test
    y_test = X_test.pop(target)

    exp = make_experiment(df_train, target=target,
                          test_data=X_test.copy(),
                          max_trials=max_trials,
                          drift_detection=drift_detection,
                          clear_cache=clear_cache,
                          **kwargs)
    print('experiment:', f'{[s.name for s in exp.steps]}', 'random_state', exp.random_state)

    print("training...")
    estimator = exp.run()
    print('estimator pipeline:', f'{[s[0] for s in estimator.steps]}')

    print("scoring...")
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)

    task = exp.task
    if task == 'regression':
        metrics = ['mse', 'mae', 'msle', 'rmse', 'r2']
    else:
        metrics = ['auc', 'accuracy', 'f1', 'recall']
    result = tb.metrics.calc_score(y_test, y_pred, y_proba, task=task, metrics=metrics,
                                   pos_label=kwargs.get('pos_label', None))

    print(result)

    return exp, estimator


def main_with_adaption(target='y', dtype=None, max_trials=3, drift_detection=False, clear_cache=True, **kwargs):
    print("preparing...")
    df = dsutils.load_bank()
    tb = get_tool_box(df)
    if dtype is not None:
        df[target] = df[target].astype(dtype)
    df, = tb.from_local(df)

    df_train, df_test = tb.train_test_split(df, test_size=0.5, random_state=123)
    X_test = df_test
    y_test = X_test.pop(target)

    exp = make_experiment(df_train, target=target,
                          test_data=X_test.copy(),
                          data_adaption_target='cuml',
                          max_trials=max_trials,
                          drift_detection=drift_detection,
                          clear_cache=clear_cache,
                          **kwargs)
    print('experiment:', f'{[s.name for s in exp.steps]}', 'random_state', exp.random_state)

    print("training...")
    estimator = exp.run()
    print('estimator pipeline:', f'{[s[0] for s in estimator.steps]}')

    print("scoring...")
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)

    task = exp.task
    if task == 'regression':
        metrics = ['mse', 'mae', 'msle', 'rmse', 'r2']
    else:
        metrics = ['auc', 'accuracy', 'f1', 'recall']
    result = tb.metrics.calc_score(y_test, y_pred, y_proba, task=task, metrics=metrics,
                                   pos_label=kwargs.get('pos_label', None))

    print(result)

    return exp, estimator


if __name__ == '__main__':
    main_with_adaption(target='y', reward_metric='auc', ensemble_size=10, pos_label='yes', log_level='info',
                       max_trials=10,clear_cache=False)
    # main(target='y', reward_metric='auc', ensemble_size=10, pos_label='yes', log_level='info', max_trials=10)
    # main(target='y', max_trials=10, cv=False, ensemble_size=0, verbose=0, pos_label='yes', )
    # main(target='day', reward_metric='f1', ensemble_size=10, log_level='info', max_trials=5)
    # main(target='day', dtype='str', reward_metric='f1', ensemble_size=0,  log_level='info',  max_trials=6)
    # main(target='age', dtype='float', ensemble_size=10, log_level='info', max_trials=8)
