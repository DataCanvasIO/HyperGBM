# -*- coding:utf-8 -*-
"""

"""
import cudf

# from hypergbm import HyperGBM
from hypergbm.cuml import CumlGeneralSearchSpaceGenerator, CumlHyperGBM, CumlHyperGBMEstimator
from hypernets.core import set_random_state, SummaryCallback
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.tabular.cuml_ex import CumlToolBox
from hypernets.tabular.datasets import dsutils


def main(enable_lightgbm=True, enable_xgb=True, enable_catboost=True, enable_histgb=False,
         n_estimators=200, class_balancing=None,
         num_pipeline_mode='simple', cat_pipeline_mode='simple',
         target='y', dtype=None, max_trials=3, **kwargs):
    set_random_state(123)

    print("preparing...")
    df = dsutils.load_bank()
    y = df.pop(target)
    if dtype is not None:
        y = y.astype(dtype)
    X, y = CumlToolBox.from_local(df, y)
    assert isinstance(X, cudf.DataFrame)
    assert isinstance(y, cudf.Series)

    dc = CumlToolBox.data_cleaner()
    X, y = dc.fit_transform(X, y)
    X_train, X_test, y_train, y_test = CumlToolBox.train_test_split(X, y, test_size=0.4, random_state=42)

    task, _ = CumlToolBox.infer_task_type(y)
    reward_metric = 'rmse' if task == 'regression' else 'accuracy'
    search_space = CumlGeneralSearchSpaceGenerator(n_estimators=n_estimators,
                                                   enable_lightgbm=enable_lightgbm,
                                                   enable_xgb=enable_xgb,
                                                   enable_catboost=enable_catboost,
                                                   enable_histgb=enable_histgb,
                                                   class_balancing=class_balancing,
                                                   cat_pipeline_mode=cat_pipeline_mode,
                                                   num_pipeline_mode=num_pipeline_mode,
                                                   )
    rs = RandomSearcher(search_space, optimize_direction='max')
    hk = CumlHyperGBM(rs, task=task, reward_metric=reward_metric, callbacks=[SummaryCallback(), ])

    print('searching...')
    hk.search(X_train, y_train, X_test, y_test, max_trials=max_trials, **kwargs)
    best_trial = hk.get_best_trial()
    print(f'best_train:{best_trial}')

    estimator = hk.final_train(best_trial.space_sample, X_train, y_train, pos_label='yes')
    # score = estimator.predict(X_test)
    assert isinstance(estimator, CumlHyperGBMEstimator)

    result = estimator.evaluate(X_test, y_test, metrics=[reward_metric], )
    print(f'final result:{result}')

    estimator = estimator.as_local()
    X_test, y_test = CumlToolBox.to_local(X_test, y_test)
    result = estimator.evaluate(X_test, y_test, metrics=[reward_metric], )
    print(f'final result with local data:{result}')

    return estimator


if __name__ == '__main__':
    # from hypernets.utils import logging
    #
    # logging.set_level('warn')
    main(target='age', dtype='float', max_trials=3, verbose=0, )
    # main(target='day', max_trials=3, verbose=0,   )
    # main(target='y', max_trials=3, verbose=0, pos_label='yes', )
