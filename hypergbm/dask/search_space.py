# -*- coding:utf-8 -*-
"""

"""
from functools import partial

from hypergbm.dask import dask_transformers as tf, dask_ops as ops
from hypergbm.estimators import LightGBMEstimator, XGBoostEstimator, CatBoostEstimator, XGBoostDaskEstimator
from hypergbm.pipeline import Pipeline, DataFrameMapper
from hypernets.core.ops import ModuleChoice, HyperInput
from hypernets.core.search_space import Choice, HyperSpace, Int
from hypernets.utils import logging
from tabular_toolbox import dask_ex as dex
from tabular_toolbox.column_selector import column_object, column_all

logger = logging.get_logger(__name__)


def search_space_general(dataframe_mapper_default=False,
                         lightgbm_init_kwargs=None,
                         lightgbm_fit_kwargs=None,
                         xgb_init_kwargs=None,
                         xgb_fit_kwargs=None,
                         catboost_init_kwargs=None,
                         catboost_fit_kwargs=None,
                         class_balancing=None,
                         n_esitimators=200,
                         enable_persist=True,
                         **kwargs):
    assert dex.dask_enabled(), 'Dask client must be initialized.'

    if lightgbm_init_kwargs is None: lightgbm_init_kwargs = {}
    if lightgbm_fit_kwargs is None: lightgbm_fit_kwargs = {}
    if xgb_init_kwargs is None: xgb_init_kwargs = {}
    if xgb_fit_kwargs is None: xgb_fit_kwargs = {}
    if catboost_init_kwargs is None: catboost_init_kwargs = {}
    if catboost_fit_kwargs is None: catboost_fit_kwargs = {}

    lightgbm_fit_kwargs.update(kwargs)
    xgb_fit_kwargs.update(kwargs)
    catboost_fit_kwargs.update(kwargs)

    cache = {}

    def transformer_decorater(cache_key, remove_keys, transformer):
        if enable_persist:
            persister = tf.DataCacher(cache,
                                      name=f'{transformer.name}_persister',
                                      cache_key=cache_key,
                                      remove_keys=remove_keys,
                                      fit_transform=True,
                                      transform=True)
            transformer = persister(transformer)

        return transformer

    def dfm_decorater(cache_key, remove_keys, dfm):
        if enable_persist:
            persister = tf.DataCacher(cache,
                                      name=f'dfm_persister',
                                      cache_key=cache_key,
                                      remove_keys=remove_keys,
                                      fit_transform=True,
                                      transform=True)
            pipeline = Pipeline([persister], name='dfm_cacher', columns=column_all)
            return pipeline(dfm)

        return dfm

    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')

        num_pipeline = ops.numeric_pipeline_complex(decorate=partial(transformer_decorater, 'num', 'dfm'))(input)
        cat_pipeline = ops.categorical_pipeline_complex(decorate=partial(transformer_decorater, 'cat', 'dfm'))(input)

        dfm = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                              df_out_dtype_transforms=[(column_object, 'int')])([num_pipeline, cat_pipeline])
        union_pipeline = dfm_decorater('dfm', 'num,cat', dfm)

        lightgbm_init_kwargs = {
            # 'n_estimators': n_esitimators,  # Choice([10, 30, 50, 100, 200, 300, 500]),
            'n_estimators': Choice([10, 30, 100, 200, 300]) if n_esitimators is None else n_esitimators,
            'boosting_type': Choice(['gbdt', 'dart', 'goss']),
            'num_leaves': Int(15, 513, 5),
            'learning_rate': Choice([0.001, 0.01, 0.5, 0.1]),
            'max_depth': Choice([3, 5, 7, 10]),
            'reg_alpha': Choice([0.001, 0.01, 0.1, 1, 10, 100]),
            'reg_lambda': Choice([0.001, 0.01, 0.1, 0.5, 1]),
            'class_balancing': class_balancing,
            # 'class_weight': 'balanced',
            # subsample_for_bin = 200000, objective = None, class_weight = None,
            #  min_split_gain = 0., min_child_weight = 1e-3, min_child_samples = 20,
            **lightgbm_init_kwargs
        }

        xgb_init_kwargs = {
            'booster': Choice(['gbtree', 'dart']),
            'max_depth': Choice([3, 5, 7, 9, 15]),
            'n_estimators': Choice([10, 30, 100, 200, 300]) if n_esitimators is None else n_esitimators,
            'learning_rate': Choice([0.001, 0.01, 0.5, 0.1]),
            'min_child_weight': Choice([1, 5, 10]),
            'gamma': Choice([0.5, 1, 1.5, 2, 5]),
            'reg_alpha': Choice([0.001, 0.01, 0.1, 1, 10, 100]),
            'reg_lambda': Choice([0.001, 0.01, 0.1, 0.5, 1]),
            'class_balancing': class_balancing,

            # 'subsample': Choice([0.6, 0.8, 1.0]),
            # 'colsample_bytree': Choice([0.6, 0.8, 1.0]),
            # 'scale_pos_weight': Int(1,5,1),
            'tree_method': 'approx',  # for dask
            **xgb_init_kwargs
        }

        catboost_init_kwargs = {
            'silent': True,
            'n_estimators': Choice([10, 30, 100, 200, 300]) if n_esitimators is None else n_esitimators,
            'depth': Choice([3, 5, 7, 10]),
            'learning_rate': Choice([0.001, 0.01, 0.5, 0.1]),
            # 'iterations': Choice([30, 50, 100, 200, 300]),
            'l2_leaf_reg': Choice([None, 2, 10, 20, 30]),
            'class_balancing': class_balancing,

            # 'bagging_temperature': Choice([None, 0, 0.5, 1]),
            # 'random_strength': Choice([None, 1, 5, 10]),
            # 'class_weights': [0.59,3.07],
            # 'border_count': Choice([5, 10, 20, 32, 50, 100, 200]),
            **catboost_init_kwargs
        }

        if dex.is_local_dask():
            estimators = [
                LightGBMEstimator(fit_kwargs=lightgbm_fit_kwargs, **lightgbm_init_kwargs),
                XGBoostEstimator(fit_kwargs=xgb_fit_kwargs, **xgb_init_kwargs),
                # CatBoostEstimator(fit_kwargs=catboost_fit_kwargs, **catboost_init_kwargs),
            ]
            estimators = [adapt_local_estimator(est) for est in estimators]
        else:
            xgb_est = XGBoostDaskEstimator(fit_kwargs=xgb_fit_kwargs, **xgb_init_kwargs)
            estimators = [xgb_est]

        or_est = ModuleChoice(estimators, name='estimator_options')(union_pipeline)

        space.set_inputs(input)
    return space


def _build_estimator_dapater(fn_call, *args, **kwargs):
    r = fn_call(*args, **kwargs)
    r = dex.wrap_local_estimator(r)
    return r


def adapt_local_estimator(estimator):
    fn_name = '_build_estimator'
    fn_name_original = f'{fn_name}_adapted'
    assert hasattr(estimator, fn_name) and not hasattr(estimator, fn_name_original)

    fn = getattr(estimator, fn_name)
    setattr(estimator, fn_name_original, fn)
    setattr(estimator, fn_name, partial(_build_estimator_dapater, fn))
    return estimator
