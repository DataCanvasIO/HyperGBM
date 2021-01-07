# -*- coding:utf-8 -*-
"""

"""
from functools import partial

from hypergbm.estimators import LightGBMDaskEstimator, XGBoostDaskEstimator
from hypergbm.pipeline import Pipeline, DataFrameMapper
from hypernets.core.ops import ModuleChoice, HyperInput
from hypernets.core.search_space import Choice, HyperSpace
from hypernets.utils import logging
from tabular_toolbox.column_selector import column_object, column_all
from . import dask_transformers as tf, dask_ops as ops

logger = logging.get_logger(__name__)


def search_space_general(dataframe_mapper_default=False,
                         lightgbm_init_kwargs={},
                         lightgbm_fit_kwargs={},
                         xgb_init_kwargs={},
                         xgb_fit_kwargs={},
                         catboost_init_kwargs={},
                         catboost_fit_kwargs={},
                         enable_persist=True):
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
            'boosting_type': Choice(['gbdt', 'dart', 'goss']),
            'num_leaves': Choice([3, 5]),
            'learning_rate': 0.1,
            'n_estimators': Choice([10, 30, 50, 100, 200, 300, 500]),
            'max_depth': Choice([3, 5, 7]),
            'reg_alpha': Choice([1e-2, 0.1, 1, 100]),
            'reg_lambda': Choice([1e-2, 0.1, 1, 100]),
            'tree_learner': 'data',  # for dask
            **lightgbm_init_kwargs
            # subsample_for_bin = 200000, objective = None, class_weight = None,
            #  min_split_gain = 0., min_child_weight = 1e-3, min_child_samples = 20,
        }
        lightgbm_est = LightGBMDaskEstimator(fit_kwargs=lightgbm_fit_kwargs, **lightgbm_init_kwargs)

        xgb_init_kwargs = {
            'max_depth': Choice([3, 5, 7]),
            # 'n_estimators': Choice([10, 30, 50, 100, 200, 300, 500]),
            'n_estimators': Choice([10, 30, 50, ]),
            'learning_rate': 0.1,
            'min_child_weight': Choice([1, 5, 10]),
            'gamma': Choice([0.5, 1, 1.5, 2, 5]),
            'reg_alpha': Choice([1e-2, 0.1, 1, 100]),
            'reg_lambda': Choice([1e-2, 0.1, 1, 100]),
            'subsample': Choice([0.6, 0.8, 1.0]),
            'colsample_bytree': Choice([0.6, 0.8, 1.0]),
            'tree_method': 'approx',  # for dask
            **xgb_init_kwargs
        }
        xgb_est = XGBoostDaskEstimator(fit_kwargs=xgb_fit_kwargs, **xgb_init_kwargs)

        # catboost_init_kwargs = {
        #     'silent': True
        # }
        # catboost_est = CatBoostEstimator(task='binary', fit_kwargs=catboost_fit_kwargs, **catboost_init_kwargs)
        # or_est = ModuleChoice([lightgbm_est, xgb_est, catboost_est], name='estimator_options')(p3)

        # or_est = ModuleChoice([lightgbm_est, xgb_est], name='estimator_options')(p3)
        or_est = ModuleChoice([xgb_est], name='estimator_options')(union_pipeline)

        space.set_inputs(input)
    return space
