# -*- coding:utf-8 -*-
"""

"""

from functools import partial

import numpy as np

from hypergbm.estimators import LightGBMDaskEstimator, XGBoostDaskEstimator
from hypergbm.pipeline import Pipeline, DataFrameMapper
from hypernets.core.ops import ModuleChoice, Optional, HyperInput
from hypernets.core.search_space import Choice, HyperSpace
from hypernets.utils import logging
from tabular_toolbox.column_selector import column_object_category_bool, column_number_exclude_timedelta, \
    column_object, column_all
from . import dask_transformers as tf

logger = logging.get_logger(__name__)


def default_transformer_decorator(transformer):
    return transformer


def categorical_pipeline_simple(impute_strategy='constant', seq_no=0,
                                decorate=default_transformer_decorator):
    pipeline = Pipeline([
        decorate(tf.SimpleImputer(missing_values=None, strategy=impute_strategy,
                                  name=f'categorical_imputer_{seq_no}', fill_value='')),
        decorate(tf.SafeOrdinalEncoder(name=f'categorical_label_encoder_{seq_no}'))
    ],
        columns=column_object_category_bool,
        name=f'categorical_pipeline_simple_{seq_no}',
    )
    return pipeline


def categorical_pipeline_complex(impute_strategy=None, svd_components=5, seq_no=0,
                                 decorate=default_transformer_decorator):
    if impute_strategy is None:
        impute_strategy = Choice(['constant', 'most_frequent'])
    elif isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)
    if isinstance(svd_components, list):
        svd_components = Choice(svd_components)

    def onehot_svd():
        onehot = decorate(tf.OneHotEncoder(name=f'categorical_onehot_{seq_no}', sparse=False))
        svd = decorate(tf.TruncatedSVD(n_components=svd_components, name=f'categorical_svd_{seq_no}'))
        optional_svd = Optional(svd, name=f'categorical_optional_svd_{seq_no}', keep_link=True)(onehot)
        return optional_svd

    imputer = decorate(tf.SimpleImputer(missing_values=np.nan, strategy=impute_strategy,
                                        name=f'categorical_imputer_{seq_no}', fill_value=''))

    # label_encoder = decorate(tf.MultiLabelEncoder(name=f'categorical_label_encoder_{seq_no}'))
    label_encoder = decorate(tf.SafeOrdinalEncoder(name=f'categorical_label_encoder_{seq_no}'))

    onehot = onehot_svd()

    # onehot = OneHotEncoder(name=f'categorical_onehot_{seq_no}', sparse=False)
    le_or_onehot_pca = ModuleChoice([label_encoder, onehot], name=f'categorical_le_or_onehot_svd_{seq_no}')

    pipeline = Pipeline([imputer, le_or_onehot_pca],
                        name=f'categorical_pipeline_complex_{seq_no}',
                        columns=column_object_category_bool)
    return pipeline


def numeric_pipeline(impute_strategy='mean', seq_no=0, decorate=default_transformer_decorator):
    pipeline = Pipeline([
        decorate(tf.SimpleImputer(missing_values=np.nan, strategy=impute_strategy,
                                  name=f'numeric_imputer_{seq_no}', fill_value=0)),
        decorate(tf.StandardScaler(name=f'numeric_standard_scaler_{seq_no}'))
    ],
        columns=column_number_exclude_timedelta,
        name=f'numeric_pipeline_simple_{seq_no}',
    )
    return pipeline


def numeric_pipeline_complex(impute_strategy=None, seq_no=0, decorate=default_transformer_decorator):
    if impute_strategy is None:
        impute_strategy = Choice(['mean', 'median', 'constant', 'most_frequent'])
    elif isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)
    # reduce_skewness_kurtosis = SkewnessKurtosisTransformer(transform_fn=Choice([np.log, np.log10, np.log1p]))
    # reduce_skewness_kurtosis_optional = Optional(reduce_skewness_kurtosis, keep_link=True,
    #                                             name=f'numeric_reduce_skewness_kurtosis_optional_{seq_no}')

    imputer = decorate(tf.SimpleImputer(missing_values=np.nan, strategy=impute_strategy,
                                        name=f'numeric_imputer_{seq_no}', fill_value=0))
    scaler_options = ModuleChoice(
        [
            decorate(tf.StandardScaler(name=f'numeric_standard_scaler_{seq_no}')),
            decorate(tf.MinMaxScaler(name=f'numeric_minmax_scaler_{seq_no}')),
            decorate(tf.MaxAbsScaler(name=f'numeric_maxabs_scaler_{seq_no}')),
            decorate(tf.RobustScaler(name=f'numeric_robust_scaler_{seq_no}'))
        ], name=f'numeric_or_scaler_{seq_no}'
    )
    scaler_optional = Optional(scaler_options, keep_link=True, name=f'numeric_scaler_optional_{seq_no}')

    pipeline = Pipeline([imputer, scaler_optional],
                        name=f'numeric_pipeline_complex_{seq_no}',
                        columns=column_number_exclude_timedelta)
    return pipeline


def get_space_num_cat_pipeline_complex(dataframe_mapper_default=False,
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

        num_pipeline = numeric_pipeline_complex(decorate=partial(transformer_decorater, 'num', 'dfm'))(input)
        cat_pipeline = categorical_pipeline_complex(decorate=partial(transformer_decorater, 'cat', 'dfm'))(input)

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
