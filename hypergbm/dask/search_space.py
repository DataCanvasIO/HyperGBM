# -*- coding:utf-8 -*-
"""

"""
from functools import partial

from hypergbm.dask import dask_transformers as tf, dask_ops as ops
from hypergbm.estimators import LightGBMDaskEstimator, CatBoostDaskEstimator, XGBoostDaskEstimator
from hypergbm.pipeline import Pipeline, DataFrameMapper
from hypergbm.search_space import GeneralSearchSpaceGenerator
from hypernets.core.search_space import get_default_space
from hypernets.tabular import dask_ex as dex
from hypernets.tabular.column_selector import column_object, column_all
from hypernets.utils import logging

logger = logging.get_logger(__name__)


def _transformer_decorater(cache, cache_key, remove_keys, transformer):
    persister = tf.DataCacher(cache,
                              name=f'{transformer.name}_persister',
                              cache_key=cache_key,
                              remove_keys=remove_keys,
                              fit_transform=True,
                              transform=True)
    transformer = persister(transformer)

    return transformer


def _dfm_decorater(cache, cache_key, remove_keys, dfm):
    persister = tf.DataCacher(cache,
                              name=f'dfm_persister',
                              cache_key=cache_key,
                              remove_keys=remove_keys,
                              fit_transform=True,
                              transform=True)
    pipeline = Pipeline([persister], name='dfm_cacher', columns=column_all)
    return pipeline(dfm)


class DaskGeneralSearchSpaceGenerator(GeneralSearchSpaceGenerator):

    def __init__(self, enable_lightgbm=True, enable_xgb=True, enable_catboost=True, enable_persist=True, **kwargs):
        super().__init__(enable_lightgbm=enable_lightgbm, enable_xgb=enable_xgb, enable_catboost=enable_catboost,
                         enable_histgb=False, **kwargs)

        self.enable_persist = enable_persist

    @property
    def default_xgb_init_kwargs(self):
        return {**super().default_xgb_init_kwargs, 'tree_method': 'approx'}

    @property
    def estimators(self):
        r = {}

        if self.enable_xgb:
            r['xgb'] = (XGBoostDaskEstimator, self.default_xgb_init_kwargs, self.default_xgb_fit_kwargs)

        if self.enable_lightgbm:
            r['lightgbm'] = (LightGBMDaskEstimator, self.default_lightgbm_init_kwargs, self.default_lightgbm_fit_kwargs)

        if dex.is_local_dask():
            if self.enable_catboost:
                r['catboost'] = \
                    (CatBoostDaskEstimator, self.default_catboost_init_kwargs, self.default_catboost_fit_kwargs)

        return r

    def create_preprocessor(self, hyper_input, options):
        cat_pipeline_mode = options.pop('cat_pipeline_mode', 'simple')
        dataframe_mapper_default = options.pop('dataframe_mapper_default', False)

        if self.enable_persist:
            space = get_default_space()
            assert not hasattr(space, 'cache_')

            cache = {}
            setattr(space, 'cache_', cache)
            dfm_decorater = partial(_dfm_decorater, cache, 'dfm', 'num,cat')
            num_transformer_decorater = partial(_transformer_decorater, cache, 'num', 'dfm')
            cat_transformer_decorater = partial(_transformer_decorater, cache, 'cat', 'dfm')
        else:
            dfm_decorater = ops.default_transformer_decorator
            num_transformer_decorater = ops.default_transformer_decorator
            cat_transformer_decorater = ops.default_transformer_decorator

        num_pipeline = ops.numeric_pipeline_complex(decorate=num_transformer_decorater)(hyper_input)
        cat_pipeline = ops.categorical_pipeline_complex(decorate=cat_transformer_decorater)(hyper_input)

        dfm = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                              df_out_dtype_transforms=[(column_object, 'int')])([num_pipeline, cat_pipeline])
        preprocessor = dfm_decorater(dfm)

        return preprocessor


search_space_general = DaskGeneralSearchSpaceGenerator(n_estimators=200, enable_persist=False)

#
# def _build_estimator_dapater(fn_call, *args, **kwargs):
#     r = fn_call(*args, **kwargs)
#     r = dex.wrap_local_estimator(r)
#     return r
#
#
# def adapt_local_estimator(estimator):
#     fn_name = '_build_estimator'
#     fn_name_original = f'{fn_name}_adapted'
#     assert hasattr(estimator, fn_name) and not hasattr(estimator, fn_name_original)
#
#     fn = getattr(estimator, fn_name)
#     setattr(estimator, fn_name_original, fn)
#     setattr(estimator, fn_name, partial(_build_estimator_dapater, fn))
#     return estimator
