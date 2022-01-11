# -*- coding:utf-8 -*-
"""

"""
from functools import partial

from hypergbm.cfg import HyperGBMCfg as cfg
from hypergbm.dask import dask_transformers as tf, dask_ops as ops
from hypergbm.search_space import GeneralSearchSpaceGenerator
from hypernets.core.search_space import get_default_space
from hypernets.pipeline.base import Pipeline
from hypernets.tabular.dask_ex import DaskToolBox
from hypernets.utils import logging
from ._estimators import LightGBMDaskEstimator, CatBoostDaskEstimator, XGBoostDaskEstimator, HistGBDaskEstimator
from ._estimators import lgbm_dask_distributed, xgb_dask_distributed, catboost_dask_distributed

logger = logging.get_logger(__name__)


def _transformer_decorator(cache, cache_key, remove_keys, transformer):
    persister = tf.DataCacher(cache,
                              name=f'{transformer.name}_persister',
                              cache_key=cache_key,
                              remove_keys=remove_keys,
                              fit_transform=True,
                              transform=True)
    transformer = persister(transformer)

    return transformer


def _dfm_decorator(cache, cache_key, remove_keys, dfm):
    persister = tf.DataCacher(cache,
                              name=f'dfm_persister',
                              cache_key=cache_key,
                              remove_keys=remove_keys,
                              fit_transform=True,
                              transform=True)
    pipeline = Pipeline([persister], name='dfm_cacher', columns=DaskToolBox.column_selector.column_all)
    return pipeline(dfm)


class DaskGeneralSearchSpaceGenerator(GeneralSearchSpaceGenerator):
    xgboost_estimator_cls = XGBoostDaskEstimator
    lightgbm_estimator_cls = LightGBMDaskEstimator
    catboost_estimator_cls = CatBoostDaskEstimator
    histgb_estimator_cls = HistGBDaskEstimator

    def __init__(self, enable_lightgbm=True, enable_xgb=True, enable_catboost=True, enable_histgb=False,
                 enable_persist=True, **kwargs):
        # warning_msg = 'Your dask cluster does not support training with %s.'
        # assert _is_local_dask or lgbm_dask_distributed or not enable_lightgbm, warning_msg % 'lightgbm'
        # assert _is_local_dask or xgb_dask_distributed or not enable_xgb, warning_msg % 'xgboost'
        # assert _is_local_dask or catboost_dask_distributed or not enable_catboost, warning_msg % 'catboost'
        # assert _is_local_dask or histgb_dask_distributed or not enable_histgb, warning_msg % 'lightgbm'

        self.enable_persist = enable_persist

        super().__init__(enable_lightgbm=enable_lightgbm, enable_xgb=enable_xgb, enable_catboost=enable_catboost,
                         enable_histgb=enable_histgb, **kwargs)

    @property
    def default_xgb_init_kwargs(self):
        return {**super().default_xgb_init_kwargs, 'tree_method': 'approx'}

    def create_preprocessor(self, hyper_input, options):
        cat_pipeline_mode = options.pop('cat_pipeline_mode', 'simple')
        dataframe_mapper_default = options.pop('dataframe_mapper_default', False)

        if self.enable_persist:
            space = get_default_space()
            assert not hasattr(space, 'cache_')

            cache = {}
            setattr(space, 'cache_', cache)
            dfm_decorator = partial(_dfm_decorator, cache, 'dfm', 'num,cat')
            num_transformer_decorator = partial(_transformer_decorator, cache, 'num', 'dfm')
            cat_transformer_decorator = partial(_transformer_decorator, cache, 'cat', 'dfm')
        else:
            dfm_decorator = ops.default_transformer_decorator
            num_transformer_decorator = ops.default_transformer_decorator
            cat_transformer_decorator = ops.default_transformer_decorator

        num_pipeline = ops.numeric_pipeline_complex(decorate=num_transformer_decorator)(hyper_input)
        cat_pipeline = ops.categorical_pipeline_complex(decorate=cat_transformer_decorator)(hyper_input)

        column_object = DaskToolBox.column_selector.column_object
        dfm = tf.DaskDataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                     df_out_dtype_transforms=[(column_object, 'int')])([num_pipeline, cat_pipeline])
        preprocessor = dfm_decorator(dfm)

        return preprocessor


_is_local_dask = DaskToolBox.is_local_dask()

search_space_general = DaskGeneralSearchSpaceGenerator(
    enable_lightgbm=cfg.estimator_lightgbm_enabled and (_is_local_dask or lgbm_dask_distributed),
    enable_xgb=cfg.estimator_xgboost_enabled and (_is_local_dask or xgb_dask_distributed),
    enable_catboost=cfg.estimator_catboost_enabled and (_is_local_dask or catboost_dask_distributed),
    enable_histgb=cfg.estimator_histgb_enabled and _is_local_dask,
    n_estimators=200,
    enable_persist=False)

#
# def _build_estimator_dapater(fn_call, *args, **kwargs):
#     r = fn_call(*args, **kwargs)
#     r = DaskToolBox.wrap_local_estimator(r)
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
