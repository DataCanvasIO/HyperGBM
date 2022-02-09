# -*- coding:utf-8 -*-
"""

"""
from hypergbm.estimators import detect_lgbm_gpu
from hypergbm.search_space import GeneralSearchSpaceGenerator
from hypernets.core import Choice
from hypernets.pipeline.base import DataFrameMapper
from hypernets.tabular.cuml_ex import CumlToolBox
from hypernets.utils import logging
from . import _estimators as es
from . import _ops as ops
from ..cfg import HyperGBMCfg as cfg

logger = logging.get_logger(__name__)


class CumlDataFrameMapper(DataFrameMapper):
    @staticmethod
    def _create_dataframe_mapper(features, **kwargs):
        dfm_cls = CumlToolBox.transformers['DataFrameMapper']
        return dfm_cls(features=features, **kwargs)


class CumlGeneralSearchSpaceGenerator(GeneralSearchSpaceGenerator):
    lightgbm_estimator_cls = es.LightGBMCumlEstimator
    xgboost_estimator_cls = es.XGBoostCumlEstimator
    catboost_estimator_cls = es.CatBoostCumlEstimator
    histgb_estimator_cls = es.HistGBCumlEstimator

    def create_preprocessor(self, hyper_input, options):
        cat_pipeline_mode = options.pop('cat_pipeline_mode', cfg.category_pipeline_mode)
        num_pipeline_mode = options.pop('num_pipeline_mode', cfg.numeric_pipeline_mode)
        dataframe_mapper_default = options.pop('dataframe_mapper_default', False)

        if num_pipeline_mode == 'simple':
            num_pipeline = ops.numeric_pipeline_simple()(hyper_input)
        else:
            num_pipeline = ops.numeric_pipeline_complex()(hyper_input)

        if cat_pipeline_mode == 'simple':
            cat_pipeline = ops.categorical_pipeline_simple()(hyper_input)
        else:
            cat_pipeline = ops.categorical_pipeline_complex()(hyper_input)

        column_object = CumlToolBox.column_selector.column_object
        dfm = CumlDataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                  df_out_dtype_transforms=[(column_object, 'int')])([num_pipeline, cat_pipeline])

        return dfm

    @property
    def default_lightgbm_init_kwargs(self):
        r = super().default_lightgbm_init_kwargs

        if detect_lgbm_gpu():
            r = {**r,
                 'device': 'GPU',
                 'max_bin': Choice([63, 127]),
                 }

        return r

    @property
    def default_xgb_init_kwargs(self):
        return {**super().default_xgb_init_kwargs,
                'tree_method': 'gpu_hist',
                }

    @property
    def default_catboost_init_kwargs(self):
        return {**super().default_catboost_init_kwargs,
                'task_type': 'GPU',
                }


search_space_general = \
    CumlGeneralSearchSpaceGenerator(enable_lightgbm=cfg.estimator_lightgbm_enabled,
                                    enable_xgb=cfg.estimator_xgboost_enabled,
                                    enable_catboost=cfg.estimator_catboost_enabled,
                                    enable_histgb=cfg.estimator_histgb_enabled,
                                    n_estimators=200)
