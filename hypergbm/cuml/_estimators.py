# -*- coding:utf-8 -*-
"""

"""
from functools import partial

from hypernets.tabular.cuml_ex import CumlToolBox
from hypernets.utils import const
from .. import estimators as es

_detector = CumlToolBox.estimator_detector
_detected_lgbm = _detector('lightgbm.LGBMClassifier', const.TASK_BINARY,
                           init_kwargs={'device': 'GPU'},
                           fit_kwargs={})()
_detected_xgb = _detector('xgboost.XGBClassifier', const.TASK_BINARY,
                          init_kwargs={'tree_method': 'gpu_hist', 'use_label_encoder': False, 'verbosity': 0, },
                          fit_kwargs={})()
_detected_catboost = _detector('catboost.CatBoostClassifier', const.TASK_BINARY,
                               init_kwargs={'task_type': 'GPU', 'verbose': 0},
                               fit_kwargs={})()

_FEATURE_FOR_GPU = 'fitted'  # can fit with pandas data
_FEATURE_FOR_CUML = 'fitted_with_cuml'  # can fit with cuml data


def wrap_estimator(estimator, methods=None):
    estimator = CumlToolBox.wrap_local_estimator(estimator, methods=methods)
    setattr(estimator, 'as_local', partial(_as_local, estimator, methods=methods))
    return estimator


def _as_local(estimator, methods=None):
    estimator = CumlToolBox.unwrap_local_estimator(estimator, methods=methods)
    delattr(estimator, 'as_local')
    return estimator


class LightGBMCumlEstimator(es.LightGBMEstimator):
    def _build_estimator(self, task, kwargs):
        est = super()._build_estimator(task, kwargs)
        methods = {'predict', 'predict_proba'} if _FEATURE_FOR_CUML in _detected_lgbm else None
        est = wrap_estimator(est, methods=methods)
        return est


class XGBoostCumlEstimator(es.XGBoostEstimator):
    def _build_estimator(self, task, kwargs):
        est = super()._build_estimator(task, kwargs)
        methods = {'predict', 'predict_proba'} if _FEATURE_FOR_CUML in _detected_xgb else None
        est = wrap_estimator(est, methods=methods)
        return est


class CatBoostCumlEstimator(es.CatBoostEstimator):
    def _build_estimator(self, task, kwargs):
        est = super()._build_estimator(task, kwargs)
        methods = {'predict', 'predict_proba'} if _FEATURE_FOR_CUML in _detected_catboost else None
        est = wrap_estimator(est, methods=methods)
        return est


class HistGBCumlEstimator(es.HistGBEstimator):
    def _build_estimator(self, task, kwargs):
        est = super()._build_estimator(task, kwargs)
        est = wrap_estimator(est)
        return est
