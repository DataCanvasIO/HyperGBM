# -*- coding:utf-8 -*-
"""

"""
from functools import partial

from hypernets.tabular.cache import cache
from hypernets.tabular.cuml_ex import CumlToolBox
from hypernets.utils import const, logging
from .. import estimators as es

logger = logging.get_logger(__name__)


@cache()
def _detect_estimator(name_or_cls, task, *,
                      init_kwargs=None, fit_kwargs=None, n_samples=100, n_features=5):
    r = CumlToolBox.estimator_detector(name_or_cls, task,
                                       init_kwargs=init_kwargs,
                                       fit_kwargs=fit_kwargs,
                                       n_samples=n_samples,
                                       n_features=n_features)()

    logger.info(f'detect_estimator {name_or_cls} as {r}')
    return r


_detected_lgbm = _detect_estimator('lightgbm.LGBMClassifier', const.TASK_BINARY,
                                   init_kwargs={'device': 'GPU'},
                                   fit_kwargs={})
_detected_xgb = _detect_estimator('xgboost.XGBClassifier', const.TASK_BINARY,
                                  init_kwargs={'tree_method': 'gpu_hist', 'use_label_encoder': False,
                                               'verbosity': 0, },
                                  fit_kwargs={})
# _detected_catboost = _detect_estimator('catboost.CatBoostClassifier', const.TASK_BINARY,
#                                        init_kwargs={'task_type': 'GPU', 'verbose': 0},
#                                        fit_kwargs={})
_detected_catboost = {'installed', 'initialized', 'fitted'}

_FEATURE_FOR_GPU = 'fitted'  # succeed in fitting with pandas data
_FEATURE_FOR_CUML = 'fitted_with_cuml'  # succeed in fitting with cuml data


def wrap_estimator(estimator, methods=None):
    estimator = CumlToolBox.wrap_local_estimator(estimator, methods=methods)
    setattr(estimator, 'as_local', partial(_as_local, estimator, methods=methods))
    return estimator


def _as_local(estimator, methods=None):
    if isinstance(estimator, es.LabelEncoderMixin):
        le = estimator.get_label_encoder()
        if hasattr(le, 'as_local'):
            estimator.set_label_encoder(le.as_local())
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


class HistGBCumlEstimator(es.HistGBEstimator):
    def _build_estimator(self, task, kwargs):
        est = super()._build_estimator(task, kwargs)
        est = wrap_estimator(est)
        return est


#
# catboost override the __getstate__/__setstate__ with their method, so we wrap it with class
#

class CatBoostClassifierCumlWrapper(es.CatBoostClassifierWrapper):
    def fit(self, *args, **kwargs):
        if _FEATURE_FOR_CUML in _detected_catboost:
            return super().fit(*args, **kwargs)
        else:
            return CumlToolBox.call_local(super().fit, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return CumlToolBox.call_local(super().predict, *args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return CumlToolBox.call_local(super().predict_proba, *args, **kwargs)

    def as_local(self):
        state = self.__getstate__()
        target = es.CatBoostClassifierWrapper()
        target.__setstate__(state)
        return target


class CatBoostRegressionCumlWrapper(es.CatBoostRegressionWrapper):
    def fit(self, *args, **kwargs):
        if _FEATURE_FOR_CUML in _detected_catboost:
            return super().fit(*args, **kwargs)
        else:
            return CumlToolBox.call_local(super().fit, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return CumlToolBox.call_local(super().predict, *args, **kwargs)

    def as_local(self):
        state = self.__getstate__()
        target = es.CatBoostRegressionWrapper()
        target.__setstate__(state)
        return target


class CatBoostCumlEstimator(es.CatBoostEstimator):
    def _build_estimator(self, task, kwargs):
        if task == const.TASK_REGRESSION:
            est = CatBoostRegressionCumlWrapper(**kwargs)
        else:
            est = CatBoostClassifierCumlWrapper(**kwargs)
        return est
