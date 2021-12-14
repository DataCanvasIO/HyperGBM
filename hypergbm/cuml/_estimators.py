# -*- coding:utf-8 -*-
"""

"""
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

_FEATURE_FOR_GPU = 'fitted'
_FEATURE_FOR_CUML = 'fitted_with_cuml'

# LightGBM
if _FEATURE_FOR_CUML in _detected_lgbm:
    LightGBMCumlEstimator = es.LightGBMEstimator
else:
    class LGBMClassifierCumlWrapper(es.LGBMClassifierWrapper):
        def fit(self, *args, **kwargs):
            return CumlToolBox.call_local(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return CumlToolBox.call_local(super().predict, *args, **kwargs)

        def predict_proba(self, *args, **kwargs):
            return CumlToolBox.call_local(super().predict_proba, *args, **kwargs)


    class LGBMRegressionCumlWrapper(es.LGBMRegressorWrapper):
        def fit(self, *args, **kwargs):
            return CumlToolBox.call_local(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return CumlToolBox.call_local(super().predict, *args, **kwargs)


    class LightGBMCumlEstimator(es.LightGBMEstimator):
        def _build_estimator(self, task, kwargs):
            if task == const.TASK_REGRESSION:
                est = LGBMRegressionCumlWrapper(**kwargs)
            else:
                est = LGBMClassifierCumlWrapper(**kwargs)
            return est

# XGBoost
if _FEATURE_FOR_CUML in _detected_xgb:
    class XGBClassifierCumlWrapper(es.XGBClassifierWrapper):
        # def fit(self, *args, **kwargs):
        #     super().fit(*args, **kwargs)
        #     le = self.get_label_encoder()
        #     if le is not None and hasattr(le, 'as_local'):
        #         le = le.as_local()
        #         self.set_label_encoder(le)
        #     return self

        def predict(self, *args, **kwargs):
            return CumlToolBox.call_local(super().predict, *args, input_to_local=False, **kwargs)

        def predict_proba(self, *args, **kwargs):
            return CumlToolBox.call_local(super().predict_proba, *args, input_to_local=False, **kwargs)


    class XGBRegressionCumlWrapper(es.XGBRegressorWrapper):
        def predict(self, *args, **kwargs):
            return CumlToolBox.call_local(super().predict, *args, input_to_local=False, **kwargs)

else:
    class XGBClassifierCumlWrapper(es.XGBClassifierWrapper):
        def fit(self, *args, **kwargs):
            return CumlToolBox.call_local(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return CumlToolBox.call_local(super().predict, *args, **kwargs)

        def predict_proba(self, *args, **kwargs):
            return CumlToolBox.call_local(super().predict_proba, *args, **kwargs)


    class XGBRegressionCumlWrapper(es.XGBRegressorWrapper):
        def fit(self, *args, **kwargs):
            return CumlToolBox.call_local(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return CumlToolBox.call_local(super().predict, *args, **kwargs)


class XGBoostCumlEstimator(es.XGBoostEstimator):
    def _build_estimator(self, task, kwargs):
        if task == const.TASK_REGRESSION:
            est = XGBRegressionCumlWrapper(**kwargs)
        else:
            est = XGBClassifierCumlWrapper(**kwargs)
        return est


# CatBoost
if _FEATURE_FOR_CUML in _detected_catboost:
    CatBoostCumlEstimator = es.CatBoostEstimator
else:
    class CatBoostClassifierCumlWrapper(es.CatBoostClassifierWrapper):
        def fit(self, *args, **kwargs):
            return CumlToolBox.call_local(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return CumlToolBox.call_local(super().predict, *args, **kwargs)

        def predict_proba(self, *args, **kwargs):
            return CumlToolBox.call_local(super().predict_proba, *args, **kwargs)


    class CatBoostRegressionCumlWrapper(es.CatBoostRegressionWrapper):
        def fit(self, *args, **kwargs):
            return CumlToolBox.call_local(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return CumlToolBox.call_local(super().predict, *args, **kwargs)


    class CatBoostCumlEstimator(es.CatBoostEstimator):
        def _build_estimator(self, task, kwargs):
            if task == const.TASK_REGRESSION:
                est = CatBoostRegressionCumlWrapper(**kwargs)
            else:
                est = CatBoostClassifierCumlWrapper(**kwargs)
            return est


# HistGB
class HistGBClassifierCumlWrapper(es.HistGradientBoostingClassifierWrapper):
    def fit(self, *args, **kwargs):
        return CumlToolBox.call_local(super().fit, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return CumlToolBox.call_local(super().predict, *args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return CumlToolBox.call_local(super().predict_proba, *args, **kwargs)


class HistGBRegressionCumlWrapper(es.HistGradientBoostingRegressorWrapper):
    def fit(self, *args, **kwargs):
        return CumlToolBox.call_local(super().fit, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return CumlToolBox.call_local(super().predict, *args, **kwargs)


class HistGBCumlEstimator(es.HistGBEstimator):
    def _build_estimator(self, task, kwargs):
        if task == const.TASK_REGRESSION:
            est = HistGBRegressionCumlWrapper(**kwargs)
        else:
            est = HistGBClassifierCumlWrapper(**kwargs)
        return est
