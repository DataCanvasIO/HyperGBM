# -*- coding:utf-8 -*-
"""

"""
import inspect
import lightgbm
import xgboost

from hypernets.tabular.dask_ex import DaskToolBox
from hypernets.utils import const, logging, is_os_linux
from ..estimators import CatBoostClassifierWrapper, CatBoostRegressionWrapper, CatBoostEstimator
from ..estimators import HistGradientBoostingClassifierWrapper, HistGradientBoostingRegressorWrapper, HistGBEstimator
from ..estimators import LGBMEstimatorMixin, LGBMClassifierWrapper, LGBMRegressorWrapper, LightGBMEstimator
from ..estimators import LabelEncoderMixin
from ..estimators import XGBEstimatorMixin, XGBClassifierWrapper, XGBRegressorWrapper, XGBoostEstimator

logger = logging.get_logger(__name__)

lgbm_dask_distributed = hasattr(lightgbm, 'dask') and is_os_linux
xgb_dask_distributed = hasattr(xgboost, 'dask') and is_os_linux
catboost_dask_distributed = False
histgb_dask_distributed = False

if lgbm_dask_distributed:
    class LGBMEstimatorDaskMixin(LGBMEstimatorMixin):
        def prepare_fit_kwargs(self, X, y, kwargs):
            kwargs.pop('eval_reward_metric', None)

            # if self.boosting_type != 'dart':
            #     if kwargs.get('early_stopping_rounds') is None and kwargs.get('eval_set') is not None:
            #         kwargs['early_stopping_rounds'] = _default_early_stopping_rounds(self)

            # lightgbm.dask does not support early_stopping_rounds
            if 'early_stopping_rounds' in kwargs.keys():
                kwargs.pop('early_stopping_rounds')
            self.feature_names_in_ = X.columns.tolist()
            return kwargs


    class LGBMClassifierDaskWrapper(lightgbm.DaskLGBMClassifier, LGBMEstimatorDaskMixin):
        def fit(self, X, y, sample_weight=None, **kwargs):
            kwargs = self.prepare_fit_kwargs(X, y, kwargs)
            if 'verbose' in kwargs.keys():
                arg_names = inspect.signature(super().fit).parameters.keys()
                if 'verbose' not in arg_names:
                    kwargs.pop('verbose')
            super(LGBMClassifierDaskWrapper, self).fit(X, y, sample_weight=sample_weight, **kwargs)

        def predict(self, X, **kwargs):
            X = self.prepare_predict_X(X)
            return super().predict(X, **kwargs)

        def predict_proba(self, X, **kwargs):
            X = self.prepare_predict_X(X)
            return super().predict_proba(X, **kwargs)


    class LGBMRegressorDaskWrapper(lightgbm.DaskLGBMRegressor, LGBMEstimatorDaskMixin):
        def fit(self, X, y, sample_weight=None, **kwargs):
            kwargs = self.prepare_fit_kwargs(X, y, kwargs)
            if 'verbose' in kwargs.keys():
                arg_names = inspect.signature(super().fit).parameters.keys()
                if 'verbose' not in arg_names:
                    kwargs.pop('verbose')
            super().fit(X, y, sample_weight=sample_weight, **kwargs)

        def predict(self, X, **kwargs):
            X = self.prepare_predict_X(X)
            return super().predict(X, **kwargs)

else:
    class LGBMClassifierDaskWrapper(LGBMClassifierWrapper):
        def fit(self, *args, **kwargs):
            return DaskToolBox.compute_and_call(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return DaskToolBox.compute_and_call(super().predict, *args, **kwargs)

        def predict_proba(self, *args, **kwargs):
            return DaskToolBox.compute_and_call(super().predict_proba, *args, **kwargs)


    class LGBMRegressorDaskWrapper(LGBMRegressorWrapper):
        def fit(self, *args, **kwargs):
            return DaskToolBox.compute_and_call(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return DaskToolBox.compute_and_call(super().predict, *args, **kwargs)


class LightGBMDaskEstimator(LightGBMEstimator):
    def _build_estimator(self, task, kwargs):
        if task == const.TASK_REGRESSION:
            lgbm = LGBMRegressorDaskWrapper(**kwargs)
        else:
            lgbm = LGBMClassifierDaskWrapper(**kwargs)
        return lgbm


if xgb_dask_distributed:
    class XGBClassifierDaskWrapper(xgboost.dask.DaskXGBClassifier, XGBEstimatorMixin, LabelEncoderMixin):
        def fit(self, X, y=None, **kwargs):
            kwargs = self.prepare_fit_kwargs(X, y, kwargs)
            return self.fit_with_encoder(super().fit, X, y, kwargs)

        def predict_proba(self, X, ntree_limit=None, **kwargs):
            X = self.prepare_predict_X(X)
            arg_names = inspect.signature(super().predict_proba).parameters.keys()
            if 'ntree_limit' in arg_names:
                proba = super().predict_proba(X, ntree_limit=ntree_limit)
            else:
                proba = super().predict_proba(X)

            if self.n_classes_ == 2:
                proba = DaskToolBox.fix_binary_predict_proba_result(proba)

            return proba

        def predict(self, X, **kwargs):
            X = self.prepare_predict_X(X)
            return self.predict_with_encoder(super().predict, X, kwargs)

        def __getattribute__(self, name):
            if name == 'classes_':
                le = self.get_label_encoder()
                if le is not None:
                    return le.classes_

            return super().__getattribute__(name)


    class XGBRegressorDaskWrapper(xgboost.dask.DaskXGBRegressor, XGBEstimatorMixin):
        def fit(self, X, y=None, **kwargs):
            kwargs = self.prepare_fit_kwargs(X, y, kwargs)
            return super().fit(X, y, **kwargs)

        def predict(self, X, **kwargs):
            X = self.prepare_predict_X(X)
            return super(XGBRegressorDaskWrapper, self).predict(X)

else:
    class XGBClassifierDaskWrapper(XGBClassifierWrapper):
        def fit(self, *args, **kwargs):
            return DaskToolBox.compute_and_call(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return DaskToolBox.compute_and_call(super().predict, *args, **kwargs)

        def predict_proba(self, *args, **kwargs):
            return DaskToolBox.compute_and_call(super().predict_proba, *args, **kwargs)


    class XGBRegressorDaskWrapper(XGBRegressorWrapper):
        def fit(self, *args, **kwargs):
            return DaskToolBox.compute_and_call(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return DaskToolBox.compute_and_call(super().predict, *args, **kwargs)


class XGBoostDaskEstimator(XGBoostEstimator):
    def _build_estimator(self, task, kwargs):
        if 'task' in kwargs:
            kwargs.pop('task')

        if task == const.TASK_REGRESSION:
            xgb = XGBRegressorDaskWrapper(**kwargs)
        else:
            xgb = XGBClassifierDaskWrapper(**kwargs)
        xgb.__dict__['task'] = task
        return xgb


class CatBoostClassifierDaskWrapper(CatBoostClassifierWrapper):
    def fit(self, *args, **kwargs):
        return DaskToolBox.compute_and_call(super().fit, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return DaskToolBox.compute_and_call(super().predict, *args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return DaskToolBox.compute_and_call(super().predict_proba, *args, **kwargs)


class CatBoostRegressionDaskWrapper(CatBoostRegressionWrapper):
    def fit(self, *args, **kwargs):
        return DaskToolBox.compute_and_call(super().fit, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return DaskToolBox.compute_and_call(super().predict, *args, **kwargs)


class CatBoostDaskEstimator(CatBoostEstimator):
    def _build_estimator(self, task, kwargs):
        if task == const.TASK_REGRESSION:
            cat = CatBoostRegressionDaskWrapper(**kwargs)
        else:
            cat = CatBoostClassifierDaskWrapper(**kwargs)
        return cat


class HistGradientBoostingClassifierDaskWrapper(HistGradientBoostingClassifierWrapper):
    def fit(self, *args, **kwargs):
        return DaskToolBox.compute_and_call(super().fit, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return DaskToolBox.compute_and_call(super().predict, *args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return DaskToolBox.compute_and_call(super().predict_proba, *args, **kwargs)


class HistGradientBoostingRegressorDaskWrapper(HistGradientBoostingRegressorWrapper):
    def fit(self, *args, **kwargs):
        return DaskToolBox.compute_and_call(super().fit, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return DaskToolBox.compute_and_call(super().predict, *args, **kwargs)


class HistGBDaskEstimator(HistGBEstimator):
    def _build_estimator(self, task, kwargs):
        if task == const.TASK_REGRESSION:
            hgb = HistGradientBoostingRegressorDaskWrapper(**kwargs)
        else:
            hgb = HistGradientBoostingClassifierDaskWrapper(**kwargs)
        return hgb
