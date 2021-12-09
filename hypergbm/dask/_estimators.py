# -*- coding:utf-8 -*-
"""

"""

import lightgbm
import xgboost

from hypernets.tabular.dask_ex import DaskToolBox
from hypernets.utils import const, logging, is_os_linux
from ..estimators import CatBoostClassifierWrapper, CatBoostRegressionWrapper, CatBoostEstimator
from ..estimators import HistGradientBoostingClassifierWrapper, HistGradientBoostingRegressorWrapper, HistGBEstimator
from ..estimators import LGBMEstimatorMixin, LGBMClassifierWrapper, LGBMRegressorWrapper, LightGBMEstimator
from ..estimators import XGBEstimatorMixin, XGBClassifierWrapper, XGBRegressorWrapper, XGBoostEstimator
from ..estimators import _default_early_stopping_rounds

logger = logging.get_logger(__name__)

lgbm_dask_distributed = hasattr(lightgbm, 'dask') and is_os_linux
xgb_dask_distributed = hasattr(xgboost, 'dask') and is_os_linux
catboost_dask_distributed = False
histgb_dask_distributed = False

if lgbm_dask_distributed:
    class LGBMEstimatorDaskMixin(LGBMEstimatorMixin):
        def prepare_fit_kwargs(self, X, y, kwargs):
            if self.boosting_type != 'dart':
                if kwargs.get('early_stopping_rounds') is None and kwargs.get('eval_set') is not None:
                    kwargs['early_stopping_rounds'] = _default_early_stopping_rounds(self)
            return kwargs


    class LGBMClassifierDaskWrapper(lightgbm.DaskLGBMClassifier, LGBMEstimatorDaskMixin):
        def fit(self, X, y, sample_weight=None, **kwargs):
            kwargs = self.prepare_fit_kwargs(X, y, kwargs)
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
    class XGBEstimatorDaskMixin(XGBEstimatorMixin):
        def prepare_fit_kwargs(self, X, y, kwargs):
            kwargs = super().prepare_fit_kwargs(X, y, kwargs)

            task = self.__dict__.get('task')
            if task is not None and task in {const.TASK_MULTICLASS, const.TASK_BINARY}:
                if y is not None and str(y.dtype) == 'object':
                    from dask_ml.preprocessing import LabelEncoder as DaskLabelEncoder
                    le = DaskLabelEncoder()
                    # y = le.fit_transform(y)
                    le.fit(y)

                    if DaskToolBox.is_dask_object(le.classes_):
                        le.classes_ = le.classes_.compute()

                    eval_set = kwargs.get('eval_set')
                    if eval_set is not None:
                        eval_set = [(ex, le.transform(ey)) for ex, ey in eval_set]
                        kwargs['eval_set'] = eval_set

                    self.y_encoder_ = le

            return kwargs


    class XGBClassifierDaskWrapper(xgboost.dask.DaskXGBClassifier, XGBEstimatorDaskMixin):
        def fit(self, X, y=None, **kwargs):
            kwargs = self.prepare_fit_kwargs(X, y, kwargs)
            encoder = getattr(self, 'y_encoder_', None)
            if encoder is not None:
                y = encoder.transform(y)
            return super().fit(X, y, **kwargs)

        def predict_proba(self, X, ntree_limit=None, **kwargs):
            X = self.prepare_predict_X(X)
            proba = super().predict_proba(X, ntree_limit=ntree_limit)

            if self.n_classes_ == 2:
                proba = DaskToolBox.fix_binary_predict_proba_result(proba)

            return proba

        def predict(self, X, **kwargs):
            X = self.prepare_predict_X(X)
            pred = super().predict(X)

            encoder = getattr(self, 'y_encoder_', None)
            if encoder is not None:
                pred = encoder.inverse_transform(pred)

            return pred

        def __getattribute__(self, name):
            if name == 'classes_' and hasattr(self, 'y_encoder_'):
                encoder = getattr(self, 'y_encoder_')
                attr = encoder.classes_
            else:
                attr = super().__getattribute__(name)

            return attr


    class XGBRegressorDaskWrapper(xgboost.dask.DaskXGBRegressor, XGBEstimatorDaskMixin):
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
