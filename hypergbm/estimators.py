# -*- coding:utf-8 -*-
"""

"""
import sys
from distutils.version import LooseVersion

import catboost
import lightgbm
import numpy as np
import xgboost
from sklearn.experimental.enable_hist_gradient_boosting import \
    HistGradientBoostingRegressor, HistGradientBoostingClassifier

from hypernets.core.search_space import ModuleSpace
from hypernets.discriminators import UnPromisingTrial
from hypernets.tabular import dask_ex as dex
from hypernets.tabular.column_selector import column_object_category_bool, column_zero_or_positive_int32
from hypernets.utils import const, logging
from .gbm_callbacks import LightGBMDiscriminationCallback, XGBoostDiscriminationCallback, CatboostDiscriminationCallback

logger = logging.get_logger(__name__)
_is_windows = sys.platform.find('win') >= 0


def get_categorical_features(X):
    cat_cols = column_object_category_bool(X)
    cat_cols += column_zero_or_positive_int32(X)
    return cat_cols


def _default_early_stopping_rounds(estimator):
    n_estimators = getattr(estimator, 'n_estimators', None)
    if isinstance(n_estimators, int):
        return max(5, n_estimators // 20)
    else:
        # return None
        return 200


class HyperEstimator(ModuleSpace):
    def __init__(self, fit_kwargs, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.fit_kwargs = fit_kwargs
        self.estimator = None
        self.class_balancing = False

    def _build_estimator(self, task, kwargs):
        raise NotImplementedError

    def build_estimator(self, task):
        pv = self.param_values
        if pv.__contains__('class_balancing'):
            self.class_balancing = pv.pop('class_balancing')

        self.estimator = self._build_estimator(task, pv)

    def _compile(self):
        pass
        # pv = self.param_values
        # self.estimator = self._build_estimator(pv)

    def _forward(self, inputs):
        return self.estimator

    def _on_params_ready(self):
        pass


class HyperEstimatorMixin:
    # @property
    # def best_n_estimators(self):
    #     return NotImplementedError()
    #
    # @property
    # def iteration_scores(self):
    #     raise NotImplementedError()

    def build_discriminator_callback(self, discriminator):
        return None


class HistGradientBoostingClassifierWrapper(HistGradientBoostingClassifier, HyperEstimatorMixin):
    def fit(self, X, y, sample_weight=None, **kwargs):
        return super(HistGradientBoostingClassifierWrapper, self).fit(X, y, sample_weight)


class HistGradientBoostingRegressorWrapper(HistGradientBoostingRegressor, HyperEstimatorMixin):
    def fit(self, X, y, sample_weight=None, **kwargs):
        return super(HistGradientBoostingRegressorWrapper, self).fit(X, y, sample_weight)


class HistGBEstimator(HyperEstimator):
    def __init__(self, fit_kwargs, loss='least_squares', learning_rate=0.1,
                 max_leaf_nodes=31, max_depth=None,
                 min_samples_leaf=20, l2_regularization=0., max_bins=255,
                 space=None, name=None, **kwargs):
        if loss is not None and loss != 'least_squares':
            kwargs['loss'] = loss
        if learning_rate is not None and learning_rate != 0.1:
            kwargs['learning_rate'] = learning_rate
        if min_samples_leaf is not None and min_samples_leaf != 20:
            kwargs['min_samples_leaf'] = min_samples_leaf
        if max_depth is not None:
            kwargs['max_depth'] = max_depth
        if max_leaf_nodes is not None and max_leaf_nodes != 31:
            kwargs['max_leaf_nodes'] = max_leaf_nodes
        if max_bins is not None and max_bins != 255:
            kwargs['max_bins'] = max_bins
        if l2_regularization is not None and l2_regularization != 0.:
            kwargs['l2_regularization'] = l2_regularization

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, kwargs):
        if task == const.TASK_REGRESSION:
            hgboost = HistGradientBoostingRegressorWrapper(**kwargs)
        else:
            hgboost = HistGradientBoostingClassifierWrapper(**kwargs)
        return hgboost


class LGBMEstimatorMixin:
    @property
    def best_n_estimators(self):
        if self.best_iteration_ is None or self.best_iteration_ <= 0:
            return self.n_estimators
        else:
            return self.best_iteration_

    @property
    def iteration_scores(self):
        scores = []
        if self.evals_result_:
            valid = self.evals_result_.get('valid_0')
            if valid:
                scores = list(valid.values())[0]
        return scores

    def build_discriminator_callback(self, discriminator):
        if discriminator is None:
            return None
        callback = LightGBMDiscriminationCallback(discriminator=discriminator, group_id=self.group_id)
        return callback

    def prepare_fit_kwargs(self, X, y, kwargs):
        if not kwargs.__contains__('categorical_feature'):
            cat_cols = get_categorical_features(X)
            kwargs['categorical_feature'] = cat_cols
        if kwargs.get('early_stopping_rounds') is None and kwargs.get('eval_set') is not None:
            kwargs['early_stopping_rounds'] = _default_early_stopping_rounds(self)
        return kwargs

    def prepare_predict_X(self, X):
        try:
            if hasattr(self, 'feature_name_'):
                X = X[self.feature_name_]
        except:
            pass
        return X


class LGBMClassifierWrapper(lightgbm.LGBMClassifier, LGBMEstimatorMixin):
    def fit(self, X, y, sample_weight=None, **kwargs):
        kwargs = self.prepare_fit_kwargs(X, y, kwargs)
        super(LGBMClassifierWrapper, self).fit(X, y, sample_weight=sample_weight, **kwargs)

    def predict(self, X, raw_score=False, start_iteration=0, num_iteration=None,
                pred_leaf=False, pred_contrib=False, **kwargs):
        X = self.prepare_predict_X(X)
        return super().predict(X, raw_score=raw_score, start_iteration=start_iteration,
                               num_iteration=num_iteration, pred_leaf=pred_leaf,
                               pred_contrib=pred_contrib, **kwargs)

    def predict_proba(self, X, raw_score=False, start_iteration=0, num_iteration=None,
                      pred_leaf=False, pred_contrib=False, **kwargs):
        X = self.prepare_predict_X(X)
        return super().predict_proba(X, raw_score=raw_score, start_iteration=start_iteration,
                                     num_iteration=num_iteration, pred_leaf=pred_leaf,
                                     pred_contrib=pred_contrib, **kwargs)


class LGBMRegressorWrapper(lightgbm.LGBMRegressor, LGBMEstimatorMixin):
    def fit(self, X, y, sample_weight=None, **kwargs):
        kwargs = self.prepare_fit_kwargs(X, y, kwargs)
        super().fit(X, y, sample_weight=sample_weight, **kwargs)

    def predict(self, X, raw_score=False, start_iteration=0, num_iteration=None,
                pred_leaf=False, pred_contrib=False, **kwargs):
        X = self.prepare_predict_X(X)
        return super().predict(X, raw_score=raw_score, start_iteration=start_iteration,
                               num_iteration=num_iteration, pred_leaf=pred_leaf,
                               pred_contrib=pred_contrib, **kwargs)


class LightGBMEstimator(HyperEstimator):
    def __init__(self, fit_kwargs, boosting_type='gbdt', num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=100,
                 subsample_for_bin=200000, objective=None, class_weight=None,
                 min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
                 subsample=1., subsample_freq=0, colsample_bytree=1.,
                 reg_alpha=0., reg_lambda=0., random_state=None,
                 n_jobs=-1, silent=True, importance_type='split', space=None, name=None, **kwargs):

        if boosting_type is not None and boosting_type != 'gbdt':
            kwargs['boosting_type'] = boosting_type
        if num_leaves is not None and num_leaves != 31:
            kwargs['num_leaves'] = num_leaves
        if max_depth is not None and max_depth != -1:
            kwargs['max_depth'] = max_depth
        if learning_rate is not None and learning_rate != 0.1:
            kwargs['learning_rate'] = learning_rate
        if n_estimators is not None and n_estimators != 100:
            kwargs['n_estimators'] = n_estimators
        if subsample_for_bin is not None and subsample_for_bin != 200000:
            kwargs['subsample_for_bin'] = subsample_for_bin
        if objective is not None:
            kwargs['objective'] = objective
        if class_weight is not None:
            kwargs['class_weight'] = class_weight
        if min_split_gain is not None and min_split_gain != 0.:
            kwargs['min_split_gain'] = min_split_gain
        if min_child_weight is not None and min_child_weight != 1e-3:
            kwargs['min_child_weight'] = min_child_weight
        if min_child_samples is not None and min_child_samples != 20:
            kwargs['min_child_samples'] = min_child_samples
        if subsample is not None and subsample != 1.:
            kwargs['subsample'] = subsample
        if subsample_freq is not None and subsample_freq != 0:
            kwargs['subsample_freq'] = subsample_freq
        if colsample_bytree is not None and colsample_bytree != 1.:
            kwargs['colsample_bytree'] = colsample_bytree
        if reg_alpha is not None and reg_alpha != 0.:
            kwargs['reg_alpha'] = reg_alpha
        if reg_lambda is not None and reg_lambda != 0.:
            kwargs['reg_lambda'] = reg_lambda
        if random_state is not None:
            kwargs['random_state'] = random_state
        if n_jobs is not None and n_jobs != -1:
            kwargs['n_jobs'] = n_jobs
        if silent is not None and silent != True:
            kwargs['silent'] = silent
        if importance_type is not None and importance_type != 'split':
            kwargs['importance_type'] = importance_type

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, kwargs):
        if task == const.TASK_REGRESSION:
            lgbm = LGBMRegressorWrapper(**kwargs)
        else:
            lgbm = LGBMClassifierWrapper(**kwargs)
        return lgbm


lgbm_dask_distributed = hasattr(lightgbm, 'dask')
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
            return dex.compute_and_call(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return dex.compute_and_call(super().predict, *args, **kwargs)

        def predict_proba(self, *args, **kwargs):
            return dex.compute_and_call(super().predict_proba, *args, **kwargs)


    class LGBMRegressorDaskWrapper(LGBMRegressorWrapper):
        def fit(self, *args, **kwargs):
            return dex.compute_and_call(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return dex.compute_and_call(super().predict, *args, **kwargs)


class LightGBMDaskEstimator(LightGBMEstimator):
    def _build_estimator(self, task, kwargs):
        if task == const.TASK_REGRESSION:
            lgbm = LGBMRegressorDaskWrapper(**kwargs)
        else:
            lgbm = LGBMClassifierDaskWrapper(**kwargs)
        return lgbm


class XGBEstimatorMixin:
    @property
    def best_n_estimators(self):
        booster = self.get_booster()
        if booster is not None:
            return booster.best_ntree_limit
        else:
            return None

    @property
    def iteration_scores(self):
        scores = []
        if hasattr(self, 'evals_result_'):
            valid = self.evals_result_.get('validation_0')
            if valid:
                scores = list(valid.values())[0]
        return scores

    def build_discriminator_callback(self, discriminator):
        if discriminator is None:
            return None
        callback = XGBoostDiscriminationCallback(discriminator=discriminator, group_id=self.group_id)
        return callback

    def prepare_fit_kwargs(self, X, y, kwargs):
        # task = self.__dict__.get('task')
        # if kwargs.get('eval_metric') is None:
        #     if task is not None and task == const.TASK_MULTICLASS:
        #         kwargs['eval_metric'] = 'mlogloss'
        #     else:
        #         kwargs['eval_metric'] = 'logloss'
        if kwargs.get('early_stopping_rounds') is None and kwargs.get('eval_set') is not None:
            kwargs['early_stopping_rounds'] = _default_early_stopping_rounds(self)
        return kwargs

    def prepare_predict_X(self, X):
        X = X[self.get_booster().feature_names]
        return X


class XGBClassifierWrapper(xgboost.XGBClassifier, XGBEstimatorMixin):
    def fit(self, X, y, **kwargs):
        kwargs = self.prepare_fit_kwargs(X, y, kwargs)
        super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        X = self.prepare_predict_X(X)
        return super().predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        X = self.prepare_predict_X(X)
        return super().predict_proba(X, **kwargs)


class XGBRegressorWrapper(xgboost.XGBRegressor, XGBEstimatorMixin):
    def fit(self, X, y, **kwargs):
        kwargs = self.prepare_fit_kwargs(X, y, kwargs)
        super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        X = self.prepare_predict_X(X)
        return super().predict(X, **kwargs)


class XGBoostEstimator(HyperEstimator):
    def __init__(self, fit_kwargs, max_depth=None, learning_rate=None, n_estimators=100,
                 verbosity=None, objective=None, booster=None,
                 tree_method=None, n_jobs=None, gamma=None,
                 min_child_weight=None, max_delta_step=None, subsample=None,
                 colsample_bytree=None, colsample_bylevel=None,
                 colsample_bynode=None, reg_alpha=None, reg_lambda=None,
                 scale_pos_weight=None, base_score=None, random_state=None,
                 missing=np.nan, num_parallel_tree=None,
                 monotone_constraints=None, interaction_constraints=None,
                 importance_type="gain", gpu_id=None,
                 validate_parameters=None, space=None, name=None, **kwargs):
        if max_depth is not None:
            kwargs['max_depth'] = max_depth
        if learning_rate is not None:
            kwargs['learning_rate'] = learning_rate
        if n_estimators is not None:
            kwargs['n_estimators'] = n_estimators
        if verbosity is not None:
            kwargs['verbosity'] = verbosity
        else:
            kwargs['verbosity'] = 0
        if objective is not None:
            kwargs['objective'] = objective
        if booster is not None:
            kwargs['booster'] = booster
        if tree_method is not None:
            kwargs['tree_method'] = tree_method
        if n_jobs is not None:
            kwargs['n_jobs'] = n_jobs
        if gamma is not None:
            kwargs['gamma'] = gamma
        if min_child_weight is not None:
            kwargs['min_child_weight'] = min_child_weight
        if max_delta_step is not None:
            kwargs['max_delta_step'] = max_delta_step
        if subsample is not None:
            kwargs['subsample'] = subsample
        if colsample_bytree is not None:
            kwargs['colsample_bytree'] = colsample_bytree
        if colsample_bylevel is not None:
            kwargs['colsample_bylevel'] = colsample_bylevel
        if colsample_bynode is not None:
            kwargs['colsample_bynode'] = colsample_bynode
        if reg_alpha is not None:
            kwargs['reg_alpha'] = reg_alpha
        if reg_lambda is not None:
            kwargs['reg_lambda'] = reg_lambda
        if scale_pos_weight is not None:
            kwargs['scale_pos_weight'] = scale_pos_weight
        if base_score is not None:
            kwargs['base_score'] = base_score
        if random_state is not None:
            kwargs['random_state'] = random_state
        if missing is not None:
            kwargs['missing'] = missing
        if num_parallel_tree is not None:
            kwargs['num_parallel_tree'] = num_parallel_tree
        if monotone_constraints is not None:
            kwargs['monotone_constraints'] = monotone_constraints
        if interaction_constraints is not None:
            kwargs['interaction_constraints'] = interaction_constraints
        if importance_type is not None and importance_type != 'gain':
            kwargs['importance_type'] = importance_type
        if gpu_id is not None:
            kwargs['gpu_id'] = gpu_id
        if validate_parameters is not None:
            kwargs['validate_parameters'] = validate_parameters

        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, kwargs):
        if task == const.TASK_REGRESSION:
            xgb = XGBRegressorWrapper(**kwargs)
        else:
            xgb = XGBClassifierWrapper(**kwargs)
        xgb.__dict__['task'] = task
        return xgb


xgb_dask_distributed = hasattr(xgboost, 'dask') and not _is_windows
if xgb_dask_distributed:
    class XGBEstimatorDaskMixin(XGBEstimatorMixin):
        def prepare_fit_kwargs(self, X, y, kwargs):
            kwargs = super().prepare_fit_kwargs(X, y, kwargs)

            task = self.__dict__.get('task')
            if task is not None and task in {const.TASK_MULTICLASS, const.TASK_BINARY}:
                if y is not None and y.dtype == np.object:
                    from dask_ml.preprocessing import LabelEncoder as DaskLabelEncoder
                    le = DaskLabelEncoder()
                    # y = le.fit_transform(y)
                    le.fit(y)

                    if dex.is_dask_object(le.classes_):
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
                proba = dex.fix_binary_predict_proba_result(proba)

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
            return dex.compute_and_call(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return dex.compute_and_call(super().predict, *args, **kwargs)

        def predict_proba(self, *args, **kwargs):
            return dex.compute_and_call(super().predict_proba, *args, **kwargs)


    class XGBRegressorDaskWrapper(XGBRegressorWrapper):
        def fit(self, *args, **kwargs):
            return dex.compute_and_call(super().fit, *args, **kwargs)

        def predict(self, *args, **kwargs):
            return dex.compute_and_call(super().predict, *args, **kwargs)


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


class CatBoostEstimatorMixin:
    @property
    def best_n_estimators(self):
        return self.best_iteration_

    @property
    def iteration_scores(self):
        scores = []
        if self.evals_result_:
            valid = self.evals_result_.get('validation')
            if valid:
                scores = list(valid.values())[0]
            else:
                learn = self.evals_result_.get('learn')
                if learn:
                    scores = list(learn.values())[0]
        return scores

    def build_discriminator_callback(self, discriminator):
        if discriminator is None:
            return None
        # if int(catboost.__version__.split('.')[1]) >= 26:
        if LooseVersion(catboost.__version__) >= LooseVersion('0.26'):
            callback = CatboostDiscriminationCallback(discriminator=discriminator, group_id=self.group_id)
            self.discriminator_callback = callback
            return callback
        else:
            logger.warn('Please upgrade `Catboost` to a version above 0.26 to support pruning.')
            return None

    def prepare_fit_kwargs(self, X, y, kwargs):
        if not kwargs.__contains__('cat_features'):
            cat_cols = get_categorical_features(X)
            kwargs['cat_features'] = cat_cols
        if kwargs.get('early_stopping_rounds') is None and kwargs.get('eval_set') is not None:
            kwargs['early_stopping_rounds'] = _default_early_stopping_rounds(self)
        return kwargs

    def prepare_predict_X(self, X):
        X = X[self.feature_names_]
        return X


class CatBoostClassifierWrapper(catboost.CatBoostClassifier, CatBoostEstimatorMixin):
    def fit(self, X, y=None, **kwargs):
        kwargs = self.prepare_fit_kwargs(X, y, kwargs)
        super().fit(X, y, **kwargs)
        discriminator_callback = self.__dict__.get('discriminator_callback')
        if discriminator_callback is not None and not discriminator_callback.is_promising_:
            raise UnPromisingTrial(f'unpromising trial:{discriminator_callback.iteration_trajectory}')

    def predict(self, X, **kwargs):
        X = self.prepare_predict_X(X)
        return super().predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        X = self.prepare_predict_X(X)
        return super().predict_proba(X, **kwargs)


class CatBoostRegressionWrapper(catboost.CatBoostRegressor, CatBoostEstimatorMixin):
    def fit(self, X, y=None, **kwargs):
        kwargs = self.prepare_fit_kwargs(X, y, kwargs)
        super().fit(X, y, **kwargs)
        discriminator_callback = self.__dict__.get('discriminator_callback')
        if discriminator_callback is not None and not discriminator_callback.is_promising_:
            raise UnPromisingTrial(f'unpromising trial:{discriminator_callback.iteration_trajectory}')

    def predict(self, X, **kwargs):
        X = self.prepare_predict_X(X)
        return super().predict(X, **kwargs)


class CatBoostEstimator(HyperEstimator):
    def __init__(self, fit_kwargs, iterations=None, learning_rate=None, depth=None,
                 l2_leaf_reg=None,
                 model_size_reg=None, rsm=None, loss_function=None, border_count=None, feature_border_type=None,
                 per_float_feature_quantization=None, input_borders=None, output_borders=None, space=None, name=None,
                 **kwargs):
        if iterations is not None:
            kwargs['iterations'] = iterations
        if learning_rate is not None:
            kwargs['learning_rate'] = learning_rate
        if depth is not None:
            kwargs['depth'] = depth
        if l2_leaf_reg is not None:
            kwargs['l2_leaf_reg'] = l2_leaf_reg
        if model_size_reg is not None:
            kwargs['model_size_reg'] = model_size_reg
        if rsm is not None:
            kwargs['rsm'] = rsm
        if loss_function is not None:
            kwargs['loss_function'] = loss_function
        if border_count is not None:
            kwargs['border_count'] = border_count
        if feature_border_type is not None:
            kwargs['feature_border_type'] = feature_border_type
        if per_float_feature_quantization is not None:
            kwargs['per_float_feature_quantization'] = per_float_feature_quantization
        if input_borders is not None:
            kwargs['input_borders'] = input_borders
        if output_borders is not None:
            kwargs['output_borders'] = output_borders
        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, kwargs):
        if task == const.TASK_REGRESSION:
            cat = CatBoostRegressionWrapper(**kwargs)
        else:
            cat = CatBoostClassifierWrapper(**kwargs)
        return cat


class CatBoostClassifierDaskWrapper(CatBoostClassifierWrapper):
    def fit(self, *args, **kwargs):
        return dex.compute_and_call(super().fit, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return dex.compute_and_call(super().predict, *args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return dex.compute_and_call(super().predict_proba, *args, **kwargs)


class CatBoostRegressionDaskWrapper(CatBoostRegressionWrapper):
    def fit(self, *args, **kwargs):
        return dex.compute_and_call(super().fit, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return dex.compute_and_call(super().predict, *args, **kwargs)


class CatBoostDaskEstimator(CatBoostEstimator):
    def _build_estimator(self, task, kwargs):
        if task == const.TASK_REGRESSION:
            cat = CatBoostRegressionDaskWrapper(**kwargs)
        else:
            cat = CatBoostClassifierDaskWrapper(**kwargs)
        return cat
