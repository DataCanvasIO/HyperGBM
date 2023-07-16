# -*- coding:utf-8 -*-
"""

"""
import contextlib
import inspect
import os

import catboost
import lightgbm
import numpy as np
import pandas as pd
import xgboost

from hypergbm.utils.detect_estimator import detect_with_process
from hypernets.core.search_space import ModuleSpace
from hypernets.discriminators import UnPromisingTrial
from hypernets.tabular import get_tool_box
from hypernets.tabular.cache import cache
from hypernets.tabular.cfg import TabularCfg as tcfg
from hypernets.tabular.column_selector import column_object_category_bool, column_zero_or_positive_int32
from hypernets.utils import const, logging, to_repr, Version
from .gbm_callbacks import LightGBMDiscriminationCallback, CatboostDiscriminationCallback
from .gbm_callbacks import XGBoostDiscriminationCallback, XGBoostFileMonitorCallback

try:
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
except ImportError:
    from sklearn.experimental.enable_hist_gradient_boosting import \
        HistGradientBoostingRegressor, HistGradientBoostingClassifier

logger = logging.get_logger(__name__)

_detected_lgbm_gpu = None


@cache()
def _tb_detect_estimator(name_or_cls, task, *,
                         init_kwargs=None, fit_kwargs=None, n_samples=100, n_features=5):
    r = get_tool_box(pd.DataFrame).estimator_detector(name_or_cls, task,
                                                      init_kwargs=init_kwargs,
                                                      fit_kwargs=fit_kwargs,
                                                      n_samples=n_samples,
                                                      n_features=n_features)()

    logger.info(f'detect_estimator {name_or_cls} as {r}')
    return r


def detect_lgbm_gpu():
    global _detected_lgbm_gpu

    if _detected_lgbm_gpu is None:
        # tb = get_tool_box(pd.DataFrame)
        # detected = tb.estimator_detector('lightgbm.LGBMClassifier', const.TASK_BINARY,
        #                                  # init_kwargs={'device': 'GPU', 'verbose': -1},
        #                                  init_kwargs={'device': 'GPU'},
        #                                  fit_kwargs={})()
        # logger.info(f'detect_estimator lightgbm.LGBMClassifier as {detected}')
        # _detected_lgbm_gpu = detected
        _detected_lgbm_gpu = detect_with_process('lightgbm.LGBMClassifier', const.TASK_BINARY,
                                                 init_kwargs={'device': 'GPU', 'verbose': -1},
                                                 fit_kwargs={})

    return 'fitted' in _detected_lgbm_gpu


def get_categorical_features(X):
    cat_cols = column_object_category_bool(X)
    cat_cols += column_zero_or_positive_int32(X)
    return cat_cols


def _default_early_stopping_rounds(estimator):
    if hasattr(estimator, 'n_estimators'):
        n_estimators = getattr(estimator, 'n_estimators', None)
    elif hasattr(estimator, 'get_params'):
        n_estimators = estimator.get_params().get('n_estimators', None)
    else:
        n_estimators = None

    if isinstance(n_estimators, int) and n_estimators > 10:
        return max(5, n_estimators // 20)
    else:
        return None


def _build_estimator_with_njobs(est_cls, kwargs, n_jobs_key='n_jobs'):
    assert isinstance(kwargs, dict)

    n_jobs = tcfg.joblib_njobs
    if n_jobs_key not in kwargs.keys() \
            and n_jobs is not None and n_jobs > 0 \
            and os.environ.get('OMP_NUM_THREADS', '-1') != str(n_jobs):
        kwargs[n_jobs_key] = n_jobs

    est = est_cls(**kwargs)
    return est


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


class LabelEncoderMixin:
    label_encoder_attr_ = 'y_encoder_'

    def make_label_encoder(self, y):
        tb = get_tool_box(y)
        le_cls = tb.transformers['LabelEncoder']
        le = le_cls()
        le.fit(y)
        le.classes_, = tb.to_local(le.classes_)
        return le

    def set_label_encoder(self, le):
        aname = self.label_encoder_attr_
        if le is not None:
            setattr(self, aname, le)
        elif hasattr(self, aname):
            delattr(self, aname)

    def get_label_encoder(self):
        return getattr(self, self.label_encoder_attr_, None)

    def encode_label(self, y):
        le = self.get_label_encoder()
        if le is not None:
            y = le.transform(y)
        return y

    def encode_fit_kwargs(self, kwargs):
        le = self.get_label_encoder()
        eval_set = kwargs.get('eval_set', None)
        if le is not None and eval_set is not None:
            eval_set = [(ex, le.transform(ey)) for ex, ey in eval_set]
            kwargs['eval_set'] = eval_set
        return kwargs

    def decode_label(self, y):
        le = self.get_label_encoder()
        if le is not None:
            y = le.inverse_transform(y)
        return y

    def fit_with_encoder(self, fn_fit, X, y, kwargs):
        if str(y.dtype).find('int') >= 0:
            uniques = get_tool_box(y).unique(y)
            enable_encoder = not (min(uniques) == 0 and max(uniques) == len(uniques) - 1)
        else:
            enable_encoder = True

        if enable_encoder:
            le = self.make_label_encoder(y)
            self.set_label_encoder(le)
            y = self.encode_label(y)
            kwargs = self.encode_fit_kwargs(kwargs)

        with self.suppress_label_encoder():
            return fn_fit(X, y, **kwargs)

    def predict_with_encoder(self, fn_predict, X, kwargs=None):
        if kwargs is None:
            kwargs = {}

        with self.suppress_label_encoder():
            pred = fn_predict(X, **kwargs)

        pred = self.decode_label(pred)
        return pred

    @contextlib.contextmanager
    def suppress_label_encoder(self):
        le = self.get_label_encoder()
        if le is not None:
            self.set_label_encoder(None)

        try:
            yield
        # except Exception as ex:
        #      raise
        finally:
            if le is not None:
                self.set_label_encoder(le)


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


# see: https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric
LGBM_REWARD2METRIC = {
    # 'auc': 'auc',
    'auc': 'logloss',
    'accuracy': 'logloss',
    'recall': 'logloss',
    'precision': 'logloss',
    'f1': 'logloss',
    'logloss': 'logloss',
    'mse': 'mse',
    'mae': 'mae',
    'msle': None,
    'rmse': 'rmse',
    'rootmeansquarederror': 'rmse',
    'root_mean_squared_error': 'rmse',
    'r2': None
}


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
        # task = self.__dict__.get('task')
        reward_metric = kwargs.pop('eval_reward_metric', None)

        if reward_metric is not None and kwargs.get('eval_metric') is None:
            eval_metric = LGBM_REWARD2METRIC.get(reward_metric)
            #: lightgbm will fix logloss
            # if eval_metric == 'binary_logloss' and task == const.TASK_MULTICLASS:
            #     eval_metric = 'multi_logloss'
            if eval_metric is not None:
                kwargs['eval_metric'] = eval_metric

        if not kwargs.__contains__('categorical_feature'):
            cat_cols = get_categorical_features(X)
            kwargs['categorical_feature'] = cat_cols if len(cat_cols) > 0 else None
        if kwargs.get('early_stopping_rounds') is None and kwargs.get('eval_set') is not None:
            kwargs['early_stopping_rounds'] = _default_early_stopping_rounds(self)

        lgbm_fit_args = inspect.signature(lightgbm.LGBMClassifier.fit).parameters.keys()
        if 'verbose' in kwargs.keys() and 'verbose' not in lgbm_fit_args:
            verbosity = kwargs.pop('verbose')
            if verbosity == 0:
                verbosity = -1
            self.set_params(verbosity=verbosity)
        if 'early_stopping_rounds' in kwargs.keys() and 'early_stopping_rounds' not in lgbm_fit_args:
            self.set_params(early_stopping_rounds=kwargs.pop('early_stopping_rounds'))

        self.feature_names_in_ = X.columns.tolist()
        return kwargs

    def prepare_predict_X(self, X):
        feature_names = self.feature_names_in_
        if feature_names != X.columns.tolist():
            X = X[feature_names]
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
            lgbm = _build_estimator_with_njobs(LGBMRegressorWrapper, kwargs, n_jobs_key='n_jobs')
        else:
            lgbm = _build_estimator_with_njobs(LGBMClassifierWrapper, kwargs, n_jobs_key='n_jobs')
        return lgbm


# see: https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst
XGB_REWARD2METRIC = {
    'auc': 'auc',
    'accuracy': 'logloss',
    'recall': 'logloss',
    'precision': 'logloss',
    'f1': 'logloss',
    'logloss': 'logloss',
    'mse': 'rmse',
    'mae': 'mae',
    'msle': 'rmsle',
    'rmse': 'rmse',
    'rootmeansquarederror': 'rmse',
    'root_mean_squared_error': 'rmse',
    'r2': None
}


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
        n_estimator = self.get_params().get('n_estimators', 0)
        callback = XGBoostDiscriminationCallback(
            discriminator=discriminator, group_id=self.group_id, n_estimator=n_estimator)
        return callback

    def build_filemonitor_callback(self, filepath):
        if filepath is None:
            return None

        callback = XGBoostFileMonitorCallback(filepath)
        return callback

    def prepare_fit_kwargs(self, X, y, kwargs):
        task = self.__dict__.get('task')
        reward_metric = kwargs.pop('eval_reward_metric', None)

        if reward_metric is not None and kwargs.get('eval_metric') is None:
            eval_metric = XGB_REWARD2METRIC.get(reward_metric)
            if eval_metric == 'logloss' and task == const.TASK_MULTICLASS:
                eval_metric = 'mlogloss'
            if eval_metric is not None:
                kwargs['eval_metric'] = eval_metric

        if kwargs.get('early_stopping_rounds') is None and kwargs.get('eval_set') is not None:
            kwargs['early_stopping_rounds'] = _default_early_stopping_rounds(self)

        if Version(xgboost.__version__) < Version('1.6'):
            self.feature_names_in_ = X.columns.tolist()

        return kwargs

    def prepare_predict_X(self, X):
        feature_names = list(self.feature_names_in_)
        if feature_names != X.columns.tolist():
            X = X[feature_names]
        return X


class XGBClassifierWrapper(xgboost.XGBClassifier, XGBEstimatorMixin, LabelEncoderMixin):
    def fit(self, X, y, **kwargs):
        kwargs = self.prepare_fit_kwargs(X, y, kwargs)
        return self.fit_with_encoder(super().fit, X, y, kwargs)

    def predict(self, X, **kwargs):
        X = self.prepare_predict_X(X)
        return self.predict_with_encoder(super().predict, X, kwargs)

    def predict_proba(self, X, **kwargs):
        X = self.prepare_predict_X(X)
        return super().predict_proba(X, **kwargs)

    def __getattribute__(self, name):
        if name == 'classes_':
            le = self.get_label_encoder()
            if le is not None:
                return le.classes_

        return super().__getattribute__(name)


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

        kwargs['use_label_encoder'] = False  # xgboost deprecate: use_label_encoder=True
        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, kwargs):
        if task == const.TASK_REGRESSION:
            xgb = _build_estimator_with_njobs(XGBRegressorWrapper, kwargs, n_jobs_key='n_jobs')
        else:
            xgb = _build_estimator_with_njobs(XGBClassifierWrapper, kwargs, n_jobs_key='n_jobs')
        xgb.__dict__['task'] = task
        return xgb


# see: https://catboost.ai/en/docs/references/custom-metric__supported-metrics
CATBOOST_REWARD2METRIC = {
    'auc': 'AUC',
    'accuracy': 'Accuracy',
    'recall': 'Recall',
    'precision': 'Precision',
    'f1': 'F1',
    'logloss': 'Logloss',
    'mse': 'RMSE',
    'mae': 'MAE',
    'msle': 'MSLE',
    'rmse': 'RMSE',
    'rootmeansquarederror': 'RMSE',
    'root_mean_squared_error': 'RMSE',
    'r2': 'R2'
}


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

        task_type = self.get_params(deep=False).get('task_type', None)

        if str(task_type).upper() == 'GPU':
            return None  # User defined loss functions, metrics and callbacks are not supported for GPU
        elif Version(catboost.__version__) >= Version('0.26'):
            callback = CatboostDiscriminationCallback(discriminator=discriminator, group_id=self.group_id)
            self.discriminator_callback = callback
            return callback
        else:
            logger.warn('Please upgrade `Catboost` to a version above 0.26 to support pruning.')
            return None

    def prepare_fit_kwargs(self, X, y, kwargs):
        task = self.__dict__.get('task')
        reward_metric = kwargs.pop('eval_reward_metric', None)

        if reward_metric is not None and kwargs.get('eval_metric') is None:
            eval_metric = CATBOOST_REWARD2METRIC.get(reward_metric)
            if task == const.TASK_MULTICLASS:
                if eval_metric == 'Logloss':
                    eval_metric = 'MultiLogloss'
                elif eval_metric == 'F1':
                    eval_metric = 'TotalF1'
                elif eval_metric in ['Recall', 'Precision']:
                    eval_metric = None

            if eval_metric is not None:
                self.set_params(eval_metric=eval_metric)

        if not kwargs.__contains__('cat_features'):
            cat_cols = get_categorical_features(X)
            kwargs['cat_features'] = cat_cols
        if kwargs.get('early_stopping_rounds') is None and kwargs.get('eval_set') is not None:
            kwargs['early_stopping_rounds'] = _default_early_stopping_rounds(self)
        return kwargs

    def prepare_predict_X(self, X):
        feature_names = self.feature_names_
        if isinstance(feature_names, list) and feature_names != X.columns.to_list():
            X = X[self.feature_names_]
        return X

    @property
    def feature_names_in_(self):
        return self.feature_names_


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

    def __repr__(self):
        return to_repr(self, excludes=['feature_names_in_'])


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

    def __repr__(self):
        return to_repr(self, excludes=['feature_names_in_'])


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
            cat = _build_estimator_with_njobs(CatBoostRegressionWrapper, kwargs, n_jobs_key='thread_count')
        else:
            cat = _build_estimator_with_njobs(CatBoostClassifierWrapper, kwargs, n_jobs_key='thread_count')
        cat.__dict__['task'] = task
        return cat
