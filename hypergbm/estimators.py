# -*- coding:utf-8 -*-
"""

"""

from hypernets.core.search_space import ModuleSpace, Choice
import numpy as np


class HyperEstimator(ModuleSpace):
    def __init__(self, task, fit_kwargs, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.task = task
        self.fit_kwargs = fit_kwargs
        self.estimator = None

    def _build_estimator(self, kwargs):
        raise NotImplementedError

    def _compile(self):
        pv = self.param_values
        self.estimator = self._build_estimator(pv)

    def _forward(self, inputs):
        return self.estimator

    def _on_params_ready(self):
        pass


class LightGBMEstimator(HyperEstimator):
    def __init__(self, task, fit_kwargs, boosting_type='gbdt', num_leaves=31, max_depth=-1,
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

        HyperEstimator.__init__(self, task, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, kwargs):
        import lightgbm
        if self.task == 'regression':
            lgbm = lightgbm.LGBMRegressor(**kwargs)
        else:
            lgbm = lightgbm.LGBMClassifier(**kwargs)
        return lgbm


class XGBoostEstimator(HyperEstimator):
    def __init__(self, task, fit_kwargs, max_depth=None, learning_rate=None, n_estimators=100,
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

        HyperEstimator.__init__(self, task, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, kwargs):
        import xgboost
        if self.task == 'regression':
            xgb = xgboost.XGBRegressor(**kwargs)
        else:
            xgb = xgboost.XGBClassifier(**kwargs)
        return xgb


class CatBoostEstimator(HyperEstimator):
    def __init__(self, task, fit_kwargs, iterations=None, learning_rate=None, depth=None, l2_leaf_reg=None,
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
        HyperEstimator.__init__(self, task, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, kwargs):
        import catboost
        if self.task == 'regression':
            cat = catboost.CatBoostRegressor(**kwargs)
        else:
            cat = catboost.CatBoostClassifier(**kwargs)
        return cat
