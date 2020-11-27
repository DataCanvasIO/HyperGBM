# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypergbm.estimators import LightGBMEstimator, XGBoostEstimator, CatBoostEstimator
from hypergbm.feature_generators import CrossCategorical
from hypergbm.pipeline import DataFrameMapper, Pipeline
from hypergbm.sklearn.sklearn_ops import numeric_pipeline_simple, numeric_pipeline_complex, categorical_pipeline_simple
from hypergbm.sklearn.transformers import FeatureGenerationTransformer
from hypernets.core.ops import ModuleChoice, HyperInput
from hypernets.core.search_space import Choice, Real
from hypernets.core.search_space import HyperSpace
from tabular_toolbox.column_selector import column_object, column_exclude_datetime


def search_space_general(dataframe_mapper_default=False,
                         eval_set=None,
                         early_stopping_rounds=None,
                         lightgbm_fit_kwargs=None,
                         xgb_fit_kwargs=None,
                         catboost_fit_kwargs=None,
                         task='binary'):
    if lightgbm_fit_kwargs is None:
        lightgbm_fit_kwargs = {}
    if xgb_fit_kwargs is None:
        xgb_fit_kwargs = {}
    if catboost_fit_kwargs is None:
        catboost_fit_kwargs = {}

    if eval_set is not None:
        lightgbm_fit_kwargs['eval_set'] = eval_set
        xgb_fit_kwargs['eval_set'] = eval_set
        catboost_fit_kwargs['eval_set'] = eval_set

    if early_stopping_rounds is not None:
        lightgbm_fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
        xgb_fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
        catboost_fit_kwargs['early_stopping_rounds'] = early_stopping_rounds

    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        num_pipeline = numeric_pipeline_complex()(input)
        cat_pipeline = categorical_pipeline_simple()(input)
        union_pipeline = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                         df_out_dtype_transforms=[(column_object, 'int')])([num_pipeline, cat_pipeline])

        lightgbm_init_kwargs = {
            'boosting_type': Choice(['gbdt', 'dart', 'goss']),
            'num_leaves': Choice([3, 5]),
            'learning_rate': 0.1,
            'n_estimators': Choice([10, 30, 50]),
            'max_depth': Choice([3, 5]),
            'reg_alpha': Choice([1e-2, 0.1, 1, 100]),
            'reg_lambda': Choice([1e-2, 0.1, 1, 100]),
            # 'class_weight': 'balanced',
            # subsample_for_bin = 200000, objective = None, class_weight = None,
            #  min_split_gain = 0., min_child_weight = 1e-3, min_child_samples = 20,
        }
        lightgbm_est = LightGBMEstimator(task=task, fit_kwargs=lightgbm_fit_kwargs, **lightgbm_init_kwargs)
        xgb_init_kwargs = {
            'max_depth': Choice([3, 5]),
            'n_estimators': Choice([10, 30, 50]),
            'learning_rate': 0.1,
            'min_child_weight': Choice([1, 5, 10]),
            'gamma': Choice([0.5, 1, 1.5, 2, 5]),
            'reg_alpha': Choice([1e-2, 0.1, 1, 100]),
            'reg_lambda': Choice([1e-2, 0.1, 1, 100]),
            'subsample': Choice([0.6, 0.8, 1.0]),
            'colsample_bytree': Choice([0.6, 0.8, 1.0]),
            # 'scale_pos_weight': Int(1,5,1),
        }
        xgb_est = XGBoostEstimator(task=task, fit_kwargs=xgb_fit_kwargs, **xgb_init_kwargs)

        catboost_init_kwargs = {
            'silent': True,
            'depth': Choice([3, 5]),
            'learning_rate': Real(0.001, 0.1, step=0.005),
            'iterations': Choice([30, 50, 100]),
            'l2_leaf_reg': Choice([1, 3, 5, 7, 9]),
            # 'class_weights': [0.59,3.07],
            # 'border_count': Choice([5, 10, 20, 32, 50, 100, 200]),
        }
        catboost_est = CatBoostEstimator(task=task, fit_kwargs=catboost_fit_kwargs, **catboost_init_kwargs)

        ModuleChoice([lightgbm_est, xgb_est, catboost_est], name='estimator_options')(union_pipeline)
        space.set_inputs(input)
    return space


def search_space_feature_gen(dataframe_mapper_default=False,
                             eval_set=None,
                             early_stopping_rounds=None,
                             lightgbm_fit_kwargs=None,
                             xgb_fit_kwargs=None,
                             catboost_fit_kwargs=None,
                             task='binary'):
    if lightgbm_fit_kwargs is None:
        lightgbm_fit_kwargs = {}
    if xgb_fit_kwargs is None:
        xgb_fit_kwargs = {}
    if catboost_fit_kwargs is None:
        catboost_fit_kwargs = {}

    if eval_set is not None:
        lightgbm_fit_kwargs['eval_set'] = eval_set
        xgb_fit_kwargs['eval_set'] = eval_set
        catboost_fit_kwargs['eval_set'] = eval_set

    if early_stopping_rounds is not None:
        lightgbm_fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
        xgb_fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
        catboost_fit_kwargs['early_stopping_rounds'] = early_stopping_rounds

    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        num_pipeline = numeric_pipeline_complex()
        cat_pipeline = categorical_pipeline_simple()
        union_pipeline = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                         df_out_dtype_transforms=[(column_object, 'int')])([num_pipeline, cat_pipeline])

        lightgbm_init_kwargs = {
            'boosting_type': Choice(['gbdt', 'dart', 'goss']),
            'num_leaves': Choice([3, 5]),
            'learning_rate': 0.1,
            'n_estimators': Choice([10, 30, 50]),
            'max_depth': Choice([3, 5]),
            'reg_alpha': Choice([1e-2, 0.1, 1, 100]),
            'reg_lambda': Choice([1e-2, 0.1, 1, 100]),
            # 'class_weight': 'balanced',
            # subsample_for_bin = 200000, objective = None, class_weight = None,
            #  min_split_gain = 0., min_child_weight = 1e-3, min_child_samples = 20,
        }
        lightgbm_est = LightGBMEstimator(task=task, fit_kwargs=lightgbm_fit_kwargs, **lightgbm_init_kwargs)
        xgb_init_kwargs = {
            'max_depth': Choice([3, 5]),
            'n_estimators': Choice([10, 30, 50]),
            'learning_rate': 0.1,
            'min_child_weight': Choice([1, 5, 10]),
            'gamma': Choice([0.5, 1, 1.5, 2, 5]),
            'reg_alpha': Choice([1e-2, 0.1, 1, 100]),
            'reg_lambda': Choice([1e-2, 0.1, 1, 100]),
            'subsample': Choice([0.6, 0.8, 1.0]),
            'colsample_bytree': Choice([0.6, 0.8, 1.0]),
            # 'scale_pos_weight': Int(1,5,1),
        }
        xgb_est = XGBoostEstimator(task=task, fit_kwargs=xgb_fit_kwargs, **xgb_init_kwargs)

        catboost_init_kwargs = {
            'silent': True,
            'depth': Choice([3, 5]),
            'learning_rate': Real(0.001, 0.1, step=0.005),
            'iterations': Choice([30, 50, 100]),
            'l2_leaf_reg': Choice([1, 3, 5, 7, 9]),
            # 'class_weights': [0.59,3.07],
            # 'border_count': Choice([5, 10, 20, 32, 50, 100, 200]),
        }
        catboost_est = CatBoostEstimator(task=task, fit_kwargs=catboost_fit_kwargs, **catboost_init_kwargs)
        cross_cat = CrossCategorical()
        feature_gen = FeatureGenerationTransformer(task=task, trans_primitives=[cross_cat, 'divide_numeric'])
        full_pipeline = Pipeline([feature_gen, union_pipeline],
                                 name=f'feature_gen_and_preprocess',
                                 columns=column_exclude_datetime)(input)
        # full_dfm = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True)(full_pipeline)
        ModuleChoice([lightgbm_est, xgb_est, catboost_est], name='estimator_options')(full_pipeline)
        space.set_inputs(input)
    return space


def search_space_one_trail(dataframe_mapper_default=False,
                           eval_set=None,
                           early_stopping_rounds=None,
                           lightgbm_fit_kwargs=None):
    if lightgbm_fit_kwargs is None:
        lightgbm_fit_kwargs = {}

    if eval_set is not None:
        lightgbm_fit_kwargs['eval_set'] = eval_set

    if early_stopping_rounds is not None:
        lightgbm_fit_kwargs['early_stopping_rounds'] = early_stopping_rounds

    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        num_pipeline = numeric_pipeline_simple()(input)
        cat_pipeline = categorical_pipeline_simple()(input)
        union_pipeline = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                         df_out_dtype_transforms=[(column_object, 'int')])([num_pipeline, cat_pipeline])

        lightgbm_init_kwargs = {
            'boosting_type': 'gbdt',
            'num_leaves': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 5,
            'class_weight': 'balanced',
        }
        lightgbm_est = LightGBMEstimator(task='binary', fit_kwargs=lightgbm_fit_kwargs, **lightgbm_init_kwargs)
        lightgbm_est(union_pipeline)
        space.set_inputs(input)
    return space


def search_space_compact():
    raise NotImplementedError


def search_space_universal():
    raise NotImplementedError
