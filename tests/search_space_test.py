# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.core.ops import Choice, ModuleChoice, Real, Int, HyperSpace, HyperInput
from hypergbm.utils.column_selector import column_object
from hypergbm.common_ops import categorical_pipeline_simple, \
    numeric_pipeline_complex
from hypergbm.estimators import LightGBMEstimator, XGBoostEstimator, CatBoostEstimator
from hypergbm.transformers import DataFrameMapper


def get_space_num_cat_pipeline_complex(dataframe_mapper_default=False,
                                       lightgbm_fit_kwargs={},
                                       xgb_fit_kwargs={},
                                       catboost_fit_kwargs={}):
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = numeric_pipeline_complex()(input)
        p2 = categorical_pipeline_simple()(input)
        p3 = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                             df_out_dtype_transforms=[(column_object, 'int')])([p1, p2])

        lightgbm_init_kwargs = {
            'boosting_type': Choice(['gbdt', 'dart', 'goss']),
            'num_leaves': Choice([11, 31, 101, 301, 501]),
            'learning_rate': Real(0.001, 0.1, step=0.005),
            'n_estimators': Choice([100, 300, 500, 800]),
            'max_depth': Choice([-1, 3, 5, 9]),
            'class_weight': Choice(['balanced', None]),
            # subsample_for_bin = 200000, objective = None, class_weight = None,
            #  min_split_gain = 0., min_child_weight = 1e-3, min_child_samples = 20,
        }
        lightgbm_est = LightGBMEstimator(task='binary', fit_kwargs=lightgbm_fit_kwargs, **lightgbm_init_kwargs)
        xgb_init_kwargs = {
            'max_depth': Choice([3, 5, 9]),
            'n_estimators': Choice([100, 300, 500, 800]),
            'learning_rate': Real(0.001, 0.1, step=0.005),
            'min_child_weight': Choice([1, 5, 10]),
            'gamma': Choice([0.5, 1, 1.5, 2, 5]),
            'subsample': Choice([0.6, 0.8, 1.0]),
            'colsample_bytree': Choice([0.6, 0.8, 1.0]),
        }
        xgb_est = XGBoostEstimator(task='binary', fit_kwargs=xgb_fit_kwargs, **xgb_init_kwargs)

        catboost_init_kwargs = {
            'silent': True,
            'depth': Choice([3, 5, 9]),
            'learning_rate': Real(0.001, 0.1, step=0.005),
            'iterations': Choice([30, 50, 100]),
            'l2_leaf_reg': Choice([1, 3, 5, 7, 9]),
            # 'border_count': Choice([5, 10, 20, 32, 50, 100, 200]),
        }
        catboost_est = CatBoostEstimator(task='binary', fit_kwargs=catboost_fit_kwargs, **catboost_init_kwargs)
        or_est = ModuleChoice([lightgbm_est, xgb_est, catboost_est], name='estimator_options')(p3)
        space.set_inputs(input)
    return space
