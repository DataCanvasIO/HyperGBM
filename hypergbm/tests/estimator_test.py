# -*- coding:utf-8 -*-
"""

"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from hypergbm import HyperGBMEstimator
from hypergbm.estimators import LightGBMEstimator, XGBoostEstimator
from hypergbm.pipeline import DataFrameMapper
from hypergbm.search_space import search_space_general, search_space_feature_gen
from hypergbm.sklearn.sklearn_ops import categorical_pipeline_simple, numeric_pipeline_simple, \
    categorical_pipeline_complex, numeric_pipeline_complex
from hypernets.core.ops import HyperInput, Choice, ModuleChoice
from hypernets.core.search_space import HyperSpace, Real
from tabular_toolbox.column_selector import column_object, column_exclude_datetime, column_number_exclude_timedelta, \
    column_object_category_bool
from tabular_toolbox.datasets import dsutils
from tabular_toolbox.drift_detection import general_preprocessor
from hypergbm.tests import test_output_dir


def get_space_multi_dataframemapper(default=False):
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = numeric_pipeline_simple(seq_no=0)(input)
        p2 = categorical_pipeline_simple(seq_no=0)(input)
        p3 = DataFrameMapper(default=default, input_df=True, df_out=True)([p1, p2])  # passthrough

        p4 = numeric_pipeline_simple(seq_no=1)(p3)
        p5 = categorical_pipeline_simple(seq_no=1)(p3)
        p6 = DataFrameMapper(default=default, input_df=True, df_out=True)([p4, p5])
        est = LightGBMEstimator(task='binary', fit_kwargs={})(p6)
        space.set_inputs(input)
    return space


def get_space_num_cat_pipeline_multi_complex(dataframe_mapper_default=False,
                                             lightgbm_fit_kwargs={},
                                             xgb_fit_kwargs={}):
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = numeric_pipeline_complex()(input)
        p2 = categorical_pipeline_complex()(input)
        p3 = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                             df_out_dtype_transforms=[(column_object, 'category')])([p1, p2])

        p4 = numeric_pipeline_complex(seq_no=1)(p3)
        p5 = categorical_pipeline_complex(seq_no=1)(p3)
        p6 = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                             df_out_dtype_transforms=[(column_object, 'category')])([p4, p5])

        lightgbm_init_kwargs = {
            'boosting_type': Choice(['gbdt', 'dart', 'goss']),
            'num_leaves': Choice([11, 31, 101, 301, 501]),
            'learning_rate': Real(0.001, 0.1, step=0.005),
            'n_estimators': 100,
            'max_depth': -1,
            # subsample_for_bin = 200000, objective = None, class_weight = None,
            #  min_split_gain = 0., min_child_weight = 1e-3, min_child_samples = 20,
        }

        lightgbm_est = LightGBMEstimator(task='binary', fit_kwargs=lightgbm_fit_kwargs, **lightgbm_init_kwargs)

        xgb_init_kwargs = {

        }
        xgb_est = XGBoostEstimator(task='binary', fit_kwargs=xgb_fit_kwargs, **xgb_init_kwargs)

        or_est = ModuleChoice([lightgbm_est, xgb_est])(p6)
        space.set_inputs(input)
    return space


def get_kwargs(**kwargs):
    return kwargs


lightgbm_fit_kwargs = get_kwargs(sample_weight=None, init_score=None,
                                 eval_set=None, eval_names=None, eval_sample_weight=None,
                                 eval_init_score=None, eval_metric=None, early_stopping_rounds=None,
                                 verbose=True, feature_name='auto', categorical_feature='auto', callbacks=None)


def get_df():
    X = DataFrame(
        {
            "a": ['a', 'b', np.nan],
            "b": list(range(1, 4)),
            "c": np.arange(3, 6).astype("u1"),
            "d": np.arange(4.0, 7.0, dtype="float64"),
            "e": [True, False, True],
            "f": pd.Categorical(['c', 'd', np.nan]),
            "g": pd.date_range("20130101", periods=3),
            "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
            "i": pd.date_range("20130101", periods=3, tz="CET"),
            "j": pd.period_range("2013-01", periods=3, freq="M"),
            "k": pd.timedelta_range("1 day", periods=3),
            "l": [1, 10, 1000]
        }
    )
    y = [1, 1, 0]
    return X, y


class Test_Estimator():
    def test_xgb_early_stoping(self):
        df = dsutils.load_bank().head(1000)
        y = df.pop('y')

        X = general_preprocessor(df).fit_transform(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=9527)
        import xgboost as xgb
        clf = xgb.XGBClassifier(n_estimators=1000)
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5)
        booster = clf.get_booster()
        assert booster.best_iteration == 8

        clf = xgb.XGBClassifier(n_estimators=booster.best_ntree_limit)
        clf.fit(X_train, y_train)
        booster = clf.get_booster()
        assert booster.best_iteration == 8

    def test_lightgbm_early_stoping(self):
        df = dsutils.load_bank().head(1000)
        y = df.pop('y')

        X = general_preprocessor(df).fit_transform(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=9527)
        import lightgbm as lgbm
        clf = lgbm.LGBMClassifier(n_estimators=1000)
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5)

        assert clf.best_iteration_ == 11

        clf = lgbm.LGBMClassifier(n_estimators=clf.best_iteration_)
        clf.fit(X_train, y_train)
        assert clf.booster_.params['num_iterations'] == 11

    def test_catboost_early_stoping(self):
        df = dsutils.load_bank().head(1000)
        y = df.pop('y')

        X = general_preprocessor(df).fit_transform(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=9527)
        import catboost as cat
        clf = cat.CatBoostClassifier(n_estimators=1000)
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5)
        assert clf.best_iteration_ == 86

        clf = cat.CatBoostClassifier(n_estimators=clf.tree_count_)
        clf.fit(X_train, y_train)
        assert clf.tree_count_ == 87

    def test_build_pipeline(self):
        space = search_space_general()
        space.random_sample()
        estimator = HyperGBMEstimator('binary', space, cache_dir=f'{test_output_dir}/hypergbm_cache')
        X, y = get_df()
        num_cols = column_number_exclude_timedelta(X)
        cat_cols = column_object_category_bool(X)
        df_1 = estimator.data_pipeline.fit_transform(X, y)
        assert list(df_1.columns) == ['a', 'e', 'f', 'b', 'c', 'd', 'l']
        assert df_1.shape == (3, 7)

        space = get_space_multi_dataframemapper()
        space.random_sample()
        estimator = HyperGBMEstimator('binary', space, cache_dir=f'{test_output_dir}/hypergbm_cache')
        X, y = get_df()
        df_1 = estimator.data_pipeline.fit_transform(X, y)
        assert list(df_1.columns) == ['a', 'e', 'f', 'b', 'c', 'd', 'l']
        assert df_1.shape == (3, 7)

    def test_build_pipeline_feature_gen(self):
        space = search_space_feature_gen()
        space.random_sample()
        estimator = HyperGBMEstimator('binary', space, cache_dir=f'{test_output_dir}/hypergbm_cache')
        X, y = get_df()
        X = X[column_exclude_datetime(X)]
        # X = dsutils.load_bank().head(100)
        # y = X.pop('y')
        df_1 = estimator.data_pipeline.fit_transform(X, y)
        assert len(set(df_1.columns.to_list()) - {'a', 'e', 'f', 'a__f', 'b', 'c', 'd', 'l', 'd / c', 'l / b', 'd / l',
                                                  'b / d', 'c / b', 'b / l', 'l / d', 'c / d', 'c / l', 'd / b',
                                                  'b / c',
                                                  'l / c'}) == 0
        assert df_1.shape == (3, 20)

        df_2 = estimator.data_pipeline.transform(X)
        assert df_2.shape == (3, 20)
