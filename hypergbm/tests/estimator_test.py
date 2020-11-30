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
from tabular_toolbox.column_selector import column_object, column_exclude_datetime
from tabular_toolbox.datasets import dsutils
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
    def test_build_pipeline(self):
        space = search_space_general()
        space.random_sample()
        estimator = HyperGBMEstimator('binary', space, cache_dir=f'{test_output_dir}/hypergbm_cache')
        X, y = get_df()
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

    def test_pipeline_signature(self):
        space = search_space_general(
            lightgbm_fit_kwargs=lightgbm_fit_kwargs,
        )
        # space.random_sample()
        # assert space.vectors
        space.assign_by_vectors([0, 0, 0, 0, 1, 1, 2, 1, 1])
        estimator = HyperGBMEstimator('binary', space, cache_dir=f'{test_output_dir}/hypergbm_cache')
        assert estimator.get_pipeline_signature(estimator.data_pipeline) in ['e1129afc88d6136d060a986d0c484a26',
                                                                             'b0bed4a992cf4f996d3da200b4769363',
                                                                             '96444f17c75f8f68857bf16a5de0d74a']

    def test_bankdata_lightgbm(self):
        space = search_space_general(
            lightgbm_fit_kwargs=lightgbm_fit_kwargs,
        )

        space.assign_by_vectors([0, 0, 0, 0, 1, 1, 2, 1, 1])
        estimator = HyperGBMEstimator('binary', space, cache_dir=f'{test_output_dir}/hypergbm_cache')
        df = dsutils.load_bank()
        df.drop(['id'], axis=1, inplace=True)
        X_train, X_test = train_test_split(df.head(10000), test_size=0.2, random_state=42)
        y_train = X_train.pop('y')
        y_test = X_test.pop('y')

        estimator.fit(X_train, y_train)
        scores = estimator.evaluate(X_test, y_test, metrics=['accuracy'])
        assert scores
        print(scores)

    def test_bankdata_xgb(self):
        space = search_space_general(
            lightgbm_fit_kwargs=lightgbm_fit_kwargs,
        )
        space.assign_by_vectors([1, 1, 0, 0, 0, 3, 2, 1, 1, 3, 3])
        estimator = HyperGBMEstimator('binary', space, cache_dir=f'{test_output_dir}/hypergbm_cache')
        df = dsutils.load_bank()
        df.drop(['id'], axis=1, inplace=True)
        X_train, X_test = train_test_split(df.head(10000), test_size=0.2, random_state=42)
        y_train = X_train.pop('y')
        y_test = X_test.pop('y')

        estimator.fit(X_train, y_train)
        scores = estimator.evaluate(X_test, y_test, metrics=['accuracy'])
        assert scores
        print(scores)

    def test_bankdata_catboost(self):
        space = search_space_general(
            lightgbm_fit_kwargs=lightgbm_fit_kwargs,
        )
        space.assign_by_vectors([2, 2, 1, 1, 0.031, 1, 3, 1])
        estimator = HyperGBMEstimator('binary', space, cache_dir=f'{test_output_dir}/hypergbm_cache')
        df = dsutils.load_bank()
        df.drop(['id'], axis=1, inplace=True)
        X_train, X_test = train_test_split(df.head(10000), test_size=0.2, random_state=42)
        y_train = X_train.pop('y')
        y_test = X_test.pop('y')

        estimator.fit(X_train, y_train)
        scores = estimator.evaluate(X_test, y_test, metrics=['accuracy'])
        assert scores
        print(scores)
