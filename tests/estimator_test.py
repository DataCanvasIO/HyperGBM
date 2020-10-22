# -*- coding:utf-8 -*-
"""

"""
from pandas import DataFrame
import pandas as pd
from hypergbm.datasets import dsutils
from hypergbm.transformers import ColumnTransformer, DataFrameMapper
from hypergbm.common_ops import categorical_pipeline_simple, numeric_pipeline, \
    categorical_pipeline_complex, numeric_pipeline_complex
from hypergbm.estimators import LightGBMEstimator, XGBoostEstimator
from hypergbm.hyper_gbm import HyperGBMEstimator
from hypergbm.utils.column_selector import column_object
from hypergbm.common_ops import get_space_num_cat_pipeline_complex
from hypernets.core.ops import *

from .common_ops_test import get_space_categorical_pipeline
from sklearn.model_selection import train_test_split
from tests import test_output_dir


def get_space_multi_dataframemapper(default=False):
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = numeric_pipeline(seq_no=0)(input)
        p2 = categorical_pipeline_simple(seq_no=0)(input)
        p3 = DataFrameMapper(default=default, input_df=True, df_out=True)([p1, p2])  # passthrough

        p4 = numeric_pipeline(seq_no=1)(p3)
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
        space = get_space_categorical_pipeline()
        space.random_sample()
        estimator = HyperGBMEstimator('binary', space, cache_dir=f'{test_output_dir}/hypergbm_cache')
        X, y = get_df()
        df_1 = estimator.pipeline.fit_transform(X, y)
        assert list(df_1.columns) == ['a', 'e', 'f']
        assert df_1.shape == (3, 3)

        space = get_space_multi_dataframemapper()
        space.random_sample()
        estimator = HyperGBMEstimator('binary', space, cache_dir=f'{test_output_dir}/hypergbm_cache')
        X, y = get_df()
        df_1 = estimator.pipeline.fit_transform(X, y)
        assert list(df_1.columns) == ['a', 'e', 'f', 'b', 'c', 'd', 'l']
        assert df_1.shape == (3, 7)

    def test_pipeline_signature(self):
        space = get_space_num_cat_pipeline_multi_complex(
            lightgbm_fit_kwargs=lightgbm_fit_kwargs,
        )
        space.assign_by_vectors([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 3, 0.01, 1, 1])
        estimator = HyperGBMEstimator('binary', space, cache_dir=f'{test_output_dir}/hypergbm_cache')
        #assert estimator.get_pipeline_signature(estimator.pipeline) == '2583ff8ce53e6c8244a91f4d6554f39a'

    def test_bankdata_lightgbm(self):
        space = get_space_num_cat_pipeline_complex(
            lightgbm_fit_kwargs=lightgbm_fit_kwargs,
        )
        space.assign_by_vectors([0, 1, 1, 0, 1, 0, 3, 0.01, 1])
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
        space = get_space_num_cat_pipeline_complex(
            lightgbm_fit_kwargs=lightgbm_fit_kwargs,
        )
        space.assign_by_vectors([1, 1, 1, 0, 1, 1])
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
        space = get_space_num_cat_pipeline_complex(
            lightgbm_fit_kwargs=lightgbm_fit_kwargs,
        )
        space.assign_by_vectors([2, 1, 1, 0, 1, 1])
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
