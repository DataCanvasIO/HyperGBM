# -*- coding:utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
from pandas import DataFrame

from hypergbm.estimators import LightGBMEstimator
from hypergbm.pipeline import DataFrameMapper
from hypergbm.sklearn.sklearn_ops import categorical_pipeline_simple, categorical_pipeline_complex, \
    numeric_pipeline_simple, numeric_pipeline_complex
from hypernets.core.ops import HyperInput
from hypernets.core.search_space import HyperSpace
from tabular_toolbox.column_selector import column_number, column_object_category_bool

ids = []


def get_id(m):
    ids.append(m.id)
    return True


def get_space_categorical_pipeline():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = categorical_pipeline_simple()(input)
        p3 = DataFrameMapper(input_df=True, df_out=True)([p1])  # passthrough
        est = LightGBMEstimator(task='binary', fit_kwargs={})(p3)
        space.set_inputs(input)
    return space


def get_space_categorical_pipeline_complex():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = categorical_pipeline_complex()(input)
        p3 = DataFrameMapper(input_df=True, df_out=True)([p1])  # passthrough
        est = LightGBMEstimator(task='binary', fit_kwargs={})(p3)
        space.set_inputs(input)
    return space


def get_space_numeric_pipeline():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = numeric_pipeline_simple()(input)
        p3 = DataFrameMapper(input_df=True, df_out=True)([p1])  # passthrough
        est = LightGBMEstimator(task='binary', fit_kwargs={})(p3)
        space.set_inputs(input)
    return space


def get_space_numeric_pipeline_complex():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = numeric_pipeline_complex()(input)
        p3 = DataFrameMapper(input_df=True, df_out=True)([p1])  # passthrough
        est = LightGBMEstimator(task='binary', fit_kwargs={})(p3)
        space.set_inputs(input)
    return space


def get_space_num_cat_pipeline(default=False):
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = numeric_pipeline_simple()(input)
        p2 = categorical_pipeline_simple()(input)
        p3 = DataFrameMapper(default=default, input_df=True, df_out=True)([p1, p2])  # passthrough
        est = LightGBMEstimator(task='binary', fit_kwargs={})(p3)
        space.set_inputs(input)
    return space


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


class Test_CommonOps():
    def test_categorical_pipeline(self):
        space = get_space_categorical_pipeline()
        space.random_sample()
        space, _ = space.compile_and_forward()
        global ids
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input1', 'ID_categorical_pipeline_simple_0_input', 'ID_categorical_imputer_0',
                       'ID_categorical_label_encoder_0', 'ID_categorical_pipeline_simple_0_output',
                       'Module_DataFrameMapper_1', 'Module_LightGBMEstimator_1']

        next, (name, p) = space.Module_DataFrameMapper_1.compose()
        X, y = get_df()
        df_1 = p.fit_transform(X, y)
        assert list(df_1.columns) == ['a', 'e', 'f']
        assert df_1.shape == (3, 3)

    def test_categorical_pipeline_complex(self):
        global ids

        space = get_space_categorical_pipeline_complex()
        space.assign_by_vectors([1, 0])
        space, _ = space.compile_and_forward()
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input1', 'ID_categorical_pipeline_complex_0_input', 'ID_categorical_imputer_0',
                       'ID_categorical_label_encoder_0', 'ID_categorical_pipeline_complex_0_output',
                       'Module_DataFrameMapper_1', 'Module_LightGBMEstimator_1']

        next, (name, p) = space.Module_DataFrameMapper_1.compose()
        X, y = get_df()
        df_1 = p.fit_transform(X, y)
        assert list(df_1.columns) == ['a', 'e', 'f']
        assert df_1.shape == (3, 3)

        space = get_space_categorical_pipeline_complex()
        space.assign_by_vectors([1, 1, 0])
        space, _ = space.compile_and_forward()
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input1', 'ID_categorical_pipeline_complex_0_input', 'ID_categorical_imputer_0',
                       'ID_categorical_onehot_0', 'ID_categorical_pipeline_complex_0_output',
                       'Module_DataFrameMapper_1', 'Module_LightGBMEstimator_1']

        next, (name, p) = space.Module_DataFrameMapper_1.compose()
        X, y = get_df()
        df_1 = p.fit_transform(X, y)
        assert list(df_1.columns) == ['a_0_a', 'a_1_b', 'e_0_False', 'e_1_True', 'f_0_c', 'f_1_d']
        assert df_1.shape == (3, 6)

        space = get_space_categorical_pipeline_complex()
        space.assign_by_vectors([0, 1, 0])
        space, _ = space.compile_and_forward()
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input1', 'ID_categorical_pipeline_complex_0_input', 'ID_categorical_imputer_0',
                       'ID_categorical_onehot_0', 'ID_categorical_pipeline_complex_0_output',
                       'Module_DataFrameMapper_1', 'Module_LightGBMEstimator_1']

        next, (name, p) = space.Module_DataFrameMapper_1.compose()
        X, y = get_df()
        df_1 = p.fit_transform(X, y)
        assert list(df_1.columns) == ['a_0_a',
                                      'a_1_b',
                                      'a_2_missing_value',
                                      'e_0_False',
                                      'e_1_True',
                                      'f_0_c',
                                      'f_1_d',
                                      'f_2_missing_value']
        assert df_1.shape == (3, 8)

        space = get_space_categorical_pipeline_complex()
        space.assign_by_vectors([1, 1, 1])
        space, _ = space.compile_and_forward()
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input1', 'ID_categorical_pipeline_complex_0_input', 'ID_categorical_imputer_0',
                       'ID_categorical_onehot_0', 'ID_categorical_svd_0', 'ID_categorical_pipeline_complex_0_output',
                       'Module_DataFrameMapper_1', 'Module_LightGBMEstimator_1']

        next, (name, p) = space.Module_DataFrameMapper_1.compose()
        X, y = get_df()
        df_1 = p.fit_transform(X, y)
        assert list(df_1.columns) == ['a_e_f_0', 'a_e_f_1', 'a_e_f_2']
        assert df_1.shape == (3, 3)

    def test_numeric_pipeline(self):
        space = get_space_numeric_pipeline()
        space.random_sample()
        space, _ = space.compile_and_forward()
        global ids
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input1', 'ID_numeric_pipeline_simple_0_input', 'ID_numeric_imputer_0',
                       'ID_numeric_standard_scaler_0', 'ID_numeric_pipeline_simple_0_output',
                       'Module_DataFrameMapper_1', 'Module_LightGBMEstimator_1']

        next, (name, p) = space.Module_DataFrameMapper_1.compose()
        X, y = get_df()
        df_1 = p.fit_transform(X, y)
        assert list(df_1.columns) == ['b', 'c', 'd', 'l']
        assert df_1.shape == (3, 4)

    def test_numeric_pipeline_complex(self):
        global ids

        space = get_space_numeric_pipeline_complex()
        space.assign_by_vectors([1, 0])
        space, _ = space.compile_and_forward()
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input1', 'ID_numeric_pipeline_complex_0_input', 'ID_numeric_imputer_0',
                       'ID_numeric_pipeline_complex_0_output', 'Module_DataFrameMapper_1', 'Module_LightGBMEstimator_1']

        next, (name, p) = space.Module_DataFrameMapper_1.compose()
        X, y = get_df()
        df_1 = p.fit_transform(X, y)
        assert list(df_1.columns) == ['b', 'c', 'd', 'l']
        assert df_1.shape == (3, 4)

        space = get_space_numeric_pipeline_complex()
        space.assign_by_vectors([0, 1, 0])
        space, _ = space.compile_and_forward()
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input1', 'ID_numeric_pipeline_complex_0_input', 'ID_numeric_imputer_0',
                       'ID_numeric_log_standard_scaler_0', 'ID_numeric_pipeline_complex_0_output',
                       'Module_DataFrameMapper_1', 'Module_LightGBMEstimator_1']

    def test_num_cat_pipeline(self):
        space = get_space_num_cat_pipeline()
        space.random_sample()
        space, _ = space.compile_and_forward()
        global ids
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input1', 'ID_categorical_pipeline_simple_0_input', 'ID_numeric_pipeline_simple_0_input',
                       'ID_categorical_imputer_0', 'ID_numeric_imputer_0', 'ID_categorical_label_encoder_0',
                       'ID_numeric_standard_scaler_0', 'ID_categorical_pipeline_simple_0_output',
                       'ID_numeric_pipeline_simple_0_output', 'Module_DataFrameMapper_1', 'Module_LightGBMEstimator_1']

        next, (name, p) = space.Module_DataFrameMapper_1.compose()
        X, y = get_df()
        df_1 = p.fit_transform(X, y)
        assert list(df_1.columns) == ['a', 'e', 'f', 'b', 'c', 'd', 'l']
        assert df_1.shape == (3, 7)

        space = get_space_num_cat_pipeline(default=None)
        space.random_sample()
        space, _ = space.compile_and_forward()
        next, (name, p) = space.Module_DataFrameMapper_1.compose()
        X, y = get_df()
        df_1 = p.fit_transform(X, y)
        assert df_1.shape == (3, 12)
        assert list(df_1.columns) == ['a', 'e', 'f', 'b', 'c', 'd', 'l', 'g', 'h', 'i', 'j', 'k']
