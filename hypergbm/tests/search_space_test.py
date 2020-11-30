# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import numpy as np
import pandas as pd

from hypergbm.search_space import search_space_general

ids = []


def get_id(m):
    ids.append(m.id)
    return True


def get_df():
    X = pd.DataFrame(
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


class Test_search_space():

    def test_general(self):
        global ids
        space = search_space_general()
        space.assign_by_vectors([0, 0, 0, 0, 1, 1, 2, 1, 1])
        space, _ = space.compile_and_forward()
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input1',
                       'ID_categorical_pipeline_simple_0_input',
                       'ID_numeric_pipeline_complex_0_input',
                       'ID_categorical_imputer_0',
                       'ID_numeric_imputer_0',
                       'ID_categorical_label_encoder_0',
                       'ID_numeric_pipeline_complex_0_output',
                       'ID_categorical_pipeline_simple_0_output',
                       'Module_DataFrameMapper_1',
                       'Module_LightGBMEstimator_1']

        next, (name, p) = space.Module_DataFrameMapper_1.compose()
        X, y = get_df()
        df_1 = p.fit_transform(X, y)
        assert list(df_1.columns) == ['a', 'e', 'f', 'b', 'c', 'd', 'l']
        assert df_1.shape == (3, 7)
