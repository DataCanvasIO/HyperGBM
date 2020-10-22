# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import copy
import pandas as pd
from pandas import DataFrame
from hypergbm.utils.column_selector import *


class Test_ColumnSelector():
    def test_select_dtypes_include_using_list_like(self):
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        num = column_number(df)
        assert num == ['b', 'c', 'd', 'k']

        num_exclude_time = column_number_exclude_timedelta(df)
        assert num_exclude_time == ['b', 'c', 'd']

        o_c_b = column_object_category_bool(df)
        assert o_c_b == ['a', 'e', 'f']

        o = column_object(df)
        assert o == ['a']
        c = column_category(df)
        assert c == ['f']
        b = column_bool(df)
        assert b == ['e']

        t = column_timedelta(df)
        assert t == ['k']

        d_tz = column_datetimetz(df)
        assert d_tz == ['h', 'i']

        d = column_datetime(df)
        assert d == ['g']

        all_d = column_all_datetime(df)
        assert all_d == ['g', 'h', 'i']

        skewed = column_skewness_kurtosis(df, 0.5)
        assert skewed == ['b', 'c', 'd']

    def test_skew(self):
        np.random.seed(1)
        x0 = np.random.uniform(0, 1, 100)

        x1 = np.random.uniform(0, 1, 100)
        x1.sort()
        x1[79:99] = x1[79:99] * 4
        x2 = np.random.uniform(0, 1, 100)
        x2.sort()
        x2[:20] = x2[:20] * 4

        x3 = np.random.uniform(0, 1, 100)
        x3.sort()
        x3[40:60] = x3[40:60] * 100

        df = pd.DataFrame(np.stack([x0, x1, x2, x3], axis=1))
        df.columns = ['x0', 'x1', 'x2', 'x3']
        skewed = column_skewness_kurtosis(df, skew_threshold=0.5, kurtosis_threshold=2)
        assert skewed == ['x1', 'x3']

        df1 = copy.deepcopy(df)
        v = np.log(df1[skewed])
        df1[skewed] = v
        skewed = column_skewness_kurtosis(df1, 0.5, kurtosis_threshold=2)
        assert skewed == ['x3']
