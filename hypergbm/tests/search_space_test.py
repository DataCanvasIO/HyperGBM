# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import numpy as np
import pandas as pd

from hypergbm.search_space import GeneralSearchSpaceGenerator

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
    y = np.array([1, 1, 0])
    return X, y


def test_list_options():
    n_estimators = [111, 222, 333]
    space = GeneralSearchSpaceGenerator(n_estimators=n_estimators)
    r = set()
    for _ in range(20):
        s = space()
        s.random_sample()
        for p in s.get_assigned_params():
            alias = p.alias
            if alias.endswith('.n_estimators'):
                r.add(p.value)

    assert r == set(n_estimators)
