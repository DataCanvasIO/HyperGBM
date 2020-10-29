# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypergbm.data_split import split_by_model
from tabular_toolbox.datasets import dsutils

from sklearn.model_selection import train_test_split


class Test_data_split():
    def test_split_by_model_basic(self):
        df = dsutils.load_bank().head(10000)
        y = df.pop('y')
        X_train, X_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=9527)

        X_train, X_eval, y_train, y_eval = split_by_model(X_train, y_train, X_test, eval_size=0.3, max_trails=5)

        assert X_train.shape == (5600, 17)
        assert X_eval.shape == (2400, 17)
        assert y_train.shape == (5600,)
        assert y_eval.shape == (2400,)
