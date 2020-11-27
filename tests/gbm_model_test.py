# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMModel

from hypergbm.hyper_gbm import is_lightgbm_model, is_catboost_model


class Test_GBModel():
    def test_class_name(self):
        m1 = LGBMClassifier()
        m2 = LGBMRegressor()
        m3 = LGBMModel()
        m4 = CatBoostRegressor()
        m5 = CatBoostClassifier()

        assert is_lightgbm_model(m1) == True
        assert is_lightgbm_model(m2) == True
        assert is_lightgbm_model(m3) == True
        assert is_lightgbm_model(m4) == False

        assert is_catboost_model(m1) == False
        assert is_catboost_model(m4) == True
        assert is_catboost_model(m5) == True

