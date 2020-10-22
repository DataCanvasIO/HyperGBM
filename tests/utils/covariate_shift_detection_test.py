# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypergbm.datasets.dsutils import load_bank
from hypergbm.utils.shift_detection import covariate_shift_score
from sklearn.metrics import matthews_corrcoef, make_scorer

matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)


class Test_covariate_shift_detection:
    def test_shift_score(self):
        df = load_bank().head(1000)
        scores = covariate_shift_score(df[:700], df[700:])
        assert scores['id'] == 1.0

    def test_shift_score_with_matthews_corrcoef(self):
        df = load_bank().head(1000)
        scores = covariate_shift_score(df[:700], df[700:], scorer=matthews_corrcoef_scorer)
        assert scores['id'] == 1.0

    def test_shift_score_cv(self):
        df = load_bank().head(1000)
        scores = covariate_shift_score(df[:700], df[700:], cv=5)
        assert scores['id'] == 0.95
