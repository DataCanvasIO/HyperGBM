# -*- coding:utf-8 -*-
"""

"""
import sys

import numpy as np
import featuretools as ft
from featuretools import IdentityFeature, variable_types
from datetime import datetime
import pandas as pd


class FeatureToolsTransformer():

    def __init__(self, trans_primitives=None,
                 fix_input=True,
                 fix_output=True,
                 continuous_cols=None,
                 datetime_cols=None,
                 max_depth=1):
        """

        Args:
            trans_primitives:
                for continuous: "add_numeric","subtract_numeric","divide_numeric","multiply_numeric","negate","modulo_numeric","modulo_by_feature","cum_mean","cum_sum","cum_min","cum_max","percentile","absolute"
                for datetime: "year", "month", "week", "minute", "day", "hour", "minute", "second", "weekday", "is_weekend"
                for text(not support now): "num_characters", "num_words"
            max_depth:
        """
        self.trans_primitives = trans_primitives
        self.max_depth = max_depth
        self._feature_defs = None

        self._imputed_output = None
        self._imputed_input = None
        self._valid_cols = None

        self.fix_input = fix_input
        self.fix_output = fix_output
        if self.fix_input is False and self.fix_output is True:
            sys.stderr.write("May fill out of place values after derivation.")

        self.continuous_cols = continuous_cols
        self.datetime_cols = datetime_cols

    def _filter_by_type(self, fields, types):
        result = []
        for f, t in fields:
            for _t in types:
                if t.type == _t:
                    result.append(f)
        return result

    def _merge_dict(self, dest_dict, *dicts):
        for d in dicts:
            for k, v in d.items():
                dest_dict.setdefault(k, v)

    def fit(self, X, **kwargs):
        # self._check_values(X)
        fields = X.dtypes.to_dict().items()

        if self.continuous_cols is None:
            self.continuous_cols = self._filter_by_type(fields, [np.int, np.int32, np.int64, np.float, np.float32, np.float64])

        if self.datetime_cols is None:
            self.datetime_cols = self._filter_by_type(fields, [datetime, pd.datetime])

        if self.fix_input:
            _mean = X[self.continuous_cols].mean().to_dict()
            _mode = X[self.datetime_cols].mode().to_dict()
            self._imputed_input = {}
            self._merge_dict(self._imputed_input, _mean, _mode)
            self._replace_invalid_values(X, self._imputed_input)

        feature_type_dict = {}
        self._merge_dict(feature_type_dict,
                         {c: variable_types.Numeric for c in self.continuous_cols},
                         {c: variable_types.Datetime for c in self.datetime_cols})

        es = ft.EntitySet(id='es_hypernets_fit')
        es.entity_from_dataframe(entity_id='e_hypernets_ft', dataframe=X, variable_types=feature_type_dict,
                                 make_index=True, index='e_hypernets_ft_index')
        feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="e_hypernets_ft",
                                              ignore_variables={"e_hypernets_ft": []},
                                              return_variable_types="all",
                                              trans_primitives=self.trans_primitives,
                                              max_depth=self.max_depth,
                                              features_only=False,
                                              max_features=-1)
        self._feature_defs = feature_defs

        if self.fix_output:
            derived_cols = list(map(lambda _: _._name, filter(lambda _: not isinstance(_, IdentityFeature), feature_defs)))
            invalid_cols = self._checkout_invalid_cols(feature_matrix)
            self._valid_cols = set(derived_cols) - set(invalid_cols)
            # td:  check no valid cols
            self._imputed_output = feature_matrix[self._valid_cols].replace([np.inf, -np.inf], np.nan).mean().to_dict()

        return self

    def _replace_invalid_values(self, df: pd.DataFrame, imputed_dict):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(imputed_dict, inplace=True)

    def transform(self, X):
        # 1. check is fitted and values
        assert self._feature_defs is not None, 'Please fit it first.'

        # 2. fix input
        if self.fix_input:
            self._replace_invalid_values(X, self._imputed_input)

        # 3. transform
        es = ft.EntitySet(id='es_hypernets_transform')
        es.entity_from_dataframe(entity_id='e_hypernets_ft', dataframe=X, make_index=False)
        feature_matrix = ft.calculate_feature_matrix(self._feature_defs, entityset=es, n_jobs=1, verbose=10)

        # 4. fix output
        if self.fix_output:
            self._replace_invalid_values(feature_matrix, self._imputed_output)

        return feature_matrix

    def _contains_null_cols(self, df):
        _df = df.replace([np.inf, -np.inf], np.nan)
        return list(map(lambda _: _[0], filter(lambda _: _[1] > 0,  _df.isnull().sum().to_dict().items())))

    def _check_values(self, df):
        nan_cols = self._contains_null_cols(df)
        if len(nan_cols) > 0:
            _s = ",".join(nan_cols)
            raise ValueError(f"Following columns contains NaN,Inf,-Inf value that can not derivation: {_s} .")

    def _checkout_invalid_cols(self, df):
        result = []
        _df = df.replace([np.inf, -np.inf], np.nan)

        if _df.shape[0] > 0:
            for col in _df:
                if _df[col].nunique(dropna=False) < 1 or _df[col].dropna().shape[0] < 1:
                    result.append(col)
        return result
