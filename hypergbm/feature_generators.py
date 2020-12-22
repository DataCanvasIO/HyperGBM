# -*- coding:utf-8 -*-
"""

"""

import featuretools as ft
import numpy as np
import pandas as pd
from featuretools import variable_types, primitives
from tabular_toolbox.column_selector import column_all_datetime, column_number_exclude_timedelta
from tabular_toolbox.sklearn_ex import FeatureSelectionTransformer


class CrossCategorical(primitives.TransformPrimitive):
    name = "cross_categorical"
    input_types = [variable_types.Categorical, variable_types.Categorical]
    return_type = variable_types.Categorical
    commutative = True
    dask_compatible = True

    def get_function(self):
        return self.char_add

    def char_add(self, x1, x2):
        return np.char.add(np.array(x1, 'U'), np.char.add('__', np.array(x2, 'U')))

    def generate_name(self, base_feature_names):
        return "%s__%s" % (base_feature_names[0], base_feature_names[1])


class FeatureGenerationTransformer():
    ft_index = 'e_hypernets_ft_index'

    def __init__(self, task=None, trans_primitives=None,
                 fix_input=False,
                 continuous_cols=None,
                 datetime_cols=None,
                 max_depth=1, feature_selection_args=None):
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
        self.task = task

        self._imputed_input = None

        self.fix_input = fix_input
        self.continuous_cols = continuous_cols
        self.datetime_cols = datetime_cols
        self.original_cols = []
        self.selection_transformer = None
        self.selection_args = feature_selection_args
        self.feature_defs_ = None

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

    def fit(self, X, y=None, **kwargs):
        self.original_cols = X.columns.to_list()
        if self.selection_args is not None:
            assert y is not None, '`y` must be provided for feature selection.'
            self.selection_args['reserved_cols'] = self.original_cols
            self.selection_transformer = FeatureSelectionTransformer(task=self.task, **self.selection_args)
        # self._check_values(X)
        if self.continuous_cols is None:
            self.continuous_cols = column_number_exclude_timedelta(X)
        if self.datetime_cols is None:
            self.datetime_cols = column_all_datetime(X)

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

        make_index = True
        if self.ft_index in X.columns.to_list():
            make_index = False

        es.entity_from_dataframe(entity_id='e_hypernets_ft', dataframe=X, variable_types=feature_type_dict,
                                 make_index=make_index, index=self.ft_index)
        feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="e_hypernets_ft",
                                              ignore_variables={"e_hypernets_ft": []},
                                              return_variable_types="all",
                                              trans_primitives=self.trans_primitives,
                                              max_depth=self.max_depth,
                                              features_only=False,
                                              max_features=-1)
        X.pop(self.ft_index)

        self.feature_defs_ = feature_defs

        if self.selection_transformer is not None:
            self.selection_transformer.fit(feature_matrix, y)
            selected_defs = []
            for fea in self.feature_defs_:
                if fea._name in self.selection_transformer.columns_:
                    selected_defs.append(fea)
            self.feature_defs_ = selected_defs

        return self

    @property
    def classes_(self):
        if self.feature_defs_ is None:
            return None
        feats = [fea._name for fea in self.feature_defs_]
        return feats

    def _replace_invalid_values(self, df: pd.DataFrame, imputed_dict):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(imputed_dict, inplace=True)

    def transform(self, X):
        # 1. check is fitted and values
        assert self.feature_defs_ is not None, 'Please fit it first.'

        # 2. fix input
        if self.fix_input:
            self._replace_invalid_values(X, self._imputed_input)

        # 3. transform
        es = ft.EntitySet(id='es_hypernets_transform')
        es.entity_from_dataframe(entity_id='e_hypernets_ft', dataframe=X, make_index=(self.ft_index not in X),
                                 index=self.ft_index)
        feature_matrix = ft.calculate_feature_matrix(self.feature_defs_, entityset=es, n_jobs=1, verbose=10)
        feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)

        return feature_matrix

    def _contains_null_cols(self, df):
        _df = df.replace([np.inf, -np.inf], np.nan)
        return list(map(lambda _: _[0], filter(lambda _: _[1] > 0, _df.isnull().sum().to_dict().items())))

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
