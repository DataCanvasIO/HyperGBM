# -*- coding:utf-8 -*-
"""

"""
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import copy
from hypergbm.utils.column_selector import column_skewness_kurtosis, column_object, column_int


class SafeLabelEncoder(LabelEncoder):
    def transform(self, y):
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        unseen = len(self.classes_)
        y = np.array([np.searchsorted(self.classes_, x) if x in self.classes_ else unseen for x in y])
        return y


class MultiLabelEncoder:
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        assert len(X.shape) == 2
        n_features = X.shape[1]
        for n in range(n_features):
            le = SafeLabelEncoder()
            le.fit(X[:, n])
            self.encoders[n] = le
        return self

    def transform(self, X):
        assert len(X.shape) == 2
        n_features = X.shape[1]
        assert n_features == len(self.encoders.items())
        for n in range(n_features):
            X[:, n] = self.encoders[n].transform(X[:, n])
        return X


class SkewnessKurtosisTransformer:
    def __init__(self, transform_fn=None, skew_threshold=0.5, kurtosis_threshold=0.5):
        self.columns_ = []
        self.skewness_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        if transform_fn is None:
            transform_fn = np.log
        self.transform_fn = transform_fn

    def fit(self, X, y=None):
        assert len(X.shape) == 2
        self.columns_ = column_skewness_kurtosis(X, skew_threshold=self.skewness_threshold,
                                                 kurtosis_threshold=self.kurtosis_threshold)
        print(f'Selected columns:{self.columns_}')
        return self

    def transform(self, X):
        assert len(X.shape) == 2
        if len(self.columns_) > 0:
            try:
                X[self.columns_] = self.transform_fn(X[self.columns_])
            except Exception as e:
                print(e)
        return X


def reduce_mem_usage(df, verbose=True):
    """
    Adaption from :https://blog.csdn.net/xckkcxxck/article/details/88170281
    :param verbose:
    :return:
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))


class DataCleaner:
    def __init__(self, nan_chars=None, deduce_object_dtype=True, clean_invalidate_columns=True,
                 clean_label_nan_rows=True,
                 replace_inf_values=np.nan, drop_columns=None, reduce_mem_usage=False, int_convert_to='float'):
        self.nan_chars = nan_chars
        self.deduce_object_dtype = deduce_object_dtype
        self.clean_invalidate_columns = clean_invalidate_columns
        self.clean_label_nan_rows = clean_label_nan_rows
        self.replace_inf_values = replace_inf_values
        self.drop_columns = drop_columns
        self.df_meta = None
        self.reduce_mem_usage = reduce_mem_usage
        self.int_convert_to = int_convert_to

    def clean_data(self, X, y):
        assert isinstance(X, pd.DataFrame)
        if y is not None:
            X.insert(0, 'hypergbm__Y__', y)

        if self.nan_chars is not None:
            print(f'Replace chars{self.nan_chars} to NaN')
            X = X.replace(self.nan_chars, np.nan)

        if self.deduce_object_dtype:
            print('Deduce data type for object columns.')
            for col in column_object(X):
                try:
                    X[col] = X[col].astype('float')
                except Exception as e:
                    print(f'Deduce object column [{col}] failed. {e}')

        if self.int_convert_to is not None:
            print(f'Convert int type to {self.int_convert_to}')
            int_cols = column_int(X)
            X[int_cols] = X[int_cols].astype(self.int_convert_to)

        if y is not None:
            if self.clean_label_nan_rows:
                print('Clean the rows which label is NaN')
                X = X.dropna(subset=['hypergbm__Y__'])
            y = X.pop('hypergbm__Y__')

        if self.drop_columns is not None:
            print(f'Drop columns:{self.drop_columns}')
            for col in self.drop_columns:
                X.pop(col)

        if self.clean_invalidate_columns:
            print('Clean invalidate columns')
            for col in X.columns:
                n_unique = X[col].nunique(dropna=True)
                if n_unique <= 1:
                    X.pop(col)

        o_cols = column_object(X)
        X[o_cols] = X[o_cols].astype('str')
        return X, y

    def fit_transform(self, X, y=None, copy_data=True):
        if copy_data:
            X = copy.deepcopy(X)
            if y is not None:
                y = copy.deepcopy(y)

        X, y = self.clean_data(X, y)
        if self.reduce_mem_usage:
            print('Reduce memory usage')
            reduce_mem_usage(X)

        if self.replace_inf_values is not None:
            print(f'Replace [inf,-inf] to {self.replace_inf_values}')
            X = X.replace([np.inf, -np.inf], self.replace_inf_values)

        print('Collect meta info from data')
        df_meta = {}
        for col_info in zip(X.columns.to_list(), X.dtypes):
            dtype = str(col_info[1])
            if df_meta.get(dtype) is None:
                df_meta[dtype] = []
            df_meta[dtype].append(col_info[0])
        self.df_meta = df_meta
        return X, y

    def transform(self, X, y=None, copy_data=True):
        if copy_data:
            X = copy.deepcopy(X)
            if y is not None:
                y = copy.deepcopy(y)
        self.clean_data(X, y)
        if self.df_meta is not None:
            print('Processing with meta info')
            all_cols = []
            for dtype, cols in self.df_meta.items():
                all_cols += cols
                X[cols] = X[cols].astype(dtype)
            drop_cols = set(X.columns.to_list()) - set(all_cols)

            for c in drop_cols:
                X.pop(c)
            print(f'droped columns:{drop_cols}')

        if self.replace_inf_values is not None:
            print(f'Replace [inf,-inf] to {self.replace_inf_values}')
            X = X.replace([np.inf, -np.inf], self.replace_inf_values)

        return X, y

