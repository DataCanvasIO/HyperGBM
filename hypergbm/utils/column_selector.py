# -*- coding:utf-8 -*-
"""

"""
import numpy as np
from dask import dataframe as dd
from scipy.stats import skew, kurtosis
from sklearn.compose import make_column_selector


class HyperColumnSelector(make_column_selector):
    __doc__ = make_column_selector.__doc__

    def __call__(self, df):
        if isinstance(df, dd.DataFrame):
            # if not hasattr(df, 'iloc'):
            #     raise ValueError("make_column_selector can only be applied to "
            #                      "pandas dataframes")
            # df_row = df.iloc[:1]
            df_row = df

            if self.dtype_include is not None or self.dtype_exclude is not None:
                df_row = df_row.select_dtypes(include=self.dtype_include,
                                              exclude=self.dtype_exclude)
            cols = df_row.columns
            if self.pattern is not None:
                cols = cols[cols.str.contains(self.pattern, regex=True)]
            return cols.tolist()

        return super(HyperColumnSelector, self).__call__(df)


column_object_category_bool = HyperColumnSelector(dtype_include=['object', 'category', 'bool'])
column_object = HyperColumnSelector(dtype_include=['object'])
column_category = HyperColumnSelector(dtype_include=['category'])
column_bool = HyperColumnSelector(dtype_include=['bool'])
column_number = HyperColumnSelector(dtype_include='number')
column_number_exclude_timedelta = HyperColumnSelector(dtype_include='number', dtype_exclude='timedelta')

column_timedelta = HyperColumnSelector(dtype_include='timedelta')
column_datetimetz = HyperColumnSelector(dtype_include='datetimetz')
column_datetime = HyperColumnSelector(dtype_include='datetime')
column_all_datetime = HyperColumnSelector(dtype_include=['datetime', 'datetimetz'])
column_int = HyperColumnSelector(dtype_include=['int16', 'int32', 'int64'])


def column_skewness_kurtosis(X, skew_threshold=0.5, kurtosis_threshold=0.5, columns=None):
    if columns is None:
        columns = column_number_exclude_timedelta(X)
    skew_values = skew(X[columns], axis=0, nan_policy='omit')
    kurtosis_values = kurtosis(X[columns], axis=0, nan_policy='omit')
    selected = [c for i, c in enumerate(columns) if
                abs(skew_values[i]) > skew_threshold or abs(kurtosis_values[i]) > kurtosis_threshold]
    return selected


def column_skewness_kurtosis_diff(X_1, X_2, diff_threshold=5, columns=None, smooth_fn=np.log, skewness_weights=1,
                                  kurtosis_weights=0):
    skew_x_1, skew_x_2, kurtosis_x_1, kurtosis_x_2, columns = calc_skewness_kurtosis(X_1, X_2, columns, smooth_fn)
    diff = np.log(
        abs(skew_x_1 - skew_x_2) * skewness_weights + np.log(abs(kurtosis_x_1 - kurtosis_x_2)) * kurtosis_weights)
    if isinstance(diff_threshold, tuple):
        index = np.argwhere((diff > diff_threshold[0]) & (diff <= diff_threshold[1]))
    else:
        index = np.argwhere(diff > diff_threshold)
    selected = [c for i, c in enumerate(columns) if i in index]
    return selected


def calc_skewness_kurtosis(X_1, X_2, columns=None, smooth_fn=np.log):
    if columns is None:
        columns = column_number_exclude_timedelta(X_1)
    X_1_t = X_1[columns]
    X_2_t = X_2[columns]
    if smooth_fn is not None:
        X_1_t[columns] = smooth_fn(X_1_t)
        X_2_t[columns] = smooth_fn(X_2_t)

    skew_x_1 = skew(X_1_t, axis=0, nan_policy='omit')
    skew_x_2 = skew(X_2_t, axis=0, nan_policy='omit')
    kurtosis_x_1 = kurtosis(X_1_t, axis=0, nan_policy='omit')
    kurtosis_x_2 = kurtosis(X_2_t, axis=0, nan_policy='omit')
    return skew_x_1, skew_x_2, kurtosis_x_1, kurtosis_x_2, columns
