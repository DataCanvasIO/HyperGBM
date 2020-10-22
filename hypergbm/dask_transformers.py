# -*- coding:utf-8 -*-
"""

"""

# from sklearn import impute, pipeline, compose, preprocessing as sk_pre, decomposition
# from . import sklearn_ex
from . import sklearn_pandas
import numpy as np
from dask_ml import decomposition
from dask_ml import impute as dimp
from dask_ml import preprocessing as dpre

from . import dask_ex as dex
from .transformers import HyperTransformer, ComposeTransformer, PipelineOutput


class StandardScaler(HyperTransformer):
    def __init__(self, copy=True, with_mean=True, with_std=True, space=None, name=None, **kwargs):
        if copy is not None and copy != True:
            kwargs['copy'] = copy
        if with_mean is not None and with_mean != True:
            kwargs['with_mean'] = with_mean
        if with_std is not None and with_std != True:
            kwargs['with_std'] = with_std
        HyperTransformer.__init__(self, dpre.StandardScaler, space, name, **kwargs)


class OneHotEncoder(HyperTransformer):
    def __init__(self, categories='auto', drop=None, sparse=True, dtype=np.float64, space=None,
                 name=None, **kwargs):
        if categories is not None and categories != 'auto':
            kwargs['categories'] = categories
        # else:
        #     kwargs['categories'] = 'object'
        if drop is not None:
            kwargs['drop'] = drop
        if sparse is not None and sparse != True:
            kwargs['sparse'] = sparse
        if dtype is not None and dtype != True:
            kwargs['dtype'] = dtype
        # if handle_unknown is not None and handle_unknown != 'error':
        # kwargs['handle_unknown'] = 'ignore' #fixme

        # HyperTransformer.__init__(self, dpre.OneHotEncoder, space, name, **kwargs)
        HyperTransformer.__init__(self, dex.OneHotEncoder, space, name, **kwargs)


class MinMaxScaler(HyperTransformer):
    def __init__(self, feature_range=(0, 1), copy=True, space=None, name=None, **kwargs):
        if copy is not None and copy != True:
            kwargs['copy'] = copy
        if feature_range is not None and feature_range != (0, 1):
            kwargs['feature_range'] = feature_range
        HyperTransformer.__init__(self, dpre.MinMaxScaler, space, name, **kwargs)


class RobustScaler(HyperTransformer):
    def __init__(self, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True, space=None,
                 name=None, **kwargs):
        if copy is not None and copy != True:
            kwargs['copy'] = copy
        if with_centering is not None and with_centering != True:
            kwargs['with_centering'] = with_centering
        if with_scaling is not None and with_scaling != True:
            kwargs['with_scaling'] = with_scaling
        if quantile_range is not None and quantile_range != (25.0, 75.0):
            kwargs['quantile_range'] = quantile_range

        HyperTransformer.__init__(self, dpre.RobustScaler, space, name, **kwargs)


class MaxAbsScaler(HyperTransformer):
    def __init__(self, copy=True, space=None, name=None, **kwargs):
        if copy is not None and copy != True:
            kwargs['copy'] = copy

        HyperTransformer.__init__(self, dex.MaxAbsScaler, space, name, **kwargs)


class SimpleImputer(HyperTransformer):
    def __init__(self, missing_values=np.nan, strategy="mean", fill_value=None, verbose=0, copy=True,
                 add_indicator=False, space=None, name=None, **kwargs):
        if missing_values is not None and missing_values != np.nan:
            kwargs['missing_values'] = missing_values
        if strategy is not None and strategy != "mean":
            kwargs['strategy'] = strategy
        if fill_value is not None:
            kwargs['fill_value'] = fill_value
        elif strategy == 'constant':
            print('eeeeeeeeeeeeeeeee')
        if verbose is not None and verbose != 0:
            kwargs['verbose'] = verbose
        if copy is not None and copy != True:
            kwargs['copy'] = copy
        if add_indicator is not None and add_indicator != False:
            kwargs['add_indicator'] = add_indicator

        HyperTransformer.__init__(self, dimp.SimpleImputer, space, name, **kwargs)


class MultiLabelEncoder(HyperTransformer):
    def __init__(self, columns=None, space=None, name=None, **kwargs):
        if columns is not None:
            kwargs['columns'] = columns
        HyperTransformer.__init__(self, dex.MultiLabelEncoder, space, name, **kwargs)


class OrdinalEncoder(HyperTransformer):
    def __init__(self, categories='auto', dtype=np.float64, space=None,
                 name=None, **kwargs):
        if categories is not None and categories != 'auto':
            kwargs['categories'] = categories
        if dtype is not None and dtype != True:
            kwargs['dtype'] = dtype

        HyperTransformer.__init__(self, dpre.OrdinalEncoder, space, name, **kwargs)


class TruncatedSVD(HyperTransformer):
    def __init__(self, n_components=2, algorithm="tsqr", n_iter=5, random_state=None, tol=0., space=None,
                 name=None, **kwargs):
        if n_components is not None:
            kwargs['n_components'] = n_components
        if tol is not None and tol != 0.0:
            kwargs['tol'] = tol
        if algorithm is not None and algorithm != 'tsqr':
            kwargs['algorithm'] = algorithm
        if n_iter is not None and n_iter != 5:
            kwargs['n_iter'] = n_iter
        if random_state is not None:
            kwargs['random_state'] = random_state

        HyperTransformer.__init__(self, dex.TruncatedSVD, space, name, **kwargs)


class DataFrameMapper(ComposeTransformer):
    def __init__(self, default=False, sparse=False, df_out=False, input_df=False, space=None, name=None, **hyperparams):
        if default != False:
            hyperparams['default'] = default
        if sparse is not None and sparse != False:
            hyperparams['sparse'] = sparse
        if df_out is not None and df_out != False:
            hyperparams['df_out'] = df_out
        if input_df is not None and input_df != False:
            hyperparams['input_df'] = input_df

        ComposeTransformer.__init__(self, space, name, **hyperparams)

    def compose(self):
        inputs = self.space.get_inputs(self)
        assert all([isinstance(m, PipelineOutput) for m in
                    inputs]), 'The upstream module of `DataFrameMapper` must be `Pipeline`.'
        transformers = []
        next = None
        for p in inputs:
            next, (pipeline_name, transformer) = p.compose()
            transformers.append((p.columns, transformer))

        pv = self.param_values
        ct = sklearn_pandas.DataFrameMapper(features=transformers, **pv)
        return next, (self.name, ct)
