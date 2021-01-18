# -*- coding:utf-8 -*-
"""

"""

import numpy as np
from dask_ml import impute as dimp
from dask_ml import preprocessing as dm_pre

from hypergbm.pipeline import HyperTransformer
from hypernets.utils import logging
from tabular_toolbox import dask_ex as dex

logger = logging.get_logger(__name__)


class StandardScaler(HyperTransformer):
    def __init__(self, copy=True, with_mean=True, with_std=True, space=None, name=None, **kwargs):
        if copy is not None and copy != True:
            kwargs['copy'] = copy
        if with_mean is not None and with_mean != True:
            kwargs['with_mean'] = with_mean
        if with_std is not None and with_std != True:
            kwargs['with_std'] = with_std
        HyperTransformer.__init__(self, dm_pre.StandardScaler, space, name, **kwargs)


class SafeOneHotEncoder(HyperTransformer):
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
        HyperTransformer.__init__(self, dex.SafeOneHotEncoder, space, name, **kwargs)


class MinMaxScaler(HyperTransformer):
    def __init__(self, feature_range=(0, 1), copy=True, space=None, name=None, **kwargs):
        if copy is not None and copy != True:
            kwargs['copy'] = copy
        if feature_range is not None and feature_range != (0, 1):
            kwargs['feature_range'] = feature_range
        HyperTransformer.__init__(self, dm_pre.MinMaxScaler, space, name, **kwargs)


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

        HyperTransformer.__init__(self, dm_pre.RobustScaler, space, name, **kwargs)


class MaxAbsScaler(HyperTransformer):
    def __init__(self, copy=True, space=None, name=None, **kwargs):
        if copy is not None and copy != True:
            kwargs['copy'] = copy

        HyperTransformer.__init__(self, dex.MaxAbsScaler, space, name, **kwargs)


class SimpleImputer(HyperTransformer):
    def __init__(self, missing_values=np.nan, strategy="mean", fill_value=None, verbose=0, copy=True,
                 add_indicator=False, space=None, name=None, **kwargs):
        if missing_values != np.nan:
            kwargs['missing_values'] = missing_values
        if strategy is not None and strategy != "mean":
            kwargs['strategy'] = strategy
        if fill_value is not None:
            kwargs['fill_value'] = fill_value
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
    def __init__(self, space=None, name=None, **kwargs):
        HyperTransformer.__init__(self, dm_pre.OrdinalEncoder, space, name, **kwargs)


class SafeOrdinalEncoder(HyperTransformer):
    def __init__(self, space=None, name=None, **kwargs):
        HyperTransformer.__init__(self, dex.SafeOrdinalEncoder, space, name, **kwargs)


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


class CallableAdapterEncoder(HyperTransformer):
    def __init__(self, fn, space=None, name=None,
                 fit=False, fit_transform=False, transform=False, inverse_transform=False):
        HyperTransformer.__init__(self, dex.CallableAdapterEncoder, space, name,
                                  fn=fn,
                                  fit=fit, fit_transform=fit_transform,
                                  transform=transform, inverse_transform=inverse_transform)


class DataCacher(HyperTransformer):
    def __init__(self, cache_dict, space=None, name=None, cache_key=None, remove_keys=None,
                 fit=False, fit_transform=False, transform=False, inverse_transform=False):
        HyperTransformer.__init__(self, dex.DataCacher, space, name,
                                  cache_dict=cache_dict, cache_key=cache_key, remove_keys=remove_keys,
                                  fit=fit, fit_transform=fit_transform,
                                  transform=transform, inverse_transform=inverse_transform)


class CacheCleaner(HyperTransformer):
    def __init__(self, cache_dict, space=None, name=None,
                 fit=False, fit_transform=False, transform=False, inverse_transform=False):
        HyperTransformer.__init__(self, dex.CacheCleaner, space, name,
                                  cache_dict=cache_dict,
                                  fit=fit, fit_transform=fit_transform,
                                  transform=transform, inverse_transform=inverse_transform)
