# -*- coding:utf-8 -*-
"""

"""

import numpy as np
from sklearn import impute, preprocessing as sk_pre, decomposition

from hypergbm.pipeline import HyperTransformer
from tabular_toolbox import sklearn_ex
from tabular_toolbox.sklearn_ex import FloatOutputImputer
from .. import feature_generators


class LogStandardScaler(HyperTransformer):
    def __init__(self, copy=True, with_mean=True, with_std=True, space=None, name=None, **kwargs):
        if copy is not None and copy != True:
            kwargs['copy'] = copy
        if with_mean is not None and with_mean != True:
            kwargs['with_mean'] = with_mean
        if with_std is not None and with_std != True:
            kwargs['with_std'] = with_std
        HyperTransformer.__init__(self, sklearn_ex.LogStandardScaler, space, name, **kwargs)


class StandardScaler(HyperTransformer):
    def __init__(self, copy=True, with_mean=True, with_std=True, space=None, name=None, **kwargs):
        if copy is not None and copy != True:
            kwargs['copy'] = copy
        if with_mean is not None and with_mean != True:
            kwargs['with_mean'] = with_mean
        if with_std is not None and with_std != True:
            kwargs['with_std'] = with_std
        HyperTransformer.__init__(self, sk_pre.StandardScaler, space, name, **kwargs)


class MinMaxScaler(HyperTransformer):
    def __init__(self, feature_range=(0, 1), copy=True, space=None, name=None, **kwargs):
        if copy is not None and copy != True:
            kwargs['copy'] = copy
        if feature_range is not None and feature_range != (0, 1):
            kwargs['feature_range'] = feature_range
        HyperTransformer.__init__(self, sk_pre.MinMaxScaler, space, name, **kwargs)


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

        HyperTransformer.__init__(self, sk_pre.RobustScaler, space, name, **kwargs)


class MaxAbsScaler(HyperTransformer):
    def __init__(self, copy=True, space=None, name=None, **kwargs):
        if copy is not None and copy != True:
            kwargs['copy'] = copy

        HyperTransformer.__init__(self, sk_pre.MaxAbsScaler, space, name, **kwargs)


class LabelEncoder(HyperTransformer):
    def __init__(self, space=None, name=None, **kwargs):
        HyperTransformer.__init__(self, sk_pre.LabelEncoder, space, name, **kwargs)


class MultiLabelEncoder(HyperTransformer):
    def __init__(self, columns=None, space=None, name=None, **kwargs):
        if columns is not None:
            kwargs['columns'] = columns
        HyperTransformer.__init__(self, sklearn_ex.MultiLabelEncoder, space, name, **kwargs)


class OneHotEncoder(HyperTransformer):
    def __init__(self, categories='auto', drop=None, sparse=True, dtype=np.float64, space=None,
                 name=None, **kwargs):
        if categories is not None and categories != 'auto':
            kwargs['categories'] = categories
        if drop is not None:
            kwargs['drop'] = drop
        if sparse is not None and sparse != True:
            kwargs['sparse'] = sparse
        if dtype is not None and dtype != True:
            kwargs['dtype'] = dtype
        # if handle_unknown is not None and handle_unknown != 'error':
        kwargs['handle_unknown'] = 'ignore'

        HyperTransformer.__init__(self, sk_pre.OneHotEncoder, space, name, **kwargs)


class SafeOneHotEncoder(HyperTransformer):
    def __init__(self, categories='auto', drop=None, sparse=True, dtype=np.float64, space=None,
                 name=None, **kwargs):
        if categories is not None and categories != 'auto':
            kwargs['categories'] = categories
        if drop is not None:
            kwargs['drop'] = drop
        if sparse is not None and sparse != True:
            kwargs['sparse'] = sparse
        if dtype is not None and dtype != True:
            kwargs['dtype'] = dtype
        # if handle_unknown is not None and handle_unknown != 'error':
        kwargs['handle_unknown'] = 'ignore'

        HyperTransformer.__init__(self, sklearn_ex.SafeOneHotEncoder, space, name, **kwargs)


class OrdinalEncoder(HyperTransformer):
    def __init__(self, categories='auto', dtype=np.float64, space=None,
                 name=None, **kwargs):
        if categories is not None and categories != 'auto':
            kwargs['categories'] = categories
        if dtype is not None and dtype != True:
            kwargs['dtype'] = dtype

        HyperTransformer.__init__(self, sk_pre.OrdinalEncoder, space, name, **kwargs)


class SafeOrdinalEncoder(HyperTransformer):
    def __init__(self, categories='auto', dtype=np.float64, space=None,
                 name=None, **kwargs):
        if categories is not None and categories != 'auto':
            kwargs['categories'] = categories
        if dtype is not None and dtype != True:
            kwargs['dtype'] = dtype

        HyperTransformer.__init__(self, sklearn_ex.SafeOrdinalEncoder, space, name, **kwargs)


class KBinsDiscretizer(HyperTransformer):
    def __init__(self, n_bins=5, encode='onehot', strategy='quantile', space=None, name=None, **kwargs):
        if n_bins is not None and n_bins != 5:
            kwargs['n_bins'] = n_bins
        if encode is not None and encode != 'onehot':
            kwargs['encode'] = encode
        if strategy is not None and strategy != 'quantile':
            kwargs['strategy'] = strategy

        HyperTransformer.__init__(self, sk_pre.KBinsDiscretizer, space, name, **kwargs)


class Binarizer(HyperTransformer):
    def __init__(self, threshold=0.0, copy=True, space=None, name=None, **kwargs):
        if threshold is not None and threshold != 0.0:
            kwargs['threshold'] = threshold
        if copy is not None and copy != True:
            kwargs['copy'] = copy

        HyperTransformer.__init__(self, sk_pre.Binarizer, space, name, **kwargs)


class LabelBinarizer(HyperTransformer):
    def __init__(self, neg_label=0, pos_label=1, sparse_output=False, space=None, name=None, **kwargs):
        if neg_label is not None and neg_label != 0:
            kwargs['neg_label'] = neg_label
        if pos_label is not None and pos_label != 1:
            kwargs['pos_label'] = pos_label
        if sparse_output is not None and sparse_output != False:
            kwargs['sparse_output'] = sparse_output

        HyperTransformer.__init__(self, sk_pre.LabelBinarizer, space, name, **kwargs)


class MultiLabelBinarizer(HyperTransformer):
    def __init__(self, classes=None, sparse_output=False, space=None, name=None, **kwargs):
        if classes is not None:
            kwargs['classes'] = classes
        if sparse_output is not None and sparse_output != False:
            kwargs['sparse_output'] = sparse_output

        HyperTransformer.__init__(self, sk_pre.MultiLabelBinarizer, space, name, **kwargs)


class FunctionTransformer(HyperTransformer):
    def __init__(self, func=None, inverse_func=None, validate=False,
                 accept_sparse=False, check_inverse=True, kw_args=None,
                 inv_kw_args=None, space=None, name=None, **kwargs):
        if func is not None:
            kwargs['func'] = func
        if inverse_func is not None:
            kwargs['inverse_func'] = inverse_func
        if validate is not None and validate != False:
            kwargs['validate'] = validate
        if accept_sparse is not None and accept_sparse != False:
            kwargs['accept_sparse'] = accept_sparse
        if check_inverse is not None and check_inverse != True:
            kwargs['check_inverse'] = check_inverse
        if kw_args is not None:
            kwargs['kw_args'] = kw_args
        if inv_kw_args is not None:
            kwargs['inv_kw_args'] = inv_kw_args

        HyperTransformer.__init__(self, sk_pre.FunctionTransformer, space, name, **kwargs)


class Normalizer(HyperTransformer):
    def __init__(self, norm='l2', copy=True, space=None, name=None, **kwargs):
        if norm is not None and norm != 'l2':
            kwargs['norm'] = norm
        if copy is not None and copy != True:
            kwargs['copy'] = copy

        HyperTransformer.__init__(self, sk_pre.Normalizer, space, name, **kwargs)


class PolynomialFeatures(HyperTransformer):
    def __init__(self, degree=2, interaction_only=False, include_bias=True, order='C', space=None, name=None, **kwargs):
        if degree is not None and degree != 2:
            kwargs['degree'] = degree
        if interaction_only is not None and interaction_only != False:
            kwargs['interaction_only'] = interaction_only
        if include_bias is not None and include_bias != True:
            kwargs['include_bias'] = include_bias
        if order is not None and order != 'C':
            kwargs['order'] = order

        HyperTransformer.__init__(self, sk_pre.PolynomialFeatures, space, name, **kwargs)


class QuantileTransformer(HyperTransformer):
    def __init__(self, n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False, subsample=int(1e5),
                 random_state=None, copy=True, space=None, name=None, **kwargs):
        if n_quantiles is not None and n_quantiles != 1000:
            kwargs['n_quantiles'] = n_quantiles
        if output_distribution is not None and output_distribution != 'uniform':
            kwargs['output_distribution'] = output_distribution
        if ignore_implicit_zeros is not None and ignore_implicit_zeros != False:
            kwargs['ignore_implicit_zeros'] = ignore_implicit_zeros
        if subsample is not None and subsample != int(1e5):
            kwargs['subsample'] = subsample
        if random_state is not None:
            kwargs['random_state'] = random_state
        if copy is not None and copy != True:
            kwargs['copy'] = copy

        HyperTransformer.__init__(self, sk_pre.QuantileTransformer, space, name, **kwargs)


class PowerTransformer(HyperTransformer):
    def __init__(self, method='yeo-johnson', standardize=True, copy=True, space=None, name=None, **kwargs):
        if method is not None and method != 'yeo-johnson':
            kwargs['method'] = method
        if standardize is not None and standardize != True:
            kwargs['standardize'] = standardize
        if copy is not None and copy != True:
            kwargs['copy'] = copy

        HyperTransformer.__init__(self, sk_pre.PowerTransformer, space, name, **kwargs)


class SimpleImputer(HyperTransformer):
    def __init__(self, missing_values=np.nan, strategy="mean", fill_value=None, verbose=0, copy=True,
                 add_indicator=False, space=None, name=None, force_output_as_float=False, **kwargs):
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

        if force_output_as_float is True:
            HyperTransformer.__init__(self, FloatOutputImputer, space, name, **kwargs)
        else:
            HyperTransformer.__init__(self, impute.SimpleImputer, space, name, **kwargs)


class PCA(HyperTransformer):
    def __init__(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None, space=None, name=None, **kwargs):
        if n_components is not None:
            kwargs['n_components'] = n_components
        if whiten is not None and whiten != False:
            kwargs['whiten'] = whiten
        if svd_solver is not None and svd_solver != 'auto':
            kwargs['svd_solver'] = svd_solver
        if tol is not None and tol != 0.0:
            kwargs['tol'] = tol
        if copy is not None and copy != True:
            kwargs['copy'] = copy
        if iterated_power is not None and iterated_power != 'auto':
            kwargs['iterated_power'] = iterated_power
        if random_state is not None:
            kwargs['random_state'] = random_state

        HyperTransformer.__init__(self, decomposition.PCA, space, name, **kwargs)


class TruncatedSVD(HyperTransformer):
    def __init__(self, n_components=2, algorithm="randomized", n_iter=5, random_state=None, tol=0., space=None,
                 name=None, **kwargs):
        if n_components is not None:
            kwargs['n_components'] = n_components
        if tol is not None and tol != 0.0:
            kwargs['tol'] = tol
        if algorithm is not None and algorithm != 'randomized':
            kwargs['algorithm'] = algorithm
        if n_iter is not None and n_iter != 5:
            kwargs['n_iter'] = n_iter
        if random_state is not None:
            kwargs['random_state'] = random_state

        HyperTransformer.__init__(self, decomposition.TruncatedSVD, space, name, **kwargs)


class SkewnessKurtosisTransformer(HyperTransformer):
    def __init__(self, transform_fn=None, space=None, name=None, **kwargs):
        if transform_fn is not None:
            kwargs['transform_fn'] = transform_fn

        HyperTransformer.__init__(self, sklearn_ex.SkewnessKurtosisTransformer, space, name, **kwargs)


class FeatureGenerationTransformer(HyperTransformer):
    def __init__(self, task=None, trans_primitives=None, fix_input=False, continuous_cols=None, datetime_cols=None,
                 max_depth=1, feature_selection_args=None, space=None, name=None, **kwargs):
        if task is not None:
            kwargs['task'] = task
        if trans_primitives is not None:
            kwargs['trans_primitives'] = trans_primitives
        if fix_input:
            kwargs['fix_input'] = fix_input
        if continuous_cols is not None:
            kwargs['continuous_cols'] = continuous_cols
        if datetime_cols is not None:
            kwargs['datetime_cols'] = datetime_cols
        if max_depth != 1:
            kwargs['max_depth'] = max_depth
        if feature_selection_args is not None:
            kwargs['feature_selection_args'] = feature_selection_args

        HyperTransformer.__init__(self, feature_generators.FeatureGenerationTransformer, space, name, **kwargs)
