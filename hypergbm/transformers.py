# -*- coding:utf-8 -*-
"""

"""

from hypernets.core.search_space import ModuleSpace, Choice
from hypernets.core.ops import ConnectionSpace
from sklearn import impute, pipeline, compose, preprocessing as sk_pre, decomposition
from . import sklearn_ex
from . import sklearn_pandas
import numpy as np


class HyperTransformer(ModuleSpace):
    def __init__(self, transformer=None, space=None, name=None, **hyperparams):
        self.transformer = transformer
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _compile(self):
        if self.transformer is not None:
            pv = self.param_values
            self.compile_fn = self.transformer(**pv)
        else:
            self.compile_fn = None

    def _forward(self, inputs):
        return self.compile_fn

    def _on_params_ready(self):
        pass


class ComposeTransformer(HyperTransformer):
    def __init__(self, space=None, name=None, **hyperparams):
        HyperTransformer.__init__(self, None, space, name, **hyperparams)

    def compose(self):
        raise NotImplementedError

    def get_transformers(self, last_module, input_id):
        transformers = []
        next = last_module
        while True:
            if next.id == input_id:
                break
            assert isinstance(next, HyperTransformer)
            if isinstance(next, ComposeTransformer):
                next, transformer = next.compose()
            else:
                transformer = (next.name, next.output)

            transformers.insert(0, transformer)
            inputs = self.space.get_inputs(next)
            if len(inputs) <= 0:
                break
            assert len(inputs) == 1, 'Pipeline does not support branching.'
            next = inputs[0]
        return next, transformers


class PipelineInput(HyperTransformer):
    def __init__(self, space=None, name=None, **hyperparams):
        HyperTransformer.__init__(self, None, space, name, **hyperparams)
        self.output_id = None


class PipelineOutput(ComposeTransformer):
    def __init__(self, pipeline_name, columns=None, space=None, name=None, **hyperparams):
        ComposeTransformer.__init__(self, space, name, **hyperparams)
        self.input_id = None
        self.pipeline_name = pipeline_name
        self.columns = columns

    def compose(self):
        inputs = self.space.get_inputs(self)
        assert len(inputs) == 1, 'Pipeline does not support branching.'
        next, steps = self.get_transformers(inputs[0], self.input_id)
        p = pipeline.Pipeline(steps)
        return next, (self.pipeline_name, p)


class Pipeline(ConnectionSpace):
    def __init__(self, module_list, columns=None, keep_link=False, space=None, name=None):
        assert isinstance(module_list, list), f'module_list must be a List.'
        assert len(module_list) > 0, f'module_list contains at least 1 Module.'
        assert all([isinstance(m, (ModuleSpace, list)) for m in
                    module_list]), 'module_list can only contains ModuleSpace or list.'
        self._module_list = module_list
        self.columns = columns
        self.hp_lazy = Choice([0])
        ConnectionSpace.__init__(self, self.pipeline_fn, keep_link, space, name, hp_lazy=self.hp_lazy)

    def pipeline_fn(self, m):
        last = self._module_list[0]
        for i in range(1, len(self._module_list)):
            self.connect_module_or_subgraph(last, self._module_list[i])
            # self._module_list[i](last)
            last = self._module_list[i]
        pipeline_input = PipelineInput(name=self.name + '_input', space=self.space)
        pipeline_output = PipelineOutput(pipeline_name=self.name, columns=self.columns, name=self.name + '_output',
                                         space=self.space)
        pipeline_input.output_id = pipeline_output.id
        pipeline_output.input_id = pipeline_input.id

        input = self.space.get_sub_graph_inputs(last)
        assert len(input) == 1, 'Pipeline does not support branching.'
        output = self.space.get_sub_graph_outputs(last)
        assert len(output) == 1, 'Pipeline does not support branching.'

        input[0](pipeline_input)
        pipeline_output(output[0])

        return pipeline_input, pipeline_output


class ColumnTransformer(ComposeTransformer):
    def __init__(self, remainder='drop', sparse_threshold=0.3, n_jobs=None, transformer_weights=None, space=None,
                 name=None, **hyperparams):
        if remainder is not None and remainder != 'drop':
            hyperparams['remainder'] = remainder
        if sparse_threshold is not None and sparse_threshold != 0.3:
            hyperparams['sparse_threshold'] = sparse_threshold
        if n_jobs is not None:
            hyperparams['n_jobs'] = n_jobs
        if transformer_weights is not None:
            hyperparams['transformer_weights'] = transformer_weights

        ComposeTransformer.__init__(self, space, name, **hyperparams)

    def compose(self):
        inputs = self.space.get_inputs(self)
        assert all([isinstance(m, PipelineOutput) for m in
                    inputs]), 'The upstream module of `ColumnTransformer` must be `Pipeline`.'
        transformers = []
        next = None
        for p in inputs:
            next, (pipeline_name, transformer) = p.compose()
            transformers.append((p.pipeline_name, transformer, p.columns))

        pv = self.param_values
        ct = compose.ColumnTransformer(transformers, **pv)
        return next, (self.name, ct)


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


class OrdinalEncoder(HyperTransformer):
    def __init__(self, categories='auto', dtype=np.float64, space=None,
                 name=None, **kwargs):
        if categories is not None and categories != 'auto':
            kwargs['categories'] = categories
        if dtype is not None and dtype != True:
            kwargs['dtype'] = dtype

        HyperTransformer.__init__(self, sk_pre.OrdinalEncoder, space, name, **kwargs)


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
                 add_indicator=False, space=None, name=None, **kwargs):
        if missing_values is not None and missing_values != np.nan:
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
