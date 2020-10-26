# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from sklearn import pipeline, compose

from tabular_toolbox import dataframe_mapper
from hypernets.core.ops import ConnectionSpace
from hypernets.core.search_space import ModuleSpace, Choice


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
        ct = dataframe_mapper.DataFrameMapper(features=transformers, **pv)
        return next, (self.name, ct)
