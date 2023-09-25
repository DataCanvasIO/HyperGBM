# -*- coding:utf-8 -*-
__author__ = 'yangjian'

from hypergbm.objectives import FeatureUsageObjective
from hypernets.core import Callback

"""

"""
import copy
from typing import List

import pandas as pd

from hypernets.tabular.ensemble import GreedyEnsemble
from hypernets.experiment import make_experiment as _make_experiment, ExperimentCallback
from hypernets.experiment import default_experiment_callbacks as default_experiment_callbacks_
from hypernets.experiment import default_search_callbacks as default_search_callbacks_

from hypernets.tabular import get_tool_box
from hypernets.utils import DocLens, isnotebook, logging

from hypergbm.hyper_gbm import HyperGBMShapExplainer, HyperGBMEstimator

try:
    import shap
    from shap import TreeExplainer, KernelExplainer, Explainer
    has_shap = True
except:
    has_shap = False

logger = logging.get_logger(__name__)


def make_experiment(train_data,
                    target=None,
                    eval_data=None,
                    test_data=None,
                    task=None,
                    id=None,
                    callbacks=None,
                    searcher=None,
                    search_space=None,
                    search_space_options=None,
                    search_callbacks=None,
                    objectives=None,
                    early_stopping_rounds=10,
                    early_stopping_time_limit=3600,
                    early_stopping_reward=None,
                    reward_metric=None,
                    optimize_direction=None,
                    estimator_early_stopping_rounds=None,
                    hyper_model_cls=None,
                    clear_cache=None,
                    discriminator=None,
                    log_level=None,
                    webui=False,
                    webui_options=None,
                    **kwargs):
    """
    Utility to make CompeteExperiment instance with HyperGBM.

    Parameters
    ----------

    Returns
    -------
    Runnable experiment object

    Notes:
    -------
    Initlialize Dask default client to enable dask in experiment.

    Examples:
    -------
    Create experiment with csv data file '/opt/data01/test.csv', and run it
    >>> experiment = make_experiment('/opt/data01/test.csv', target='y')
    >>> estimator = experiment.run()

    Create experiment with csv data file '/opt/data01/test.csv' with INFO logging, and run it
    >>> import logging
    >>>
    >>> experiment = make_experiment('/opt/data01/test.csv', target='y', log_level=logging.INFO)
    >>> estimator = experiment.run()

    or
    >>> experiment = make_experiment('/opt/data01/test.csv', target='y', log_level='info')
    >>> estimator = experiment.run()

    """
    # Create experiment with parquet data files '/opt/data02/*.parquet', and run it with Dask
    # >>> from dask.distributed import Client
    # >>>
    # >>> client = Client()
    # >>> experiment = make_experiment('/opt/data02/*.parquet', target='y')
    # >>> estimator = experiment.run()

    assert train_data is not None, 'train_data is required.'
    assert eval_data is None or type(eval_data) is type(train_data)
    assert test_data is None or type(test_data) is type(train_data)

    if log_level is None:
        log_level = logging.WARN
    logging.set_level(log_level)

    data_adaption_target = kwargs.get('data_adaption_target')
    data_adaption_to_cuml = str(data_adaption_target).lower() in {'cuml', 'cudf', 'cuda', 'gpu'}
    if data_adaption_to_cuml:
        import cudf
        assert isinstance(train_data, (str, pd.DataFrame, cudf.DataFrame)), \
            'Only pandas or cudf DataFrame can be adapted to cuml'

    if isinstance(train_data, str):
        tb = get_tool_box(pd.DataFrame)
    else:
        tb = get_tool_box(train_data)
    
    if hyper_model_cls is None:
       if data_adaption_to_cuml or tb.__name__.lower().find('cuml') >= 0:
           from hypergbm.cuml import CumlHyperGBM
           hyper_model_cls = CumlHyperGBM
       else:
           from hypergbm.hyper_gbm import HyperGBM
           hyper_model_cls = HyperGBM

    def default_search_space():
        args = search_space_options if search_space_options is not None else {}
        if estimator_early_stopping_rounds is not None:
            assert isinstance(estimator_early_stopping_rounds, int), \
                f'estimator_early_stopping_rounds should be int or None, {estimator_early_stopping_rounds} found.'
            args['early_stopping_rounds'] = estimator_early_stopping_rounds

        for key in ('n_estimators', 'class_balancing'):
            if key in kwargs.keys():
                args[key] = kwargs.pop(key)

        for key in ('verbose',):
            if key in kwargs.keys():
                args[key] = kwargs.get(key)

        if tb.__name__.lower().find('dask') >= 0:
            from hypergbm.dask.search_space import search_space_general as dask_search_space
            result = dask_search_space
        elif data_adaption_to_cuml or tb.__name__.lower().find('cuml') >= 0:
            from hypergbm.cuml import search_space_general as cuml_search_space
            result = cuml_search_space
        else:
            from hypergbm.search_space import search_space_general as sk_search_space
            result = sk_search_space

        if args:
            result = copy.deepcopy(result)
            result.options.update(args)
        return result

    if (searcher is None or isinstance(searcher, str)) and search_space is None:
        search_space = default_search_space()
        if isinstance(train_data, pd.DataFrame) and search_space.enable_catboost:
            tb = get_tool_box(pd.DataFrame)
            mem_usage = tb.memory_usage(train_data, test_data, eval_data)
            mem_free = tb.memory_free()
            if mem_usage / (mem_free + mem_usage) > 0.03:
                search_space = copy.deepcopy(search_space)
                catboost_init_kwargs = search_space.options.get('catboost_init_kwargs', {})
                catboost_init_kwargs['max_ctr_complexity'] = 1  # reduce training memory
                search_space.options['catboost_init_kwargs'] = catboost_init_kwargs
            logger.info(f'search space options: {search_space.options}')

    def is_notebook_widget_ready():
        try:
            import hboard_widget
            return True
        except:
            return False

    def is_webui_ready():
        try:
            import hboard
            return True
        except:
            return False

    def default_experiment_callbacks():
        if isnotebook():
            if is_notebook_widget_ready():
                from hypergbm.experiment_callbacks import create_notebook_experiment_callback
                cbs = [create_notebook_experiment_callback()]
            else:
                logger.info("you can install experiment notebook widget by command "
                            "\"pip install hboard-widget\" for better user experience in jupyter notebook")
                cbs = default_experiment_callbacks_()
        else:
            cbs = default_experiment_callbacks_()
        return cbs

    def default_search_callbacks():
        if isnotebook() and is_notebook_widget_ready():
            from hypergbm.experiment_callbacks import create_notebook_hyper_model_callback
            cbs = [create_notebook_hyper_model_callback()]
        else:
            cbs = default_search_callbacks_()
        return cbs

    if callbacks is None:
        callbacks: List[ExperimentCallback] = default_experiment_callbacks()

    if search_callbacks is None:
        search_callbacks: List[Callback] = default_search_callbacks()

    if webui:
        if webui_options is None:
            webui_options = {}
        if is_webui_ready():
            from hypergbm.experiment_callbacks import create_web_vis_hyper_model_callback, \
                create_web_vis_experiment_callback
            search_callbacks.append(create_web_vis_hyper_model_callback())
            callbacks.append(create_web_vis_experiment_callback(**webui_options))
        else:
            logger.warning("No web visualization module detected, please install by command:"
                           "\"pip install hboard\"")

    # objectives
    objectives_new = None if objectives is None else \
        list(map(lambda _: FeatureUsageObjective() if _ == 'feature_usage' else _, objectives))

    experiment = _make_experiment(hyper_model_cls, train_data,
                                  target=target,
                                  eval_data=eval_data,
                                  test_data=test_data,
                                  task=task,
                                  id=id,
                                  callbacks=callbacks,
                                  searcher=searcher,
                                  search_space=search_space,
                                  search_callbacks=search_callbacks,
                                  objectives=objectives_new,
                                  early_stopping_rounds=early_stopping_rounds,
                                  early_stopping_time_limit=early_stopping_time_limit,
                                  early_stopping_reward=early_stopping_reward,
                                  reward_metric=reward_metric,
                                  optimize_direction=optimize_direction,
                                  clear_cache=clear_cache,
                                  discriminator=discriminator,
                                  log_level=log_level,
                                  **kwargs
                                  )
    return experiment


_search_space_doc = """
    default is hypergbm.search_space.search_space_general (if Dask isn't enabled)
    or hypergbm.dask.search_space.search_space_general (if Dask is enabled)."""

_class_balancing_doc = """ : str, optional, (default=None)
    Strategy for imbalanced learning (classification task only).  Possible values:
        - ClassWeight
        - RandomOverSampler
        - SMOTE
        - ADASYN
        - RandomUnderSampler
        - NearMiss
        - TomekLinks
        - EditedNearestNeighbours"""

_cross_validator_doc = """ : cross-validation generator, optional
    Used to split a fit_transformed dataset into a sequence of train and test portions.
    KFold for regression task and StratifiedKFold for classification task by default."""

_estimator_early_stopping_rounds_doc = """ int, optional, (default=None)
    Estimator *early_stopping_rounds* option, inferred from *n_estimators* by default."""

_webui_doc = """ : bool (default False),
    Whether to start the experiment visualization web server
"""

_webui_options_doc = """ : dict, optional, (default None),
    Dictionary of parameters to initialize the `LogEventExperimentCallback` instance.
    If None, will be initialized the instance with default values.
"""

_skip_if_file_doc = """ : str, optional, (default None),
    Skip current trial if the file was found and could be removed.
"""


def _merge_doc():
    my_doc = DocLens(make_experiment.__doc__)
    params = DocLens(_make_experiment.__doc__).parameters
    params.pop('hyper_model_cls')
    params['search_space'] += _search_space_doc
    params['class_balancing'] = _class_balancing_doc
    params['cross_validator'] = _cross_validator_doc
    params['estimator_early_stopping_rounds'] = _estimator_early_stopping_rounds_doc
    params['skip_if_file'] = _skip_if_file_doc

    params['webui'] = _webui_doc
    params['webui_options'] = _webui_options_doc

    for k in ['clear_cache', 'log_level']:
        params.move_to_end(k)

    my_doc.parameters = params

    make_experiment.__doc__ = my_doc.render()


_merge_doc()


class PipelineKernelExplainer:

    def __init__(self, pipeline_model, data, **kwargs):
        """SHAP Kernel Explainer for HyperGBM pipeline model

          Parameters
          ----------
          pipeline_model: sklearn.pipeline.Pipeline, required
              hypergbm pipeline model, training by CompeteExperiment

          data: pd.DataFrame, optional
              the background dataset to use for integrating out features, recommend less than 100 rows samples.

          kwargs: params for shap.KernelExplainer
          """

        if not has_shap:
            raise RuntimeError('Please install `shap` package first. command: pip install shap')

        self.pipeline_model = pipeline_model
        self.hypergbm_explainers = None
        self.data_columns = None

        if hasattr(pipeline_model, 'predict_proba'):
            pred_f = pipeline_model.predict_proba  # output shape values shape = (n_output, n_rows, n_cols)
        else:
            pred_f = pipeline_model.predict  # regression, output shape values shape = (n_rows, n_cols)

        self.data_columns = data.columns
        self._explainer = KernelExplainer(pred_f, data=data, keep_index=True, **kwargs)

    def __call__(self, X, **kwargs):
        """Calc explanation of X using shap kernel method.

        Parameters
        ----------
        X
        kwargs

        Returns
        -------
            For classification task, output type is List[Explanation], length is `n_classes` in the model,
                and shape of each element is equal to X.shape.
            For regression task, output type is Explanation, shape is equal to X.shape
        """

        explainer = self._explainer
        shap_values_data = explainer.shap_values(X, **kwargs)
        from shap._explanation import Explanation
        if isinstance(shap_values_data, list):  # usually for classification
            el_list = []
            for i, shap_values in enumerate(shap_values_data):
                el = Explanation(shap_values, base_values=explainer.expected_value[i], data=X.values,
                                 feature_names=self.data_columns.tolist())
                el_list.append(el)
            return el_list
        else:
            return Explanation(shap_values_data, base_values=explainer.expected_value,
                               data=X.values, feature_names=self.data_columns.tolist())


class PipelineTreeExplainer:

    def __init__(self, pipeline_model, data=None, model_indexes=None, **kwargs):
        """SHAP Tree Explainer for HyperGBM pipeline model

          Parameters
          ----------
          pipeline_model: sklearn.pipeline.Pipeline, required
              hypergbm pipeline model, training by CompeteExperiment

          data: pd.DataFrame, optional
              the background dataset to use for integrating out features.

          model_indexes: model indexes in GreedEnsemble, default is the first model.

          kwargs: params for HyperGBMShapExplainer
          """
        if not has_shap:
            raise RuntimeError('Please install `shap` package first. command: pip install shap')

        self.pipeline_model = pipeline_model

        last_step = pipeline_model.steps[-1][1]

        if data is not None:
            data = self._transform(data)

        if isinstance(last_step, GreedyEnsemble):
            self._is_ensemble = True
            hypergbm_explainers = []
            if model_indexes is None:
                model_indexes = [0]
            for model_index in model_indexes:
                estimator = last_step.estimators[model_index]
                if estimator is not None:
                    hypergbm_explainers.append(HyperGBMShapExplainer(estimator, data=data, **kwargs))
                else:
                    logger.warning(f"Index of {model_index} is None ")
                    hypergbm_explainers.append(None)
            self._hypergbm_explainers = hypergbm_explainers

        elif isinstance(last_step, HyperGBMEstimator):
            if model_indexes is not None:
                logger.warning(f"model indexes is ignored since this is not a ensemble model.")

            self._is_ensemble = False
            self._hypergbm_explainers = [HyperGBMShapExplainer(last_step, data=data, **kwargs)]
        else:
            raise RuntimeError(f"Unseen estimator type {type(last_step)}")

    def _transform(self, X):
        Xt = X
        #  "pipeline_model.transform" requires a passthrough estimator , it's not available for hypergbm pipeline
        for i, _, transform in self.pipeline_model._iter(with_final=False):
            Xt = transform.transform(Xt)
        return Xt

    def __call__(self, X, **kwargs):
        Xt = self._transform(X)
        if self._is_ensemble is True:
            return [_(Xt, **kwargs) if _ is not None else None for _ in self._hypergbm_explainers]
        else:
            return self._hypergbm_explainers[0](Xt, **kwargs)
