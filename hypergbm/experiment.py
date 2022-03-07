# -*- coding:utf-8 -*-
__author__ = 'yangjian'

from hypernets.core import Callback

"""

"""
import copy
from typing import List

import pandas as pd

from hypernets.experiment import make_experiment as _make_experiment, ExperimentCallback
from hypernets.experiment import default_experiment_callbacks as default_experiment_callbacks_
from hypernets.experiment import default_search_callbacks as default_search_callbacks_

from hypernets.tabular import get_tool_box
from hypernets.utils import DocLens, isnotebook, logging

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
                    early_stopping_rounds=10,
                    early_stopping_time_limit=3600,
                    early_stopping_reward=None,
                    reward_metric=None,
                    optimize_direction=None,
                    estimator_early_stopping_rounds=None,
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

    Create experiment with parquet data files '/opt/data02/*.parquet', and run it with Dask
    >>> from dask.distributed import Client
    >>>
    >>> client = Client()
    >>> experiment = make_experiment('/opt/data02/*.parquet', target='y')
    >>> estimator = experiment.run()

    """

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


def _merge_doc():
    my_doc = DocLens(make_experiment.__doc__)
    params = DocLens(_make_experiment.__doc__).parameters
    params.pop('hyper_model_cls')
    params['search_space'] += _search_space_doc
    params['class_balancing'] = _class_balancing_doc
    params['cross_validator'] = _cross_validator_doc
    params['estimator_early_stopping_rounds'] = _estimator_early_stopping_rounds_doc

    params['webui'] = _webui_doc
    params['webui_options'] = _webui_options_doc

    for k in ['clear_cache', 'log_level']:
        params.move_to_end(k)

    my_doc.parameters = params

    make_experiment.__doc__ = my_doc.render()


_merge_doc()
