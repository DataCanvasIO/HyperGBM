# -*- coding:utf-8 -*-
__author__ = 'yangjian'

"""

"""
import copy

from hypergbm.hyper_gbm import HyperGBM
from hypernets.experiment import make_experiment as _make_experiment
from hypernets.tabular.dask_ex import DaskToolBox
from hypernets.utils import DocLens


def make_experiment(train_data,
                    target=None,
                    eval_data=None,
                    test_data=None,
                    task=None,
                    id=None,
                    callbacks=None,
                    searcher=None,
                    search_space=None,
                    search_callbacks=None,
                    early_stopping_rounds=10,
                    early_stopping_time_limit=3600,
                    early_stopping_reward=None,
                    reward_metric=None,
                    optimize_direction=None,
                    estimator_early_stopping_rounds=None,
                    use_cache=None,
                    clear_cache=None,
                    discriminator=None,
                    log_level=None,
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

    def default_search_space():
        args = {}
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

        dask_enable = DaskToolBox.exist_dask_object(train_data, test_data, eval_data) or DaskToolBox.dask_enabled()
        if dask_enable:
            from hypergbm.dask.search_space import search_space_general as dask_search_space
            result = copy.deepcopy(dask_search_space)
        else:
            from hypergbm.search_space import search_space_general as sk_search_space
            result = copy.deepcopy(sk_search_space)

        if args:
            result.options.update(args)
        return result

    if (searcher is None or isinstance(searcher, str)) and search_space is None:
        search_space = default_search_space()

    experiment = _make_experiment(HyperGBM, train_data,
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


def _merge_doc():
    my_doc = DocLens(make_experiment.__doc__)
    params = DocLens(_make_experiment.__doc__).parameters
    params.pop('hyper_model_cls')
    params['search_space'] += _search_space_doc
    params['class_balancing'] = _class_balancing_doc
    params['cross_validator'] = _cross_validator_doc
    for k in ['clear_cache', 'log_level']:
        params.move_to_end(k)

    my_doc.parameters = params

    make_experiment.__doc__ = my_doc.render()


_merge_doc()
