# -*- coding:utf-8 -*-
__author__ = 'yangjian'

"""

"""

from hypergbm.hyper_gbm import HyperGBM
from hypernets.experiment import make_experiment as _make_experiment
from hypernets.tabular import dask_ex as dex


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

    Parameters
    ----------
    train_data : str, Pandas or Dask DataFrame
        Feature data for training with target column.
        For str, it's should be the data path in file system,
        we'll detect data format from this path (only .csv and .parquet are supported now) and read it.
    target : str, optional
        Target feature name for training, which must be one of the drain_data columns, default is 'y'.
    eval_data : str, Pandas or Dask DataFrame, optional
        Feature data for evaluation with target column.
        For str, it's should be the data path in file system,
        we'll detect data format from this path (only .csv and .parquet are supported now) and read it.
    test_data : str, Pandas or Dask DataFrame, optional
        Feature data for testing without target column.
        For str, it's should be the data path in file system,
        we'll detect data format from this path (only .csv and .parquet are supported now) and read it.
    task : str or None, (default=None)
        Task type(*binary*, *multiclass* or *regression*).
        If None, inference the type of task automatically
    id : str or None, (default=None)
        The experiment id.
    callbacks: list of ExperimentCallback, optional
        ExperimentCallback list.
    searcher : str, searcher class, search object, optional
        The hypernets Searcher instance to explore search space, default is EvolutionSearcher instance.
        For str, should be one of 'evolution', 'mcts', 'random'.
        For class, should be one of EvolutionSearcher, MCTSSearcher, RandomSearcher, or subclass of hypernets Searcher.
        For other, should be instance of hypernets Searcher.
    search_space : callable, optional
        Used to initialize searcher instance (if searcher is None, str or class),
        default is hypergbm.search_space.search_space_general (if Dask isn't enabled)
        or hypergbm.dask.search_space.search_space_general (if Dask is enabled) .
    search_callbacks
        Hypernets search callbacks, used to initialize searcher instance (if searcher is None, str or class).
        If log_level >= WARNNING, default is EarlyStoppingCallback only.
        If log_level < WARNNING, defalult is EarlyStoppingCallback plus SummaryCallback.
    early_stopping_rounds :　int, optional
        Setting of EarlyStoppingCallback, is used if EarlyStoppingCallback instance not found from search_callbacks.
        Set zero or None to disable it, default is 10.
    early_stopping_time_limit : int, optional
        Setting of EarlyStoppingCallback, is used if EarlyStoppingCallback instance not found from search_callbacks.
        Set zero or None to disable it, default is 3600 seconds.
    early_stopping_reward : float, optional
        Setting of EarlyStoppingCallback, is used if EarlyStoppingCallback instance not found from search_callbacks.
        Set zero or None to disable it, default is None.
    reward_metric : str, callable, optional, (default 'accuracy' for binary/multicalss task, 'rmse' for regression task)
        Hypernets search reward metric name or callable. Possible values:
            - accuracy
            - auc
            - f1
            - logloss
            - mse
            - mae
            - msle
            - precision
            - rmse
            - r2
            - recall
    optimize_direction : str, optional
        Hypernets search reward metric direction, default is detected from reward_metric.
    estimator_early_stopping_rounds : int or None, optional
        Esitmator fit early_stopping_rounds option.
    discriminator : instance of hypernets.discriminator.BaseDiscriminator, optional
        Discriminator is used to determine whether to continue training
    use_cache : bool, optional, (deprecated)
    clear_cache: bool, optional, (default False)
    log_level : int, str, or None, (default=None),
        Level of logging, possible values:
            -logging.CRITICAL
            -logging.FATAL
            -logging.ERROR
            -logging.WARNING
            -logging.WARN
            -logging.INFO
            -logging.DEBUG
            -logging.NOTSET
    kwargs:
        Parameters to initialize experiment instance, refrence CompeteExperiment for more details.
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

        dask_enable = dex.exist_dask_object(train_data, test_data, eval_data) or dex.dask_enabled()
        if dask_enable:
            from hypergbm.dask.search_space import search_space_general as dask_search_space
            return lambda: dask_search_space(**args)
        else:
            from hypergbm.search_space import search_space_general as sk_search_space
            return lambda: sk_search_space(**args)

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
