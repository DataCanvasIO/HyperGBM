# -*- coding:utf-8 -*-
__author__ = 'yangjian'

"""

"""

from sklearn.metrics import get_scorer

from hypergbm.hyper_gbm import HyperGBM
from hypernets.experiment import CompeteExperiment
from hypernets.searchers import make_searcher
from hypernets.tabular import dask_ex as dex
from hypernets.tabular.metrics import metric_to_scoring
from hypernets.utils import load_data, infer_task_type, hash_data, logging, const

logger = logging.get_logger(__name__)

DEFAULT_TARGET_SET = {'y', 'target'}


def make_experiment(train_data,
                    target=None,
                    eval_data=None,
                    test_data=None,
                    task=None,
                    id=None,
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
    use_cache : bool, optional, (default True if Dask is not enabled, else False)
    clear_cache: bool, optional, (default True)
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

    assert train_data is not None, 'train data is required.'

    kwargs = kwargs.copy()
    dask_enable = dex.exist_dask_object(train_data, test_data, eval_data) or dex.dask_enabled()

    if log_level is None:
        log_level = logging.WARN
    logging.set_level(log_level)

    def find_target(df):
        columns = df.columns.to_list()
        for col in columns:
            if col.lower() in DEFAULT_TARGET_SET:
                return col
        raise ValueError(f'Not found one of {DEFAULT_TARGET_SET} from your data, implicit target must be specified.')

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

        if dask_enable:
            from hypergbm.dask.search_space import search_space_general as dask_search_space
            return lambda: dask_search_space(**args)
        else:
            from hypergbm.search_space import search_space_general as sk_search_space
            return lambda: sk_search_space(**args)

    def default_searcher(cls):
        search_space_fn = search_space if search_space is not None \
            else default_search_space()
        op = optimize_direction if optimize_direction is not None \
            else 'max' if scorer._sign > 0 else 'min'

        s = make_searcher(cls, search_space_fn, optimize_direction=op)

        return s

    def to_search_object(sch):
        from hypernets.core.searcher import Searcher as SearcherSpec
        from hypernets.searchers import EvolutionSearcher

        if sch is None:
            sch = default_searcher(EvolutionSearcher)
        elif isinstance(sch, (type, str)):
            sch = default_searcher(sch)
        elif not isinstance(sch, SearcherSpec):
            logger.warning(f'Unrecognized searcher "{sch}".')

        return sch

    def default_search_callbacks():
        from hypernets.core.callbacks import SummaryCallback
        if logging.get_level() < logging.WARN:
            callbacks = [SummaryCallback()]
        else:
            callbacks = []
        return callbacks

    def append_early_stopping_callbacks(callbacks):
        from hypernets.core.callbacks import EarlyStoppingCallback

        assert isinstance(callbacks, (tuple, list))
        if any([isinstance(cb, EarlyStoppingCallback) for cb in callbacks]):
            return callbacks

        op = optimize_direction if optimize_direction is not None \
            else 'max' if scorer._sign > 0 else 'min'
        es = EarlyStoppingCallback(early_stopping_rounds, op,
                                   time_limit=early_stopping_time_limit,
                                   expected_reward=early_stopping_reward)

        return [es] + callbacks

    X_train, X_eval, X_test = [load_data(data) if data is not None else None
                               for data in (train_data, eval_data, test_data)]

    X_train, X_eval, X_test = [dex.reset_index(x) if dex.is_dask_dataframe(x) else x
                               for x in (X_train, X_eval, X_test)]

    if target is None:
        target = find_target(X_train)

    y_train = X_train.pop(target)
    y_eval = X_eval.pop(target) if X_eval is not None else None

    if task is None:
        task, _ = infer_task_type(y_train)

    if reward_metric is None:
        reward_metric = 'rmse' if task == const.TASK_REGRESSION else 'accuracy'
        logger.info(f'no reward metric specified, use "{reward_metric}" for {task} task by default.')

    scorer = metric_to_scoring(reward_metric) if kwargs.get('scorer') is None else kwargs.get('scorer')
    if isinstance(scorer, str):
        scorer = get_scorer(scorer)

    searcher = to_search_object(searcher)

    if search_callbacks is None:
        search_callbacks = default_search_callbacks()
    search_callbacks = append_early_stopping_callbacks(search_callbacks)

    if id is None:
        id = hash_data(dict(X_train=X_train, y_train=y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval,
                            eval_size=kwargs.get('eval_size'), target=target, task=task))
        id = f'hypergbm_{id}'
    default_cache_dir = f'{id}/cache'

    hm = HyperGBM(searcher, reward_metric=reward_metric, callbacks=search_callbacks,
                  cache_dir=kwargs.pop('cache_dir', default_cache_dir),
                  clear_cache=clear_cache if clear_cache is not None else True)

    use_cache = not dex.exist_dask_object(X_train, X_test, X_eval) if use_cache is None else bool(use_cache)

    experiment = CompeteExperiment(hm, X_train, y_train, X_eval=X_eval, y_eval=y_eval, X_test=X_test,
                                   task=task, id=id, scorer=scorer, use_cache=use_cache,
                                   **kwargs)

    if logger.is_info_enabled():
        train_shape, test_shape, eval_shape = \
            dex.compute(X_train.shape,
                        X_test.shape if X_test is not None else None,
                        X_eval.shape if X_eval is not None else None,
                        traverse=True)
        logger.info(f'make_experiment with train data:{train_shape}, '
                    f'test data:{test_shape}, eval data:{eval_shape}, target:{target}')

    return experiment
