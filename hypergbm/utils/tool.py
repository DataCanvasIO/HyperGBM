import argparse
import math
import os
import pickle
import re
import sys
from multiprocessing import Process

import pandas as pd
import psutil

import hypergbm
from hypergbm.cfg import HyperGBMCfg as cfg
from hypernets.experiment import ExperimentCallback, StepNames
from hypernets.tabular import get_tool_box as get_tool_box_, is_dask_installed, is_cuml_installed
from hypernets.utils import const, logging, dump_perf

metric_choices = ['accuracy', 'auc', 'f1', 'logloss', 'mse', 'mae', 'msle', 'precision', 'rmse', 'r2', 'recall']
strategy_choices = ['threshold', 'quantile', 'number']

estimators = ['lightgbm', 'xgboost', 'catboost', 'histgb']


class FreeDataCallback(ExperimentCallback):
    def step_end(self, exp, step, output, elapsed):
        if step == StepNames.DATA_ADAPTION:
            exp.X_train = None
            exp.y_train = None
            exp.X_eval = None
            exp.y_eval = None
            exp.X_test = None


def to_bool(v):
    if isinstance(v, bool) or v is None:
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Unexpected boolean value: {v}.')


def setup_dask(overload):
    from dask.distributed import LocalCluster, Client

    # client = Client(LocalCluster(processes=False))
    # client = Client(LocalCluster(processes=True, n_workers=20, threads_per_worker=4))
    # client = Client(LocalCluster(processes=True, n_workers=20, threads_per_worker=4, memory_limit='30GB'))
    # client = Client(LocalCluster(processes=True, n_workers=4, threads_per_worker=4, memory_limit='10GB'))

    if os.environ.get('DASK_SCHEDULER_ADDRESS') is not None:
        # use dask default settings
        client = Client()
    else:
        # start local cluster
        cores = psutil.cpu_count()
        workers = math.ceil(cores / 3)
        if workers > 1:
            if overload <= 0:
                overload = 1.0
            mem_total = psutil.virtual_memory().available / 1024 / 1024 / 1024  # GB
            mem_per_worker = math.ceil(mem_total / workers * overload)
            if mem_per_worker > mem_total:
                mem_per_worker = mem_total
            cluster = LocalCluster(processes=True, n_workers=workers, threads_per_worker=4,
                                   memory_limit=f'{mem_per_worker}GB')
        else:
            cluster = LocalCluster(processes=False)

        client = Client(cluster)
    return client


def create_dump_perf_process(perf_file, recursive=True, interval=1):
    proc = Process(target=dump_perf, args=(perf_file, os.getpid(), recursive, interval), daemon=True)

    return proc


def get_tool_box(args):
    if args.enable_dask:
        import dask.dataframe as dd
        tb = get_tool_box_(dd.DataFrame)
    elif args.enable_gpu and is_cuml_installed:
        import cudf
        tb = get_tool_box_(cudf.DataFrame)
    else:
        tb = get_tool_box_(pd.DataFrame)
    return tb


def main(argv=None):
    def setup_train_args(a):
        tg = a.add_argument_group('Training settings')
        tg.add_argument('--train-data', '--train-file', '--train', type=str, default=None, required=True,
                        help='the file path of training data, .csv and .parquet files are supported.')
        tg.add_argument('--eval-data', '--eval-file', '--eval', type=str, default=None,
                        help='the file path of evaluation data, .csv and .parquet files are supported.')
        tg.add_argument('--test-data', '--test-file', '--test', type=str, default=None,
                        help='the file path of testing data, .csv and .parquet files are supported.')
        tg.add_argument('--train-test-split-strategy', type=str, default=None,
                        choices=[None, 'adversarial_validation'],
                        help=None)
        tg.add_argument('--target', '--y', type=str, default=None,
                        help='training target feature name, default is %(default)s')
        tg.add_argument('--task', type=str, default=None, choices=['binary', 'multiclass', 'regression'],
                        help='train task type, will be detected from target by default')
        tg.add_argument('--max-trials', '--trials', '--n', type=int, default=10,
                        help='search trial number limit, default %(default)s')
        tg.add_argument('--reward-metric', '--reward', '--metric', type=str, default=None, metavar='METRIC',
                        choices=metric_choices,
                        help='search reward metric name, one of [%(choices)s], default %(default)s')
        tg.add_argument('--cv', type=to_bool, default=True,
                        help='enable to disable cross validation, default %(default)s')
        tg.add_argument('-cv', '-cv+', dest='cv', action='store_true',
                        help='alias of "--cv true"')
        tg.add_argument('-cv-', dest='cv', action='store_false',
                        help='alias of "--cv false"')
        tg.add_argument('--cv-num-folds', '--num-folds', dest='num_folds', type=int, default=3,
                        help='fold number of cv, default %(default)s')
        tg.add_argument('--pos-label', type=str, default=None,
                        help='pos label')
        tg.add_argument('--down_sample_search', type=to_bool, default=False,
                        help='enable down_sample_search, default %(default)s')
        tg.add_argument('--down_sample_search_size', type=float, default=0.1,
                        help='down_sample_search_size, default %(default)s')
        tg.add_argument('--down_sample_search_time_limit', type=int, default=None,
                        help='down_sample_search_time_limit, default %(default)s')
        tg.add_argument('--down_sample_search_max_trials', type=int, default=None,
                        help='down_sample_search_max_trials, default %(default)s')
        tg.add_argument('--ensemble-size', '--ensemble', type=int, default=20,
                        help='how many estimators are involved, set "0" to disable ensemble, default %(default)s')
        tg.add_argument('--random-state', type=int, default=None,
                        help='random state seed (int), default %(default)s')
        tg.add_argument('--model-file', '--model', type=str, default='model.pkl',
                        help='the output pickle file name for trained model, default %(default)s')
        tg.add_argument('--as-local', type=to_bool, default=True,
                        help='output the model file as local (only if GPU is enabled), default %(default)s')
        tg.add_argument('--history-file', '--history', type=str, default=None,
                        help='the output file name for search history, default %(default)s')

        for est in estimators:
            tg.add_argument(f'--{est}', type=to_bool, default=getattr(cfg, f'estimator_{est}_enabled', True),
                            help=f'enable {est} or not, default %(default)s')
            tg.add_argument(f'-{est}', f'-{est}+', dest=est, action='store_true',
                            help=f'alias of --{est} true')
            tg.add_argument(f'-{est}-', dest=est, action='store_false',
                            help=f'alias of --{est} false')

        sg = a.add_argument_group('Searcher settings')
        sg.add_argument('--searcher', type=str, default=None,
                        choices=['mcts', 'evolution', 'random'],
                        help='Algorithm used to explore the search space, default %(default)s seconds')
        sg.add_argument('-mcts', dest='searcher', action='store_const', const='mcts',
                        help='alias of "--searcher mcts"')
        sg.add_argument('-evolution', dest='searcher', action='store_const', const='evolution',
                        help='alias of "--searcher evolution"')
        sg.add_argument('-random', dest='searcher', action='store_const', const='random',
                        help='alias of "--searcher random"')
        sg.add_argument('--trial-store', type=str, default=None,
                        help='trial store location, default %(default)s')

        eg = a.add_argument_group('Early stopping settings')
        eg.add_argument('--early-stopping-time-limit', '--es-time-limit', type=int, default=3600,
                        help='time limit to search and train, set 0 to disable, default %(default)s seconds')
        eg.add_argument('--early-stopping-rounds', '--es-rounds', type=int, default=10,
                        help='default %(default)s ')
        eg.add_argument('--early-stopping-reward', '--es-reward', type=float, default=0.0,
                        help='default %(default)s')
        eg.add_argument('--estimator_early_stopping_rounds', type=int, default=None,
                        help='default %(default)s')

        fg = a.add_argument_group('Data adaption')
        fg.add_argument('--data-adaption', type=to_bool, default=None,
                        help='Enable/disable data adaption, default %(default)s')
        fg.add_argument('-da', '-da+', '-data-adaption', '-data-adaption+', dest='data_adaption', action='store_true',
                        help='alias of "--data-adaption true"')
        fg.add_argument('-da-', '-data-adaption-', dest='data_adaption', action='store_false',
                        help='alias of "--data-adaption false"')
        fg.add_argument('--data-adaption-memory-limit', type=float, default=0.05,
                        help='proportion of the system free memory, default %(default)s')
        fg.add_argument('--data-adaption-min-cols', type=float, default=0.3,
                        help='proportion of the original dataframe column number, default %(default)s')

        fg = a.add_argument_group('Feature generation')
        fg.add_argument('--feature-generation', type=to_bool, default=False,
                        help='Enable/disable feature generation, default %(default)s')
        fg.add_argument('-fg', '-fg+', dest='feature_generation', action='store_true',
                        help='alias of "--feature_generation true"')
        fg.add_argument('-fg-', dest='feature_generation', action='store_false',
                        help='alias of "--feature_generation false"')
        fg.add_argument('--feature-generation-categories-cols', '--fg-categories-cols',
                        type=str, default=None, nargs='*',
                        help='generate features as category column, default %(default)s')
        fg.add_argument('--feature-generation-continuous-cols', '--fg-continuous-cols',
                        type=str, default=None, nargs='*',
                        help='generate features as continuous column, default %(default)s')
        fg.add_argument('--feature-generation-datetime-cols', '--fg-datetime-cols',
                        type=str, default=None, nargs='*',
                        help='generate features as datetime column, default %(default)s')
        fg.add_argument('--feature-generation-latlong-cols', '--fg-latlong-cols',
                        type=str, default=None, nargs='*',
                        help='generate features as latlong column, default %(default)s')
        fg.add_argument('--feature-generation-text-cols', '--fg-text-cols',
                        type=str, default=None, nargs='*',
                        help='generate features as text column, default %(default)s')

        dg = a.add_argument_group('Drift detection')
        dg.add_argument('--drift-detection', type=to_bool, default=True,
                        help='Enable/disable drift detection if test data is provided, default %(default)s')
        dg.add_argument('-dd', '-dd+', dest='drift_detection', action='store_true',
                        help='alias of "--drift-detection true"')
        dg.add_argument('-dd-', dest='drift_detection', action='store_false',
                        help='alias of "--drift-detection false"')
        dg.add_argument('--drift-detection-remove-shift-variable', '--dd-remove-shift-variable',
                        type=to_bool, default=True,
                        help='default %(default)s')
        dg.add_argument('--drift-detection-variable-shift-threshold', '--dd-variable-shift-threshold',
                        type=float, default=0.7,
                        help='default %(default)s')
        dg.add_argument('--drift-detection-threshold', '--dd-threshold', type=float, default=0.6,
                        help='default default %(default)s')
        dg.add_argument('--drift-detection-remove-size', '--dd-remove-size', type=float, default=0.1,
                        help='default %(default)s')
        dg.add_argument('--drift-detection-min-features', '--dd-min-features', type=int, default=10,
                        help='default %(default)s')
        dg.add_argument('--drift-detection-num-folds', '--dd-num-folds', type=int, default=5,
                        help='default %(default)s')

        cg = a.add_argument_group('Collinearity detection')
        cg.add_argument('--collinearity-detection', type=to_bool, default=False)
        cg.add_argument('-cd', '-cd+', dest='collinearity_detection', action='store_true',
                        help='alias of "--collinearity-detection true"')
        cg.add_argument('-cd-', dest='collinearity_detection', action='store_false',
                        help='alias of "--collinearity-detection false"')

        fsg = a.add_argument_group('Feature selection')
        fsg.add_argument('--feature-selection', type=to_bool, default=False,
                         help='default %(default)s')
        fsg.add_argument('-fs', '-fs+', dest='feature_selection', action='store_true',
                         help='alias of "--feature-selection true"')
        fsg.add_argument('-fs-', dest='feature_selection', action='store_false',
                         help='alias of "--feature-selection false"')
        fsg.add_argument('--feature-selection-strategy', '--fs-strategy',
                         type=str, default=strategy_choices[0], choices=strategy_choices,
                         help='default %(default)s')
        fsg.add_argument('--feature-selection-threshold', '--fs-threshold',
                         type=float, default=0.1,
                         help='default %(default)s')
        fsg.add_argument('--feature-selection-quantile', '--fs-quantile',
                         type=float, default=0.2,
                         help='default %(default)s')
        fsg.add_argument('--feature-selection-number', '--fs-number',
                         type=float, default=0.8,
                         help='default %(default)s')

        fs2g = a.add_argument_group('Feature selection (the 2nd stage)')
        fs2g.add_argument('--feature-reselection', type=to_bool, default=False,
                          help='default %(default)s')
        fs2g.add_argument('-fs2', '-fs2+', '-pi', '-pi+', dest='feature_reselection', action='store_true',
                          help='alias of "--feature-reselection true"')
        fs2g.add_argument('-fs2-', '-fs2-', '-pi-', dest='feature_reselection', action='store_false',
                          help='alias of "--feature-reselection false"')
        fs2g.add_argument('--feature-reselection-estimator-size', '--fs2-estimator-size', '--pi-estimator-size',
                          type=int, default=10,
                          help='default %(default)s')
        fs2g.add_argument('--feature-reselection-strategy', '--fs2-strategy', '--pi-strategy',
                          type=str, default=strategy_choices[0], choices=strategy_choices,
                          help='default %(default)s')
        fs2g.add_argument('--feature-reselection-threshold', '--fs2-threshold', '--pi-threshold',
                          type=float, default=0.1,
                          help='default %(default)s')
        fs2g.add_argument('--feature-reselection-quantile', '--fs2-quantile', '--pi-quantile',
                          type=float, default=0.2,
                          help='default %(default)s')
        fs2g.add_argument('--feature-reselection-number', '--fs2-number', '--pi-number',
                          type=float, default=0.8,
                          help='default %(default)s')

        plg = a.add_argument_group('Pseudo labeling (the 2nd stage)')
        plg.add_argument('--pseudo-labeling', type=to_bool, default=False,
                         help='default %(default)s')
        plg.add_argument('-pl', '-pl+', dest='pseudo_labeling', action='store_true',
                         help='alias of "--pseudo-labeling true"')
        plg.add_argument('-pl-', dest='pseudo_labeling', action='store_false',
                         help='alias of "--pseudo-labeling false"')
        plg.add_argument('--pseudo-labeling-strategy', '--pl-strategy',
                         type=str, default=strategy_choices[0], choices=strategy_choices,
                         help='default %(default)s')
        plg.add_argument('--pseudo-labeling-proba-threshold', '--pl-threshold',
                         type=float, default=0.8,
                         help='default %(default)s')
        plg.add_argument('--pseudo-labeling-proba-quantile', '--pl-quantile',
                         type=float, default=0.8,
                         help='default %(default)s')
        plg.add_argument('--pseudo-labeling-sample-number', '--pl-number',
                         type=float, default=0.2,
                         help='default %(default)s')
        plg.add_argument('--pseudo-labeling-resplit', '--pl-resplit', type=to_bool, default=False,
                         help='default %(default)s')

        # others
        og = a.add_argument_group('Other settings')
        og.add_argument('--id', type=str, default=None,
                        help='experiment id')

        og.add_argument('--clear-cache', type=to_bool, default=None)
        og.add_argument('-clear-cache', '-clear-cache+', dest='clear_cache', action='store_true',
                        help='alias of "--clear-cache true"')
        og.add_argument('-clear-cache-', dest='clear_cache', action='store_false',
                        help='alias of "--clear-cache false"')

    def setup_evaluate_args(a):
        a.add_argument('--eval-data', '--eval-file', '--eval', '--data',
                       type=str, required=True,
                       help='the file path of evaluation data, .csv and .parquet files are supported.')
        a.add_argument('--target', '--y', type=str, default='y',
                       help='target feature name, default is %(default)s')
        a.add_argument('--model-file', '--model', default='model.pkl',
                       help='the pickle file name for trained model, default %(default)s')
        a.add_argument('--metric', '--metrics', type=str, default=None, nargs='*', metavar='METRIC',
                       choices=metric_choices,
                       help='metric name list, one or more of [%(choices)s], default detected from task type.')
        a.add_argument('--threshold', type=float, default=0.5,
                       help=f'probability threshold to detect pos label, '
                            f'use when task="{const.TASK_BINARY}" only, default %(default)s')
        a.add_argument('--pos-label', type=str, default=None,
                       help='pos label')

    def setup_predict_args(a):
        a.add_argument('--data', '--data-file', type=str, required=True,
                       help='the data path of to predict, .csv and .parquet files are supported.')
        a.add_argument('--model-file', '--model', default='model.pkl',
                       help='the pickle file name for trained model, default %(default)s')
        a.add_argument('--proba', type=to_bool, default=False,
                       help='predict probability instead of target, default %(default)s')
        a.add_argument('-proba', '-proba+', dest='proba', action='store_true',
                       help='alias of "--proba true"')
        a.add_argument('--threshold', type=float, default=0.5,
                       help=f'probability threshold to detect pos label, '
                            f'use when task="{const.TASK_BINARY}" only, default %(default)s')

        a.add_argument('--target', '--y', type=str, default='y',
                       help='target feature name for output, default is %(default)s')
        a.add_argument('--output', '--output-file', type=str, default='prediction.csv',
                       help='the output file name, default is %(default)s')
        a.add_argument('--output-with-data', '--with-data', type=str, default=None, nargs='*',
                       help='column name patterns stored with prediction result, '
                            '"*" for all columns, default %(default)s')
        a.add_argument('-output-with-data', '-with-data', dest='output_with_data',
                       action='store_const', const=['*'],
                       help='alias of "--output-with-data *"')

    def setup_global_args(a):
        a.add_argument('--version', type=to_bool, default=False,
                       help='print version number')
        a.add_argument('-version', '-version+', dest='version', action='store_true',
                       help='alias of "--version true"')
        a.add_argument('--n-jobs', type=int, default=None,
                       help='the number of parallel job processes or threads, default %(default)s')

        # console output
        logging_group = a.add_argument_group('Console outputs')

        logging_group.add_argument('--log-level', '--log', type=str, default='warn',
                                   help='logging level, default is %(default)s')
        logging_group.add_argument('-error', dest='log_level', action='store_const', const='error',
                                   help='alias of "--log-level error"')
        logging_group.add_argument('-warn', dest='log_level', action='store_const', const='warn',
                                   help='alias of "--log-level warn"')
        logging_group.add_argument('-info', dest='log_level', action='store_const', const='info',
                                   help='alias of "--log-level info"')
        logging_group.add_argument('-debug', dest='log_level', action='store_const', const='debug',
                                   help='alias of "--log-level debug"')

        logging_group.add_argument('--verbose', type=int, default=0,
                                   help='verbose level, default is %(default)s')
        logging_group.add_argument('-v', '-v+', dest='verbose', action='count',
                                   help='increase verbose level')

        # dask settings
        dask_group = a.add_argument_group('Dask settings')
        dask_group.add_argument('--enable-dask', '--dask', dest='enable_dask',
                                type=to_bool, default=False,
                                help='enable dask supported, default is %(default)s')
        dask_group.add_argument('-dask', '-dask+', dest='enable_dask', action='store_true',
                                help='alias of "--enable-dask true"')

        dask_group.add_argument('--overload', '--load', type=float, default=2.0,
                                help='memory overload of dask local cluster, '
                                     'used only when dask is enabled and  DASK_SCHEDULER_ADDRESS is not found.')
        # gpu settings
        gpu_group = a.add_argument_group('Gpu settings')
        gpu_group.add_argument('--enable-gpu', '--gpu', dest='enable_gpu',
                               type=to_bool, default=False,
                               help='enable dask supported, default is %(default)s')
        gpu_group.add_argument('-gpu', '-gpu+', dest='enable_gpu', action='store_true',
                               help='alias of "--enable-gpu true"')

        # perf settings
        perf_group = a.add_argument_group('Performance record settings')
        perf_group.add_argument('--perf-file', type=str,
                                help='a csv file path to store performance data, default is None(disable recording).')
        perf_group.add_argument('--perf-interval', type=int, default=1,
                                help='second number, default is %(default)s')
        perf_group.add_argument('--perf-recursive', type=to_bool, default=True,
                                help='collect children cpu/mem usage or not, default is %(default)s')

    p = argparse.ArgumentParser(description='HyperGBM command line utility')
    setup_global_args(p)

    sub_parsers = p.add_subparsers(dest='command',
                                   help='command to run')

    setup_train_args(sub_parsers.add_parser(
        'train',
        description='Run an experiment and save trained model to pickle file.'))
    setup_evaluate_args(sub_parsers.add_parser(
        'evaluate',
        description='Evaluate a trained model.'))
    setup_predict_args(sub_parsers.add_parser(
        'predict',
        description='Run prediction with given model and data.'))

    args = p.parse_args(argv)

    if args.version:
        print(f'Version: {hypergbm.__version__}')

    if args.command is None:
        if args.version:
            return
        p.parse_args(['--help'])

    if args.log_level is not None:
        logging.set_level(args.log_level)

    # setup dask if enabled
    if is_dask_installed and os.environ.get('DASK_SCHEDULER_ADDRESS') is not None:
        args.enable_dask = True
    if args.enable_dask:
        if is_dask_installed:
            client = setup_dask(args.overload)
            if args.verbose:
                print(f'enable dask: {client}')
        else:
            print('Either "dask" or "dask_ml" package could not be found, check your installation please.',
                  file=sys.stderr)
            exit(1)

    # start performance process
    if args.perf_file:
        perf_proc = create_dump_perf_process(args.perf_file, recursive=args.perf_recursive, interval=args.perf_interval)
        perf_proc.start()

    # exec command
    fns = [train, evaluate, predict]
    fn = next(filter(lambda f: f.__name__ == args.command, fns))
    fn(args)


def train(args):
    from hypergbm import make_experiment

    for est in estimators:
        setting = getattr(args, est)
        if setting != getattr(cfg, f'estimator_{est}_enabled'):
            setattr(cfg, f'estimator_{est}_enabled', setting)

    reversed_keys = ['command', 'enable_dask', 'overload', 'enable_gpu', 'version',
                     'perf_file', 'perf_interval', 'perf_recursive',
                     'train_data', 'eval_data', 'test_data', 'model_file', 'history_file',
                     'as_local',
                     ] + estimators
    kwargs = {k: v for k, v in args.__dict__.items() if k not in reversed_keys and not k.startswith('_')}

    if args.enable_gpu and not is_cuml_installed:
        from hypergbm.search_space import search_space_general_gpu as search_space
        estimator_early_stopping_rounds = kwargs.get('estimator_early_stopping_rounds')
        if estimator_early_stopping_rounds is not None:
            search_space.options['early_stopping_rounds'] = estimator_early_stopping_rounds
        kwargs['search_space'] = search_space

    tb = get_tool_box(args)
    train_data = tb.load_data(args.train_data)
    eval_data = tb.load_data(args.eval_data) if args.eval_data is not None else None
    test_data = tb.load_data(args.test_data) if args.test_data is not None else None

    if kwargs.get('id') is None:
        kwargs['id'] = 'hypergbm-' + tb.data_hasher()(args.__dict__)
        if args.verbose:
            print(f'experiment id: {kwargs["id"]}')

    experiment = make_experiment(train_data, eval_data=eval_data, test_data=test_data, **kwargs)

    if experiment.callbacks is None:
        experiment.callbacks = []
    if len(experiment.callbacks) == 0:
        experiment.callbacks.append(FreeDataCallback())

    if args.verbose:
        print('>>> running experiment with train data {train_data}, '
              f'eval data: {args.eval_data}, test data: {args.test_data}.')
    del train_data
    del eval_data
    del test_data

    estimator = experiment.run()

    if args.history_file:
        history = experiment.hyper_model_.history
        history.save(args.history_file)

        if args.verbose:
            print(f'>>> history saved to {args.history_file}')

    if args.enable_gpu and is_cuml_installed and args.as_local and hasattr(estimator, 'as_local'):
        estimator = estimator.as_local()

    with open(args.model_file, 'wb') as f:
        pickle.dump(estimator, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.verbose:
        print(f'>>> model saved to {args.model_file}')


def evaluate(args):
    eval_data = args.eval_data
    target = args.target
    model_file = args.model_file
    metrics = args.metric

    assert os.path.exists(model_file), f'Not found {model_file}'
    assert os.path.exists(eval_data), f'Not found {eval_data}'
    assert not (args.enable_dask and args.n_jobs is not None and args.n_jobs > 1)

    tb = get_tool_box(args)

    if args.verbose:
        print(f'>>> load data {eval_data}')
    X = tb.load_data(eval_data, reset_index=True)
    y = X.pop(target)

    if args.verbose:
        print(f'>>> evaluate {model_file} with {metrics}')
    scores = tb.metrics.evaluate(model_file, X, y, metrics,
                                 pos_label=args.pos_label, threshold=args.threshold, n_jobs=args.n_jobs)

    print(scores)


def predict(args):
    data_file = args.data
    model_file = args.model_file
    output_file = args.output
    output_with_data = args.output_with_data
    target = args.target if args.target is not None else 'y'

    assert os.path.exists(model_file), f'Not found {model_file}'
    assert os.path.exists(data_file), f'Not found {data_file}'
    assert not (args.enable_dask and args.n_jobs is not None and args.n_jobs > 1)

    tb = get_tool_box(args)

    if args.verbose:
        print(f'>>> load data {data_file}')
    X = tb.load_data(data_file, reset_index=True)

    if output_with_data:
        if '*' == output_with_data or '*' in output_with_data:
            data = X
        else:
            data_columns = [c for c in X.columns if any([re.match(r, c) for r in output_with_data])]
            if len(data_columns) == 0:
                print(f'>>> No output column found to match {output_with_data}', file=sys.stderr)
                exit(1)
            data = X[data_columns]
    else:
        data = None

    if args.proba:
        if args.verbose:
            print(f'>>> run predict_proba')
        pred = tb.metrics.predict_proba(model_file, X, n_jobs=args.n_jobs)
    else:
        if args.verbose:
            print(f'>>> run predict')
        pred = tb.metrics.predict(model_file, X, n_jobs=args.n_jobs, threshold=args.threshold)

    if args.verbose:
        print(f'>>> save prediction to {output_file}')

    if len(pred.shape) > 1 and pred.shape[1] > 1:
        columns = [f'{target}_{i}' for i in range(pred.shape[1])]
    else:
        columns = [target]

    y = tb.array_to_df(pred, columns=columns)
    df = tb.concat_df([data, y], axis=1) if data is not None else y

    if args.enable_dask:
        tb.parquet().store(df, output_file)
    else:
        df, = tb.to_local(df)
        if output_file.endswith('.parquet') or output_file.endswith('.par'):
            df.to_parquet(output_file)
        elif output_file.endswith('.pkl') or output_file.endswith('.pickle'):
            df.to_pickle(output_file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            df.to_csv(output_file, index=False)

    if args.verbose:
        print('>>> done')


if __name__ == '__main__':
    main()
