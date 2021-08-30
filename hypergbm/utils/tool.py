import argparse
import math
import os
import pickle
import re
import sys

import psutil

from hypernets.utils import const, logging

metric_choices = ['accuracy', 'auc', 'f1', 'logloss', 'mse', 'mae', 'msle', 'precision', 'rmse', 'r2', 'recall']
strategy_choices = ['threshold', 'quantile', 'number']


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


def main():
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

        eg = a.add_argument_group('Early stopping settings')
        eg.add_argument('--early-stopping-time-limit', '--es-time-limit', type=int, default=3600,
                        help='time limit to search and train, set 0 to disable, default %(default)s seconds')
        eg.add_argument('--early-stopping-rounds', '--es-rounds', type=int, default=10,
                        help='default %(default)s ')
        eg.add_argument('--early-stopping-reward', '--es-reward', type=float, default=0.0)

        fg = a.add_argument_group('Feature generation')
        fg.add_argument('--feature-generation', type=to_bool, default=False)
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
        a.add_argument('--metric', '--metrics', type=str, default=['accuracy'], nargs='*', metavar='METRIC',
                       choices=metric_choices,
                       help='metric name list, one or more of [%(choices)s], default %(default)s')
        a.add_argument('--threshold', type=float, default=0.5,
                       help=f'probability threshold to detect pos label, '
                            f'use when task="{const.TASK_BINARY}" only, default %(default)s')
        a.add_argument('--pos-label', type=str, default=None,
                       help='pos label')
        a.add_argument('--jobs', type=int, default=-1,
                       help='job process count, default %(default)s')

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
        a.add_argument('--jobs', type=int, default=-1,
                       help='job process count, default %(default)s')

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

    args = p.parse_args()

    if args.command is None:
        p.parse_args(['--help'])

    if args.log_level is not None:
        logging.set_level(args.log_level)

    # setup dask if enabled
    if os.environ.get('DASK_SCHEDULER_ADDRESS') is not None:
        args.enable_dask = True
    if args.enable_dask:
        client = setup_dask(args.overload)
        if args.verbose:
            print(f'enable dask: {client}')

    # exec command
    fns = [train, evaluate, predict]
    fn = next(filter(lambda f: f.__name__ == args.command, fns))
    fn(args)


def train(args):
    from hypergbm import make_experiment

    train_data = args.train_data
    reversed_keys = ['command', 'enable_dask', 'overload',
                     'train_data', 'model_file']
    kwargs = {k: v for k, v in args.__dict__.items() if k not in reversed_keys and not k.startswith('_')}

    experiment = make_experiment(train_data, **kwargs)

    if args.verbose:
        print('>>> running experiment with train data {train_data}, '
              f'eval data: {args.eval_data}, test data: {args.test_data}.')

    estimator = experiment.run()
    with open(args.model_file, 'wb') as f:
        pickle.dump(estimator, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.verbose:
        print(f'>>> model saved to {args.model_file}')


def evaluate(args):
    from hypernets.utils import load_data
    from hypernets.tabular.metrics import evaluate

    eval_data = args.eval_data
    target = args.target
    model_file = args.model_file
    metrics = args.metric

    assert os.path.exists(model_file), f'Not found {model_file}'
    assert os.path.exists(eval_data), f'Not found {eval_data}'
    assert not (args.enable_dask and args.jobs > 1)

    if args.verbose:
        print(f'>>> load data {eval_data}')
    X = load_data(eval_data)
    y = X.pop(target)

    if args.verbose:
        print(f'>>> evaluate {model_file} with {metrics}')
    scores = evaluate(model_file, X, y, metrics,
                      pos_label=args.pos_label, threshold=args.threshold, n_jobs=args.jobs)

    print(scores)


def predict(args):
    from hypernets.utils import load_data
    from hypernets.tabular import dask_ex as dex, metrics
    import pandas as pd

    data_file = args.data
    model_file = args.model_file
    output_file = args.output
    output_with_data = args.output_with_data
    target = args.target if args.target is not None else 'y'

    assert os.path.exists(model_file), f'Not found {model_file}'
    assert os.path.exists(data_file), f'Not found {data_file}'
    assert not (args.enable_dask and args.jobs > 1)

    if args.verbose:
        print(f'>>> load data {data_file}')
    X = load_data(data_file)

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
        pred = metrics.predict_proba(model_file, X, n_jobs=args.jobs)
    else:
        if args.verbose:
            print(f'>>> run predict')
        pred = metrics.predict(model_file, X, n_jobs=args.jobs, threshold=args.threshold)

    if args.verbose:
        print(f'>>> save prediction to {output_file}')

    if len(pred.shape) > 1 and pred.shape[1] > 1:
        columns = [f'{target}_{i}' for i in range(pred.shape[1])]
    else:
        columns = [target]

    if dex.is_dask_object(pred):
        y = dex.dd.from_dask_array(pred, columns=columns)
    else:
        y = pd.DataFrame(pred, columns=columns)

    df = dex.concat_df([data, y], axis=1) if data is not None else y
    if dex.is_dask_object(df):
        from hypernets.tabular.persistence import to_parquet
        to_parquet(df, output_file)
    else:
        if output_file.endswith('.parquet'):
            df.to_parquet(output_file)
        elif output_file.endswith('.pkl') or output_file.endswith('.pickle'):
            df.to_pickle(output_file, protocol=4)
        else:
            df.to_csv(output_file, index=False)

    if args.verbose:
        print('>>> done')


if __name__ == '__main__':
    main()
