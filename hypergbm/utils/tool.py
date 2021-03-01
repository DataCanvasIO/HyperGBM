import argparse
import math
import os
import pickle
import sys

import psutil

# from hypernets.utils import logging
#
# logger = logging.get_logger(__name__)


metric_choices = ['accuracy', 'auc', 'f1', 'logloss', 'mse', 'mae', 'msle', 'precision', 'rmse', 'r2', 'recall']


def to_bool(v):
    if isinstance(v, bool) or v is None:
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
        tg.add_argument('--train-data', '--train-file', '--train', type=str, default=None,
                        help='the file path of training data, .csv and .parquet files are supported.')
        tg.add_argument('--eval-data', '--eval-file', '--eval', type=str, default=None,
                        help='the file path of evaluation data, .csv and .parquet files are supported.')
        tg.add_argument('--test-data', '--test-file', '--test', type=str, default=None,
                        help='the file path of testing data, .csv and .parquet files are supported.')
        tg.add_argument('--train-test-split-strategy', type=str, default=None,
                        choices=[None, 'adversarial_validation'],
                        help=None)
        tg.add_argument('--target', '--y', type=str, default=None,
                        help='training target feature name, default is "y"')
        tg.add_argument('--task', type=str, default=None, choices=['binary', 'multiclass', 'regression'],
                        help='train task type, will be detected from target by default')
        tg.add_argument('--max-trials', '--trials', '--n', type=int, default=10,
                        help='search trial number limit, default %(default)s')
        tg.add_argument('--reward-metric', '--reward', '--metric', type=str, default='accuracy', metavar='METRIC',
                        choices=metric_choices,
                        help='search reward metric name, one of [%(choices)s], default %(default)s')
        tg.add_argument('--cv', type=to_bool, default=True,
                        help='enable to disable cross validation, default %(default)s')
        tg.add_argument('-cv', '-cv+', default='cv', action='store_true',
                        help='alias of "--cv true"')
        tg.add_argument('-cv-', default='cv', action='store_false',
                        help='alias of "--cv false"')
        tg.add_argument('--ensemble-size', '--ensemble', type=int, default=20,
                        help='how many estimators are involved, set "0" to disable ensemble, default %(default)s')
        tg.add_argument('--model-file', '--model', type=str, default='model.pkl',
                        help='the output pickle file name for trained model, default %(default)s')

        eg = a.add_argument_group('Early stopping settings')
        eg.add_argument('--early-stopping-time-limit', '--es-time-limit', type=int, default=3600,
                        help='time limit to search and train, set 0 to disable, default %(default)s seconds')
        eg.add_argument('--early-stopping-rounds', '--es-rounds', type=int, default=10,
                        help='default %(default)s ')
        eg.add_argument('--early-stopping-reward', '--es-reward', type=float, default=0.0)

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

        s2g = a.add_argument_group('Two stage searching')
        s2g.add_argument('--feature-reselection', type=to_bool, default=False,
                         help='default %(default)s')
        s2g.add_argument('-re', '-re+', '-pi', '-pi+', dest='feature_reselection', action='store_true',
                         help='alias of "--feature-reselection true"')
        s2g.add_argument('-re-', '-pi-', dest='feature_reselection', action='store_false',
                         help='alias of "--feature-reselection false"')

        s2g.add_argument('--pseudo-labeling', type=to_bool, default=False,
                         help='default %(default)s')
        s2g.add_argument('-pl', '-pl+', dest='pseudo_labeling', action='store_true',
                         help='alias of "--pseudo-labeling true"')
        s2g.add_argument('-pl-', dest='pseudo_labeling', action='store_false',
                         help='alias of "--pseudo-labeling false"')
        s2g.add_argument('--pseudo-labeling-proba-threshold', '--pl-threshold', type=float, default=0.8,
                         help='default %(default)s')
        s2g.add_argument('--pseudo-labeling-resplit', '--pl-resplit', type=to_bool, default=False,
                         help='default %(default)s')

        # others
        og = a.add_argument_group('Other settings')
        og.add_argument('--use-cache', type=to_bool, default=None)
        og.add_argument('-use-cache', '-use-cache+', dest='use_cache', action='store_true',
                        help='alias of "--use-cache true"')
        og.add_argument('-use-cache-', dest='use_cache', action='store_false',
                        help='alias of "--use-cache false"')

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
        a.add_argument('--metric', type=str, default=['accuracy'], nargs='*', metavar='METRIC',
                       choices=metric_choices,
                       help='metric name list, one or more of [%(choices)s], default %(default)s')
        a.add_argument('--model-file', '--model', default='model.pkl',
                       help='the pickle file name for trained model, default %(default)s')

    def setup_predict_args(a):
        a.add_argument('--data', '--data-file', type=str, required=True,
                       help='the data path of to predict, .csv and .parquet files are supported.')
        a.add_argument('--model-file', '--model', default='model.pkl',
                       help='the pickle file name for trained model, default %(default)s')
        a.add_argument('--target', '--y', type=str, default='y',
                       help='target feature name, default is %(default)s')
        a.add_argument('--output', '--output-file', type=str, default='prediction.csv',
                       help='the output file name, default is %(default)s')
        a.add_argument('--output-with-data', '--with-data', type=to_bool, default=False,
                       help='output source data ot output also')
        a.add_argument('-output-with-data', '-with-data', dest='output_with_data', action='store_true',
                       help='alias of "--output-with-data true"')

    def setup_global_args(a):
        # console output
        logging_group = a.add_argument_group('Console outputs')

        logging_group.add_argument('--log-level', '--log', type=str, default=None,
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

    # setup dask if enabled
    enable_dask = args.enable_dask or os.environ.get('DASK_SCHEDULER_ADDRESS') is not None
    if enable_dask:
        client = setup_dask(args.overload)
        if args.verbose:
            print(f'enable dask: {client}')

    # exec command
    fns = [train, evaluate, predict]
    fn = next(filter(lambda f: f.__name__ == args.command, fns))
    fn(args)


def train(args):
    from hypergbm import make_experiment
    from tabular_toolbox.datasets import dsutils
    from tabular_toolbox import dask_ex as dex

    if args.train_data is None or len(args.train_data) == 0:
        if dex.dask_enabled():
            df = dsutils.load_bank_by_dask()
            # import dask_ml.preprocessing as dm_pre
            # df['y'] = dm_pre.LabelEncoder().fit_transform(df['y'])
        else:
            df = dsutils.load_bank()
        df.drop(['id'], axis=1)
        target = 'y'
        train_data = df.sample(frac=0.1)
        print('>>> Not train-data found, make experiment with tabular bank data.', file=sys.stderr)
    else:
        train_data = args.train_data
        target = args.target
        if args.verbose:
            print(f'>>> make experiment with train data {train_data}, '
                  f'eval data: {args.eval_data}, test data: {args.test_data}.')

    experiment = make_experiment(train_data, target=target, test_data=args.test_data, eval_data=args.eval_data,
                                 task=args.task,
                                 max_trials=args.max_trials,
                                 ensemble_size=args.ensemble_size,
                                 reward_metric=args.reward_metric,
                                 train_test_split_strategy=args.train_test_split_strategy,  # 'adversarial_validation'
                                 early_stopping_time_limit=args.early_stopping_time_limit,
                                 early_stopping_rounds=args.early_stopping_rounds,
                                 early_stopping_reward=args.early_stopping_reward,
                                 cv=args.cv,
                                 drift_detection=args.drift_detection,
                                 drift_detection_remove_shift_variable=args.drift_detection_remove_shift_variable,
                                 drift_detection_variable_shift_threshold=args.drift_detection_variable_shift_threshold,
                                 drift_detection_threshold=args.drift_detection_threshold,
                                 drift_detection_remove_size=args.drift_detection_remove_size,
                                 drift_detection_min_features=args.drift_detection_min_features,
                                 drift_detection_num_folds=args.drift_detection_num_folds,
                                 collinearity_detection=args.collinearity_detection,
                                 pseudo_labeling=args.pseudo_labeling,
                                 pseudo_labeling_proba_threshold=args.pseudo_labeling_proba_threshold,
                                 pseudo_labeling_resplit=args.pseudo_labeling_resplit,
                                 feature_reselection=args.feature_reselection,
                                 use_cache=args.use_cache,
                                 clear_cache=args.clear_cache,
                                 log_level=args.log_level,
                                 verbose=args.verbose,
                                 )

    if args.verbose:
        print('>>> running experiment ...')

    estimator = experiment.run()
    with open(args.model_file, 'wb') as f:
        pickle.dump(estimator, f, protocol=4)

    if args.verbose:
        print(f'>>> model saved to {args.model_file}')


def evaluate(args):
    from tabular_toolbox.utils import load_data
    from tabular_toolbox.const import TASK_BINARY, TASK_MULTICLASS
    from tabular_toolbox.metrics import calc_score

    eval_data = args.eval_data
    target = args.target
    model_file = args.model_file
    metrics = args.metric

    assert os.path.exists(model_file), f'Not found {model_file}'
    assert os.path.exists(eval_data), f'Not found {eval_data}'

    if args.verbose:
        print(f'>>> load model {model_file}')
    with open(model_file, 'rb') as f:
        estimator = pickle.load(f)

    if hasattr(estimator, 'task'):
        task = getattr(estimator, 'task', None)
    elif type(estimator).__name__.find('Pipeline') >= 0 and hasattr(estimator, 'steps'):
        task = getattr(estimator.steps[-1][1], 'task', None)
    else:
        task = None

    if args.verbose:
        print(f'>>> model task type: {task}')

    if args.verbose:
        print(f'>>> load data {eval_data}')

    X = load_data(eval_data)
    y = X.pop(target)

    if args.verbose:
        print(f'>>> run predict')

    pred = estimator.predict(X)
    if task in {TASK_BINARY, TASK_MULTICLASS}:
        if args.verbose:
            print(f'>>> run predict_proba')
        proba = estimator.predict_proba(X)
    else:
        proba = None

    if args.verbose:
        print(f'>>> calc scores: {metrics}')
    scores = calc_score(y_true=y, y_preds=pred, y_proba=proba, metrics=metrics)

    print(scores)


def predict(args):
    from tabular_toolbox.utils import load_data
    from tabular_toolbox import dask_ex as dex
    import pandas as pd

    data_file = args.data
    model_file = args.model_file
    output_file = args.output
    output_with_data = args.output_with_data
    target = args.target

    assert os.path.exists(model_file), f'Not found {model_file}'
    assert os.path.exists(data_file), f'Not found {data_file}'

    if args.verbose:
        print(f'>>> load model {model_file}')
    with open(model_file, 'rb') as f:
        estimator = pickle.load(f)

    if args.verbose:
        print(f'>>> load data {data_file}')
    X = load_data(data_file)

    if args.verbose:
        print(f'>>> run predict')
    pred = estimator.predict(X)

    if args.verbose:
        print(f'>>> save prediction to  {output_file}')

    if dex.is_dask_object(pred):
        y = dex.dd.from_dask_array(pred, columns=[target])
    else:
        y = pd.Series(pred, name=target)

    if output_with_data:
        df = dex.concat_df([X, y], axis=1)
    else:
        df = y

    df.to_csv(output_file, index=False)

    if args.verbose:
        print('>>> done')


if __name__ == '__main__':
    main()
