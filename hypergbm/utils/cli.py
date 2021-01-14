import argparse
from os import path as P
import os, logging
logger = logging.getLogger("cli")


def main():
    # 1. setup parser
    parser = argparse.ArgumentParser(description='HyperGBM is a full pipeline AutoML tool for tabular data', add_help=True)
    parser.add_argument("--train_file", help="Csv file path for train", default=None, required=True)
    parser.add_argument("--eval_file", help="Csv file path for evaluate, target column should be included,"
                                            " if not specified,  will use part of training set", default=None, required=False)
    parser.add_argument("--eval_size", help="Scale of evaluation if not specified 'eval_file'", default=0.3, required=False)
    parser.add_argument("--test_file", help="Csv file path to predict", default=None, required=False)

    parser.add_argument("--target", help="Target to train ", default=None, required=True)
    parser.add_argument("--pos_label", help="Positive label, is required for binary task", default=None, required=False)
    parser.add_argument("--max_trials", help="Maximum search times", default=30, required=False)
    parser.add_argument("--model_output", help="Path to save model", default="model.pkl", required=False)
    parser.add_argument("--prediction_output", help="Path to save model", default="prediction.csv", required=False)
    parser.add_argument("--searcher", help="Search strategy, is one of 'random','evolution','MCTS' ", default='MCTS', required=False)

    # 2. read and check params
    args_namespace = parser.parse_args()

    train_file = args_namespace.train_file
    eval_file = args_namespace.eval_file
    eval_size = args_namespace.eval_size
    test_file = args_namespace.test_file

    target = args_namespace.target
    pos_label = args_namespace.pos_label
    max_trials = int(args_namespace.max_trials)
    model_output = P.abspath(args_namespace.model_output)
    prediction_output = P.abspath(args_namespace.prediction_output)


    def _require_file(path):
        if not P.exists(path):
            raise ValueError(f"Path {path} is not exists ")

        if not P.isfile(train_file):
            raise ValueError(f"Path {path} is not a file ")


    # 2.1. check train_file
    _require_file(train_file)
    train_file = P.abspath(train_file)

    # 2.2. check eval file
    if eval_file is not None and len(eval_file) > 1:
        _require_file(eval_file)
        eval_file = P.abspath(eval_file)
    else:
        eval_size = float(eval_size)
        logger.info(f"You did not specify an evaluation set， will use part of training set, the eval_size is {eval_size}")

    # 2.3. check test file
    if test_file is not None and len(test_file) > 1:
        _require_file(test_file)
        test_file = P.abspath(test_file)

    # 2.4. check output not exists
    if P.exists(model_output):
        raise ValueError(f"Path {model_output} already exists")

    # 2.5. check prediction output not exists
    if P.exists(prediction_output):
        raise ValueError(f"Path {model_output} already exists")


    # import after check
    from tabular_toolbox import const as tt_const
    import pandas as pd
    import pickle
    from hypernets.core.callbacks import EarlyStoppingCallback, SummaryCallback, FileLoggingCallback
    from hypernets.core.ops import HyperInput, ModuleChoice
    from hypernets.core.search_space import HyperSpace, Choice
    from hypernets.core.searcher import OptimizeDirection
    from hypernets.searchers.mcts_searcher import MCTSSearcher

    from hypergbm.estimators import LightGBMEstimator, XGBoostEstimator, CatBoostEstimator
    from hypergbm.hyper_gbm import HyperGBM
    from hypergbm.pipeline import DataFrameMapper
    from hypergbm.sklearn.sklearn_ops import numeric_pipeline_complex, categorical_pipeline_simple
    from hypergbm.search_space import search_space_general
    from hypernets.experiment.general import GeneralExperiment
    from tabular_toolbox.column_selector import column_object

    # 3. data partition
    df_train = pd.read_csv(train_file)
    n_rows = df_train.shape[0]
    df_train.dropna(axis=0, how='all', subset=[target], inplace=True)
    logger.warning(f"Total rows which label is None: {n_rows - df_train.shape[0]}")
    X_train = df_train
    y_train = X_train.pop(target)
    classes = list(set(y_train))
    from tabular_toolbox.utils._common import infer_task_type
    task_type = infer_task_type(y_train)[0]

    if task_type == tt_const.TASK_BINARY:
        if pos_label is None or len(pos_label) < 1:
            raise ValueError("Param pos_label can not be empty for binary task ")


    if eval_file is not None and len(eval_file) > 1:
        df_eval = pd.read_csv(eval_file)
        y_eval = df_eval.pop(target)  # target must in eval set
        X_eval = df_eval
    else:
        y_eval = None
        X_eval = None

    if test_file is not None and len(test_file) > 1:
        X_test = pd.read_csv(test_file)
    else:
        X_test = None

    # 4. search best params
    gbm_optimize_metrics = {
        tt_const.TASK_BINARY: "auc",
        tt_const.TASK_MULTICLASS: "accuracy",
        tt_const.TASK_REGRESSION: "rmse"
    }


    def get_direction(metric_name):
        metric_name = metric_name.lower()
        if metric_name in ["auc", "accuracy"]:
            return OptimizeDirection.Maximize
        elif metric_name in ["rootmeansquarederror", 'rmse']:
            return OptimizeDirection.Minimize
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")


    reward_metric = gbm_optimize_metrics[task_type]
    optimize_direction = get_direction(reward_metric)


    def regression_search_space(dataframe_mapper_default=False, lightgbm_fit_kwargs=None, xgb_fit_kwargs=None, catboost_fit_kwargs=None):
        if lightgbm_fit_kwargs is None:
            lightgbm_fit_kwargs = {}
        if xgb_fit_kwargs is None:
            xgb_fit_kwargs = {}
        if catboost_fit_kwargs is None:
            catboost_fit_kwargs = {}

        space = HyperSpace()
        with space.as_default():
            input = HyperInput(name='input1')
            num_pipeline = numeric_pipeline_complex()(input)
            cat_pipeline = categorical_pipeline_simple()(input)
            union_pipeline = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                             df_out_dtype_transforms=[(column_object, 'int')])([num_pipeline, cat_pipeline])

            lightgbm_init_kwargs = {
                'boosting_type': Choice(['gbdt', 'dart', 'goss']),
                'num_leaves': Choice([3, 5]),
                'learning_rate': 0.1,
                'n_estimators': Choice([10, 30, 50]),
                'max_depth': Choice([3, 5]),
                'reg_alpha': Choice([1e-2, 0.1, 1, 100]),
                'reg_lambda': Choice([1e-2, 0.1, 1, 100]),
                # 'class_weight': 'balanced',
                # subsample_for_bin = 200000, objective = None, class_weight = None,
                #  min_split_gain = 0., min_child_weight = 1e-3, min_child_samples = 20,
            }
            lightgbm_est = LightGBMEstimator(fit_kwargs=lightgbm_fit_kwargs, **lightgbm_init_kwargs)
            xgb_init_kwargs = {}
            xgb_est = XGBoostEstimator(fit_kwargs=xgb_fit_kwargs, **xgb_init_kwargs)

            catboost_init_kwargs = {
                'silent': True
            }
            catboost_est = CatBoostEstimator(fit_kwargs=catboost_fit_kwargs, **catboost_init_kwargs)

            ModuleChoice([lightgbm_est, xgb_est, catboost_est], name='estimator_options')(union_pipeline)
            space.set_inputs(input)
        return space


    search_space = search_space_general
    gbm_task = "classification"
    if task_type == tt_const.TASK_REGRESSION:
        search_space = regression_search_space
        gbm_task = "regression"

    rs = MCTSSearcher(search_space, max_node_space=10, optimize_direction=optimize_direction)
    hk = HyperGBM(rs, task=gbm_task, reward_metric=reward_metric,
                  callbacks=[SummaryCallback(),
                             FileLoggingCallback(rs),
                             EarlyStoppingCallback(max_no_improvement_trials=5, mode=optimize_direction.value)])

    experiment = GeneralExperiment(hk, X_train, y_train, X_eval=X_eval, y_eval=y_eval, X_test=X_test)
    estimator = experiment.run(use_cache=True, max_trails=max_trials)

    # 5. do predict is has test data
    if X_test is not None:
        if target in X_test:
            X_test_cleaned = X_test.drop(target, axis=1)
        else:
            X_test_cleaned = X_test
        y_prediction = estimator.predict(X_test_cleaned)
        X_test['prediction'] = y_prediction
        if not P.exists(P.dirname(prediction_output)):
            os.makedirs(P.dirname(prediction_output))
        X_test.to_csv(prediction_output, index=None)

    # 6. persist model
    if not P.exists(P.dirname(model_output)):
        os.makedirs(P.dirname(model_output))

    with open(model_output, 'wb') as f:
        pickle.dump(estimator, f)


if __name__ == '__main__':
    main()
