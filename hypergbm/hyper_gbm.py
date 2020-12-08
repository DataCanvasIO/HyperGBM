# -*- coding:utf-8 -*-
"""

"""
import hashlib
import pickle
import re
import time

import numpy as np
import pandas as pd
from sklearn import pipeline as sk_pipeline

from hypergbm.pipeline import ComposeTransformer
from hypernets.model.estimator import Estimator
from hypernets.model.hyper_model import HyperModel
from hypernets.utils import logging, fs
from tabular_toolbox.column_selector import column_object_category_bool, column_zero_or_positive_int32
from tabular_toolbox.data_cleaner import DataCleaner
from tabular_toolbox.metrics import calc_score
from tabular_toolbox.persistence import read_parquet, to_parquet
from tabular_toolbox.utils import hash_dataframe
from .estimators import HyperEstimator

try:
    import shap
    from shap import TreeExplainer

    has_shap = True
except:
    has_shap = False

logger = logging.get_logger(__name__)


class HyperGBMExplainer:
    def __init__(self, hypergbm_estimator, data=None):
        if not has_shap:
            raise RuntimeError('Please install `shap` package first. command: pip install shap')
        self.hypergbm_estimator = hypergbm_estimator
        if data is not None:
            data = self.hypergbm_estimator.transform_data(data)
        self.explainer = TreeExplainer(self.hypergbm_estimator.estimator, data)

    @property
    def expected_value(self):
        return self.explainer.expected_value

    def shap_values(self, X, y=None, tree_limit=None, approximate=False, check_additivity=True, from_call=False,
                    **kwargs):
        X = self.hypergbm_estimator.transform_data(X, **kwargs)
        return self.explainer.shap_values(X, y, tree_limit=tree_limit, approximate=approximate,
                                          check_additivity=check_additivity, from_call=from_call)

    def shap_interaction_values(self, X, y=None, tree_limit=None, **kwargs):
        X = self.hypergbm_estimator.transform_data(X, **kwargs)
        return self.explainer.shap_interaction_values(X, y, tree_limit)

    def transform_data(self, X, **kwargs):
        X = self.hypergbm_estimator.transform_data(X, **kwargs)
        return X


class HyperGBMEstimator(Estimator):
    def __init__(self, task, space_sample, data_cleaner_params=None, cache_dir=None):
        super(HyperGBMEstimator, self).__init__(space_sample=space_sample, task=task)
        self.data_pipeline = None
        self.cache_dir = cache_dir
        self.data_cleaner_params = data_cleaner_params
        self.gbm_model = None
        self.data_cleaner = None
        self.pipeline_signature = None
        self.fit_kwargs = None
        self._build_model(space_sample)

    @property
    def classes_(self):
        if self.gbm_model is not None and hasattr(self.gbm_model, 'classes_'):
            return self.gbm_model.classes_
        else:
            return None

    def _build_model(self, space_sample):
        space, _ = space_sample.compile_and_forward()

        outputs = space.get_outputs()
        assert len(outputs) == 1, 'The space can only contains 1 output.'
        assert isinstance(outputs[0], HyperEstimator), 'The output of space must be `HyperEstimator`.'
        if outputs[0].estimator is None:
            outputs[0].build_estimator(self.task)
        self.gbm_model = outputs[0].estimator
        self.fit_kwargs = outputs[0].fit_kwargs

        pipeline_module = space.get_inputs(outputs[0])
        assert len(pipeline_module) == 1, 'The `HyperEstimator` can only contains 1 input.'
        assert isinstance(pipeline_module[0],
                          ComposeTransformer), 'The upstream node of `HyperEstimator` must be `ComposeTransformer`.'
        # next, (name, p) = pipeline_module[0].compose()
        self.data_pipeline = self.build_pipeline(space, pipeline_module[0])
        logger.debug(f'data_pipeline:{self.data_pipeline}')
        self.pipeline_signature = self.get_pipeline_signature(self.data_pipeline)
        if self.data_cleaner_params is not None:
            self.data_cleaner = DataCleaner(**self.data_cleaner_params)
        else:
            self.data_cleaner = None

    def get_pipeline_signature(self, pipeline):
        repr = pipeline.__repr__(1000000)
        repr = re.sub(r'object at 0x(.*)>', "", repr)
        md5 = hashlib.md5(repr.encode('utf-8')).hexdigest()
        return md5

    def build_pipeline(self, space, last_transformer):
        transformers = []
        while True:
            next, (name, p) = last_transformer.compose()
            transformers.insert(0, (name, p))
            inputs = space.get_inputs(next)
            if inputs == space.get_inputs():
                break
            assert len(inputs) == 1, 'The `ComposeTransformer` can only contains 1 input.'
            assert isinstance(inputs[0],
                              ComposeTransformer), 'The upstream node of `ComposeTransformer` must be `ComposeTransformer`.'
            last_transformer = inputs[0]
        assert len(transformers) > 0
        if len(transformers) == 1:
            return transformers[0][1]
        else:
            pipeline = sk_pipeline.Pipeline(steps=transformers)
            return pipeline

    def summary(self):
        s = f"{self.data_pipeline.__repr__(1000000)}\r\n{self.gbm_model.__repr__()}"
        return s

    def transform_data(self, X, y=None, fit=False, use_cache=None, verbose=0):
        if use_cache is None:
            use_cache = True
        if use_cache:
            data_path, pipeline_path = self._get_cache_filepath(X)
            X_cache = self._get_X_from_cache(data_path, pipeline_path,
                                             as_dask=not isinstance(X, (pd.DataFrame, np.ndarray)))
        else:
            data_path, pipeline_path = None, None
            X_cache = None

        if X_cache is None:
            starttime = time.time()
            if fit:
                if self.data_cleaner is not None:
                    if verbose > 0:
                        print('clean data')
                    X, y = self.data_cleaner.fit_transform(X, y)
                if verbose > 0:
                    print('fit and transform')
                X = self.data_pipeline.fit_transform(X, y)
            else:
                if self.data_cleaner is not None:
                    if verbose > 0:
                        print('clean data')
                    X = self.data_cleaner.transform(X)
                if verbose > 0:
                    print('transform')
                X = self.data_pipeline.transform(X)

            if verbose > 0:
                print(f'taken {time.time() - starttime}s')
            if use_cache:
                self._save_X_to_cache(X, data_path, pipeline_path)
        else:
            X = X_cache

        return X

    def get_categorical_features(self, X):
        cat_cols = column_object_category_bool(X)
        # int_cols = column_int(X)
        # for c in int_cols:
        #     if X[c].min() >= 0 and X[c].max() < np.iinfo(np.int32).max:
        #         cat_cols.append(c)
        cat_cols += column_zero_or_positive_int32(X)
        return cat_cols

    def fit(self, X, y, use_cache=None, verbose=0, **kwargs):
        starttime = time.time()
        if verbose is None:
            verbose = 0
        if verbose > 0:
            print('estimator is transforming the train set')
        X = self.transform_data(X, y, fit=True, use_cache=use_cache, verbose=verbose)

        eval_set = kwargs.get('eval_set')
        kwargs = self.fit_kwargs
        if kwargs.get('verbose') is None and str(type(self.gbm_model)).find('dask') < 0:
            kwargs['verbose'] = verbose
        if verbose > 0:
            print(f'fit kwargs:{kwargs}')

        if eval_set is None:
            eval_set = kwargs.get('eval_set')
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                X_eval, y_eval = eval_set
                if verbose > 0:
                    print('estimator is transforming the eval set')
                X_eval = self.transform_data(X_eval, use_cache=use_cache, verbose=verbose)
                kwargs['eval_set'] = [(X_eval, y_eval)]
            elif isinstance(eval_set, list):
                es = []
                for i, eval_set_ in enumerate(eval_set):
                    X_eval, y_eval = eval_set_
                    if verbose > 0:
                        print(f'estimator is transforming the eval set({i})')
                    X_eval = self.transform_data(X_eval, use_cache=use_cache, verbose=verbose)
                    es.append((X_eval, y_eval))
                    kwargs['eval_set'] = es

        if verbose > 0:
            print('estimator is fitting the data')
        if is_lightgbm_model(self.gbm_model):
            cat_cols = self.get_categorical_features(X)
            kwargs['categorical_feature'] = cat_cols
        elif is_catboost_model(self.gbm_model):
            cat_cols = self.get_categorical_features(X)
            kwargs['cat_features'] = cat_cols

        # print(X.dtypes)
        # print(X.head(10))
        self.gbm_model.fit(X, y, **kwargs)
        if verbose > 0:
            print(f'taken {time.time() - starttime}s')

    def predict(self, X, use_cache=None, verbose=0, **kwargs):
        if verbose is None:
            verbose = 0
        X = self.transform_data(X, use_cache=use_cache, verbose=verbose)
        starttime = time.time()
        if verbose > 0:
            print('estimator is predicting the data')
        preds = self.gbm_model.predict(X, **kwargs)
        if verbose > 0:
            print(f'taken {time.time() - starttime}s')
        return preds

    def predict_proba(self, X, use_cache=None, verbose=0, **kwargs):
        if verbose is None:
            verbose = 0
        X = self.transform_data(X, use_cache=use_cache, verbose=verbose, **kwargs)
        starttime = time.time()
        if verbose > 0:
            print('estimator is predicting the data')
        if hasattr(self.gbm_model, 'predict_proba'):
            preds = self.gbm_model.predict_proba(X, **kwargs)
        else:
            preds = self.gbm_model.predict(X, **kwargs)
        if verbose > 0:
            print(f'taken {time.time() - starttime}s')
        return preds

    def evaluate(self, X, y, metrics=None, use_cache=None, verbose=0, **kwargs):
        if metrics is None:
            metrics = ['accuracy']
        if self.task != 'regression':
            proba = self.predict_proba(X, use_cache=use_cache, verbose=verbose, **kwargs)
        else:
            proba = None
        preds = self.predict(X, use_cache=use_cache, verbose=verbose, **kwargs)
        scores = calc_score(y, preds, proba, metrics, self.task)
        return scores

    def save(self, model_file):
        with fs.open(f'{model_file}', 'wb') as output:
            pickle.dump(self, output, protocol=2)

    @staticmethod
    def load(model_file):
        with fs.open(f'{model_file}', 'rb') as input:
            model = pickle.load(input)
            return model

    def _get_cache_filepath(self, X):
        # file_path = f'{self.cache_dir}/{X.shape[0]}_{X.shape[1]}_{self.pipeline_signature}.h5'
        start_at = time.time()
        shape = self._get_dataframe_shape(X)
        at1 = time.time()
        logger.debug(f'get shape in {at1 - start_at} seconds, {shape}')
        hash = hash_dataframe(X)
        at2 = time.time()
        logger.debug(f'calc hash in {at2 - at1} seconds, {hash}')

        data_path = f'{self.cache_dir}/{shape[0]}_{shape[1]}_{hash}_{self.pipeline_signature}.parquet'
        pipeline_path = f'{self.cache_dir}/{shape[0]}_{shape[1]}_{hash}_pipeline_{self.pipeline_signature}.pkl'
        return data_path, pipeline_path

    def _get_dataframe_shape(self, X):
        if isinstance(X, pd.DataFrame):
            return X.shape
        else:
            rows = X.reduction(lambda df: df.shape[0], np.sum).compute()
            return rows, X.shape[1]

    def _get_X_from_cache(self, data_path, pipeline_path, as_dask=False):
        if fs.exists(data_path):
            start_at = time.time()
            df = self._load_df(data_path, as_dask)
            if pipeline_path:
                try:
                    with fs.open(pipeline_path, 'rb') as input:
                        self.data_pipeline, self.data_cleaner = pickle.load(input)
                except:
                    if fs.exists(data_path):
                        fs.rm(data_path)
                    return None
            done_at = time.time()
            logger.debug(f'load cache in {done_at - start_at} seconds')
            return df
        else:
            return None

    def _save_X_to_cache(self, X, data_path, pipeline_path):
        start_at = time.time()
        self._save_df(data_path, X)

        if pipeline_path:
            try:
                with fs.open(pipeline_path, 'wb') as output:
                    pickle.dump((self.data_pipeline, self.data_cleaner), output, protocol=2)
            except Exception as e:
                logger.error(e)
                if fs.exists(pipeline_path):
                    fs.rm(pipeline_path)
        done_at = time.time()
        logger.debug(f'save cache in {done_at - start_at} seconds')

    def _load_df(self, filepath, as_dask=False):
        try:
            # with fs.open(filepath, 'rb') as f:
            #     df = pd.read_parquet(f)
            #     return df
            df = read_parquet(filepath, delayed=as_dask, filesystem=fs)
            return df
        except:
            if fs.exists(filepath):
                fs.rm(filepath)
            return None

    def _save_df(self, filepath, df):
        try:
            # with fs.open(filepath, 'wb') as f:
            #     df.to_parquet(f)
            if not isinstance(df, pd.DataFrame):
                fs.mkdirs(filepath, exist_ok=True)
            to_parquet(df, filepath, fs)
        except Exception as e:
            logger.error(e)
            # traceback.print_exc()
            if fs.exists(filepath):
                fs.rm(filepath)

    def get_explainer(self, data=None):
        explainer = HyperGBMExplainer(self, data=data)
        return explainer


class HyperGBM(HyperModel):
    def __init__(self, searcher, dispatcher=None, callbacks=None, reward_metric='accuracy', task=None,
                 data_cleaner_params=None, cache_dir=None, clear_cache=True):
        if callbacks is None:
            callbacks = []
        self.data_cleaner_params = data_cleaner_params
        self.cache_dir = self._prepare_cache_dir(cache_dir, clear_cache)
        self.clear_cache = clear_cache
        HyperModel.__init__(self, searcher, dispatcher=dispatcher, callbacks=callbacks, reward_metric=reward_metric,
                            task=task)

    def _get_estimator(self, space_sample):
        estimator = HyperGBMEstimator(task=self.task, space_sample=space_sample,
                                      data_cleaner_params=self.data_cleaner_params,
                                      cache_dir=self.cache_dir)
        return estimator

    def load_estimator(self, model_file):
        assert model_file is not None
        return HyperGBMEstimator.load(model_file)

    def export_trail_configuration(self, trail):
        return '`export_trail_configuration` does not implemented'

    def blend_models(self, samples, X, y, **kwargs):
        models = []
        for sample in samples:
            estimator = self.final_train(sample, X, y, **kwargs)
            models.append(estimator)
        blends = BlendModel(models, self.task)
        return blends

    @staticmethod
    def _prepare_cache_dir(cache_dir, clear_cache):
        if cache_dir is None:
            cache_dir = 'tmp/cache'
        if cache_dir[-1] == '/':
            cache_dir = cache_dir[:-1]

        # cache_dir = os.path.expanduser(cache_dir)

        try:
            if not fs.exists(cache_dir):
                fs.makedirs(cache_dir, exist_ok=True)
            else:
                if clear_cache:
                    fs.rm(cache_dir, recursive=True)
                    fs.mkdirs(cache_dir, exist_ok=True)
        except PermissionError:
            pass  # ignore
        except FileExistsError:
            pass  # ignore

        return cache_dir


class BlendModel():
    def __init__(self, gbm_models, task):
        self.gbm_models = gbm_models
        self.task = task

    def predict_proba(self, X, **kwargs):
        proba_avg = None
        count = 0
        for gbm_model in self.gbm_models:
            proba = gbm_model.predict_proba(X, **kwargs)
            if proba is not None:
                if len(proba.shape) == 1:
                    proba = proba.reshape((-1, 1))
                if proba_avg is None:
                    proba_avg = proba
                else:
                    proba_avg += proba
                count = count + 1
        proba_avg = proba_avg / count
        np.random.uniform()
        return proba_avg

    def predict(self, X, proba_threshold=0.5, **kwargs):
        proba = self.predict_proba(X, **kwargs)
        return self.proba2predict(proba, proba_threshold)

    def proba2predict(self, proba, proba_threshold=0.5):
        if self.task == 'regression':
            return proba
        if proba.shape[-1] > 2:
            predict = proba.argmax(axis=-1)
        elif proba.shape[-1] == 2:
            predict = (proba[:, 1] > proba_threshold).astype('int32')
        else:
            predict = (proba > proba_threshold).astype('int32')
        return predict

    def evaluate(self, X, y, metrics=None, **kwargs):
        if metrics is None:
            metrics = ['accuracy']
        proba = self.predict_proba(X, **kwargs)
        preds = self.proba2predict(proba)
        scores = calc_score(y, preds, proba, metrics)
        return scores

    def save_model(self, filepath):
        with fs.open(f'{filepath}', 'wb') as output:
            pickle.dump(self, output)

    @staticmethod
    def load_model(filepath):
        with fs.open(f'{filepath}', 'rb') as input:
            model = pickle.load(input)
            return model


def _is_any_class(model, classes):
    try:
        model_classes = (model.__class__,) + model.__class__.__bases__
        return any(c.__name__ in classes for c in model_classes)
    except:
        return False


def is_lightgbm_model(model):
    return _is_any_class(model, {'LGBMClassifier', 'LGBMRegressor', 'LGBMModel'})


def is_catboost_model(model):
    return _is_any_class(model, {'CatBoostClassifier', 'CatBoostRegressor'})
