# -*- coding:utf-8 -*-
"""

"""
import copy
import hashlib
import pickle
import re
import time

import dask.array as da
import dask.dataframe as dd
import dask_ml.model_selection as dm_sel
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
from sklearn import pipeline as sk_pipeline
from sklearn.model_selection import KFold, StratifiedKFold

from hypergbm.pipeline import ComposeTransformer
from hypernets.model.estimator import Estimator
from hypernets.model.hyper_model import HyperModel
from hypernets.utils import logging, fs
from tabular_toolbox import dask_ex as dex
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

DEFAULT_EVAL_SIZE_LIMIT = 10000


def get_sampler(sampler):
    samplers = {'RandomOverSampler': RandomOverSampler,
                'SMOTE': SMOTE,
                'ADASYN': ADASYN,
                'RandomUnderSampler': RandomUnderSampler,
                'NearMiss': NearMiss,
                'TomekLinks': TomekLinks,
                'EditedNearestNeighbours': EditedNearestNeighbours
                }
    sampler_cls = samplers.get(sampler)
    if sampler_cls is not None:
        return sampler_cls()
    else:
        return None


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
        self.cv_gbm_models_ = None
        self.data_cleaner = None
        self.pipeline_signature = None
        self.fit_kwargs = None
        self.class_balancing = None
        self.classes_ = None
        self._build_model(space_sample)

    def _build_model(self, space_sample):
        space, _ = space_sample.compile_and_forward()

        outputs = space.get_outputs()
        assert len(outputs) == 1, 'The space can only contains 1 output.'
        assert isinstance(outputs[0], HyperEstimator), 'The output of space must be `HyperEstimator`.'
        if outputs[0].estimator is None:
            outputs[0].build_estimator(self.task)
        self.gbm_model = outputs[0].estimator
        self.class_balancing = outputs[0].class_balancing
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
        s = f"{self.data_pipeline.__repr__(1000000)}"
        # s = f"{self.data_pipeline.__repr__(1000000)}\r\n{self.gbm_model.__repr__()}"
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
                        logger.info('clean data')
                    X, y = self.data_cleaner.fit_transform(X, y)
                if verbose > 0:
                    logger.info('fit and transform')
                X = self.data_pipeline.fit_transform(X, y)
            else:
                if self.data_cleaner is not None:
                    if verbose > 0:
                        logger.info('clean data')
                    X = self.data_cleaner.transform(X)
                if verbose > 0:
                    logger.info('transform')
                X = self.data_pipeline.transform(X)

            if verbose > 0:
                logger.info(f'taken {time.time() - starttime}s')
            if use_cache:
                self._save_X_to_cache(X, data_path, pipeline_path)
        else:
            X = X_cache

        return X

    def fit_cross_validation(self, X, y, use_cache=None, verbose=0, stratified=True, num_folds=3,
                             shuffle=False, random_state=9527, metrics=None, **kwargs):
        if dex.exist_dask_object(X, y):
            return self.fit_cross_validation_by_dask(X, y, use_cache=use_cache, verbose=verbose,
                                                     stratified=stratified, num_folds=num_folds,
                                                     shuffle=shuffle, random_state=random_state,
                                                     metrics=metrics, **kwargs)
        starttime = time.time()
        if verbose is None:
            verbose = 0
        if verbose > 0:
            logger.info('estimator is transforming the train set')

        X = self.transform_data(X, y, fit=True, use_cache=use_cache, verbose=verbose)

        if stratified and self.task == 'binary':
            iterators = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=9527)
        else:
            iterators = KFold(n_splits=num_folds, shuffle=True, random_state=9527)

        y = np.array(y)

        kwargs = self.fit_kwargs
        if kwargs.get('verbose') is None:
            kwargs['verbose'] = verbose

        oof_ = None
        self.cv_gbm_models_ = []
        for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X, y)):
            x_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
            x_val_fold, y_val_fold = X.iloc[valid_idx], y[valid_idx]

            kwargs['eval_set'] = [(x_val_fold, y_val_fold)]
            sample_weight = None
            if self.task != 'regression' and self.class_balancing is not None:
                sampler = get_sampler(self.class_balancing)
                if sampler is None:
                    sample_weight = self._get_sample_weight(y_train_fold)
                else:
                    x_train_fold, y_train_fold = sampler.fit_sample(x_train_fold, y_train_fold)
            kwargs['sample_weight'] = sample_weight

            fold_est = copy.deepcopy(self.gbm_model)
            fit_kwargs = {**kwargs, 'verbose': 0}
            fold_est.fit(x_train_fold, y_train_fold, **fit_kwargs)
            # print(f'fold {n_fold}, est:{fold_est.__class__},  best_n_estimators:{fold_est.best_n_estimators}')
            if self.classes_ is None and hasattr(fold_est, 'classes_'):
                self.classes_ = fold_est.classes_
            if self.task == 'regression':
                proba = fold_est.predict(x_val_fold)
            else:
                proba = fold_est.predict_proba(x_val_fold)

            if oof_ is None:
                if len(proba.shape) == 1:
                    oof_ = np.zeros(y.shape, proba.dtype)
                else:
                    oof_ = np.zeros((y.shape[0], proba.shape[-1]), proba.dtype)
            oof_[valid_idx] = proba
            self.cv_gbm_models_.append(fold_est)

        if metrics is None:
            metrics = ['accuracy']
        proba = oof_
        if self.task == 'regression':
            proba = None
            preds = oof_
        else:
            preds = self.proba2predict(oof_)
            preds = np.array(self.classes_).take(preds, axis=0)
        scores = calc_score(y, preds, proba, metrics, self.task)
        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')
        return scores, oof_

    def fit_cross_validation_by_dask(self, X, y, use_cache=None, verbose=0, stratified=True, num_folds=3,
                                     shuffle=False, random_state=9527, metrics=None, **kwargs):
        starttime = time.time()
        if verbose is None:
            verbose = 0
        if verbose > 0:
            logger.info('estimator is transforming the train set')

        eval_size_limit = kwargs.get('eval_size_limit', DEFAULT_EVAL_SIZE_LIMIT)

        X = self.transform_data(X, y, fit=True, use_cache=use_cache, verbose=verbose)

        iterators = dm_sel.KFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)
        X_values = X.to_dask_array(lengths=True)
        y_values = y.to_dask_array(lengths=True)

        kwargs = self.fit_kwargs
        # if kwargs.get('verbose') is None and str(type(self.gbm_model)).find('dask') < 0:
        #     kwargs['verbose'] = verbose

        oof_ = []
        models = []
        for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X_values, y_values)):
            x_train_fold, y_train_fold = X_values[train_idx], y_values[train_idx]
            x_val_fold, y_val_fold = X_values[valid_idx], y_values[valid_idx]
            x_train_fold = dex.array_to_df(x_train_fold, meta=X)
            x_val_fold = dex.array_to_df(x_val_fold, meta=X)

            sample_weight = None
            if self.task != 'regression' and self.class_balancing is not None:
                sampler = get_sampler(self.class_balancing)
                if sampler is None:
                    sample_weight = self._get_sample_weight(y_train_fold)
                else:
                    x_train_fold, y_train_fold = sampler.fit_sample(x_train_fold, y_train_fold)

            if valid_idx.shape[0] > eval_size_limit:
                eval_idx = valid_idx[0:eval_size_limit]
                x_eval, y_eval = X_values[eval_idx], y_values[eval_idx]
                x_eval = dex.array_to_df(x_eval, meta=X)
            else:
                x_eval, y_eval = x_val_fold, y_val_fold

            if self.task != 'regression' and \
                    len(da.unique(y_eval).compute()) != len(da.unique(y_train_fold).compute()):
                eval_set = None
            else:
                eval_set = [dex.compute(x_eval, y_eval)]

            fold_est = copy.deepcopy(self.gbm_model)
            fit_kwargs = {**kwargs, 'sample_weight': sample_weight, 'eval_set': eval_set, 'verbose': 0,
                          'early_stopping_rounds': kwargs.get('early_stopping_rounds')
                          if eval_set is not None else None}
            fold_est.fit(x_train_fold, y_train_fold, **fit_kwargs)

            # print(f'fold {n_fold}, est:{fold_est.__class__},  best_n_estimators:{fold_est.best_n_estimators}')
            if self.classes_ is None and hasattr(fold_est, 'classes_'):
                self.classes_ = fold_est.classes_
            if self.task == 'regression':
                proba = fold_est.predict(x_val_fold)
            else:
                proba = fold_est.predict_proba(x_val_fold)

            index = valid_idx.copy().reshape((valid_idx.shape[0], 1))
            oof_.append(dex.hstack_array([index, proba]))
            models.append(fold_est)

        oof_ = dex.vstack_array(oof_)
        oof_df = dd.from_dask_array(oof_).set_index(0)
        oof_ = oof_df.to_dask_array(lengths=True)

        self.cv_gbm_models_ = models

        if metrics is None:
            metrics = ['accuracy']
        if self.task == 'regression':
            proba = None
            preds = oof_
        else:
            proba = oof_
            preds = self.proba2predict(oof_)
            preds = da.take(np.array(self.classes_), preds, axis=0)
        scores = calc_score(y, preds, proba, metrics, self.task)
        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')
        return scores, oof_

    def fit(self, X, y, use_cache=None, verbose=0, **kwargs):
        starttime = time.time()
        if verbose is None:
            verbose = 0
        if verbose > 0:
            logger.info('estimator is transforming the train set')
        X = self.transform_data(X, y, fit=True, use_cache=use_cache, verbose=verbose)

        eval_set = kwargs.pop('eval_set', None)
        kwargs = self.fit_kwargs
        if eval_set is None:
            eval_set = kwargs.get('eval_set')
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                X_eval, y_eval = eval_set
                if verbose > 0:
                    logger.info('estimator is transforming the eval set')
                X_eval = self.transform_data(X_eval, use_cache=use_cache, verbose=verbose)
                kwargs['eval_set'] = [(X_eval, y_eval)]
            elif isinstance(eval_set, list):
                es = []
                for i, eval_set_ in enumerate(eval_set):
                    X_eval, y_eval = eval_set_
                    if verbose > 0:
                        logger.info(f'estimator is transforming the eval set({i})')
                    X_eval = self.transform_data(X_eval, use_cache=use_cache, verbose=verbose)
                    es.append((X_eval, y_eval))
                    kwargs['eval_set'] = es

        if kwargs.get('verbose') is None:
            kwargs['verbose'] = verbose

        if self.task != 'regression' and self.class_balancing is not None:
            sampler = get_sampler(self.class_balancing)
            if sampler is None:
                if verbose > 0:
                    logger.info('setting sample weight')
                sample_weight = self._get_sample_weight(y)
                kwargs['sample_weight'] = sample_weight
            else:
                if verbose > 0:
                    logger.info(f'sample balancing:{self.class_balancing}')
                X, y = sampler.fit_sample(X, y)

        if verbose > 0:
            logger.info('estimator is fitting the data')
        fit_kwargs = {**kwargs, 'verbose': 0}
        self.gbm_model.fit(X, y, **fit_kwargs)

        if self.classes_ is None and hasattr(self.gbm_model, 'classes_'):
            self.classes_ = self.gbm_model.classes_
        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')

    def _get_sample_weight(self, y):
        # unique = np.unique(y)
        # cw = list(class_weight.compute_class_weight('balanced', unique, y))
        # sample_weight = np.ones(y.shape)
        # for i, c in enumerate(unique):
        #     sample_weight[y == c] *= cw[i]
        # return sample_weight
        return dex.compute_sample_weight(y)

    def predict(self, X, use_cache=None, verbose=0, **kwargs):
        starttime = time.time()
        if verbose is None:
            verbose = 0

        if self.cv_gbm_models_ is not None:
            if self.task == 'regression':
                pred_sum = None
                for est in self.cv_gbm_models_:
                    pred = est.predict(X)
                    if pred_sum is None:
                        if dex.is_dask_object(X):
                            pred_sum = da.zeros_like(pred)
                        else:
                            pred_sum = np.zeros_like(pred)
                    pred_sum += pred
                preds = pred_sum / len(self.cv_gbm_models_)
            else:
                proba = self.predict_proba(X)
                preds = self.proba2predict(proba)
                if dex.is_dask_object(preds):
                    preds = da.take(np.array(self.classes_), preds, axis=0)
                else:
                    preds = np.array(self.classes_).take(preds, axis=0)
        else:
            X = self.transform_data(X, use_cache=use_cache, verbose=verbose)
            if verbose > 0:
                logger.info('estimator is predicting the data')
            preds = self.gbm_model.predict(X, **kwargs)

        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')
        return preds

    def predict_proba(self, X, use_cache=None, verbose=0, **kwargs):
        starttime = time.time()

        if verbose is None:
            verbose = 0
        X = self.transform_data(X, use_cache=use_cache, verbose=verbose)
        if verbose > 0:
            logger.info('estimator is predicting the data')
        if hasattr(self.gbm_model, 'predict_proba'):
            method = 'predict_proba'
        else:
            method = 'predict'

        if self.cv_gbm_models_ is not None:
            proba_sum = None
            for est in self.cv_gbm_models_:
                proba = getattr(est, method)(X)
                if proba_sum is None:
                    if dex.exist_dask_object(X):
                        proba_sum = da.zeros_like(proba)
                    else:
                        proba_sum = np.zeros_like(proba)
                proba_sum += proba
            proba = proba_sum / len(self.cv_gbm_models_)
        else:
            proba = getattr(self.gbm_model, method)(X)

        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')
        return proba

    def evaluate(self, X, y, metrics=None, use_cache=None, verbose=0, **kwargs):
        if metrics is None:
            metrics = ['accuracy']
        if self.task != 'regression':
            proba = self.predict_proba(X, use_cache=use_cache, verbose=verbose)
        else:
            proba = None
        preds = self.predict(X, use_cache=use_cache, verbose=verbose)
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
                        fs.rm(data_path, recursive=True)
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
                    fs.rm(pipeline_path, recursive=True)
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
                fs.rm(filepath, recursive=True)
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
                fs.rm(filepath, recursive=True)

    def get_explainer(self, data=None):
        explainer = HyperGBMExplainer(self, data=data)
        return explainer

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()
        # Don't pickle eval_set
        fit_kwargs = state.get('fit_kwargs')
        if fit_kwargs is not None and 'eval_set' in fit_kwargs.keys():
            fit_kwargs = fit_kwargs.copy()
            fit_kwargs.pop('eval_set')
            state['fit_kwargs'] = fit_kwargs
        return state


class HyperGBM(HyperModel):
    """
    HyperGBM
    """

    def __init__(self, searcher, dispatcher=None, callbacks=None, reward_metric='accuracy', task=None,
                 data_cleaner_params=None, cache_dir=None, clear_cache=True):
        """

        :param searcher: hypernets.searcher.Searcher
            A Searcher instance. Available searcher:
                - hypernets.searchers.RandomSearcher
                - hypernets.searcher.MCTSSearcher
                - hypernets.searchers.EvolutionSearcher
        :param dispatcher: hypernets.core.Dispatcher
            Dispatcher is used to provide different execution modes for search trials,
            such as in process mode (`InProcessDispatcher`), distributed parallel mode (`DaskDispatcher`), etc.
             `InProcessDispatcher` is used by default.
        :param callbacks: list of callback functions or None, optional (default=None)
            List of callback functions that are applied at each trial. See `hypernets.callbacks` for more information.
        :param reward_metric: str or None, optinal(default=accuracy)
            Set corresponding metric  according to task type to guide search direction of searcher.
        :param task: str or None, optinal(default=None)
            Task type. If None, inference the type of task automatically
            Possible values:
                - 'binary'
                - 'multiclass'
                - 'regression'
        :param data_cleaner_params: dict, (default=None)
            dictionary of parameters to initialize the `DataCleaner` instance. If None, `DataCleaner` will initialized with
            default values.
        :param cache_dir: str or None, (default=None)
            Path of data cache. If None, uses 'working directory/tmp/cache' as cache dir
        :param clear_cache: bool, (default=True)
            Whether clear the cache dir before searching
        """
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

    def export_trial_configuration(self, trial):
        return '`export_trial_configuration` does not implemented'

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
