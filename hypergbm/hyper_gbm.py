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
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
from sklearn import pipeline as sk_pipeline
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.auto import tqdm

from hypernets.core import get_random_state
from hypergbm.pipeline import ComposeTransformer
from hypernets.core import Callback, ProgressiveCallback
from hypernets.model.estimator import Estimator
from hypernets.model.hyper_model import HyperModel
from hypernets.tabular import dask_ex as dex
from hypernets.tabular.cache import cache
from hypernets.tabular.data_cleaner import DataCleaner
from hypernets.tabular.lifelong_learning import select_valid_oof
from hypernets.tabular.metrics import calc_score
from hypernets.utils import logging, fs
from .estimators import HyperEstimator

try:
    import shap
    from shap import TreeExplainer

    has_shap = True
except:
    has_shap = False

logger = logging.get_logger(__name__)


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


class FitCrossValidationCallback(Callback):
    def __init__(self):
        super(FitCrossValidationCallback, self).__init__()

        self.pbar = None

    def on_search_start(self, hyper_model, X, y, X_eval, y_eval, cv, num_folds, max_trials, dataset_id, trial_store,
                        **fit_kwargs):
        if cv and num_folds > 1:
            self.pbar = tqdm(total=num_folds, leave=False, desc='fit_cross_validation')

    def on_search_end(self, hyper_model):
        if self.pbar is not None:
            self.pbar.update(self.pbar.total)
            self.pbar.close()
            self.pbar = None

    def on_search_error(self, hyper_model):
        self.on_search_end(hyper_model)


class HyperGBMExplainer:
    def __init__(self, hypergbm_estimator, data=None):
        if not has_shap:
            raise RuntimeError('Please install `shap` package first. command: pip install shap')
        self.hypergbm_estimator = hypergbm_estimator
        if data is not None:
            data = self.hypergbm_estimator.transform_data(data)
        self.explainer = TreeExplainer(self.hypergbm_estimator.gbm_model, data)

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
    def __init__(self, task, space_sample, data_cleaner_params=None):
        super(HyperGBMEstimator, self).__init__(space_sample=space_sample, task=task)
        self.data_pipeline = None
        self.data_cleaner_params = data_cleaner_params
        self.gbm_model = None
        self.cv_gbm_models_ = None
        self.data_cleaner = None
        self.pipeline_signature = None
        self.fit_kwargs = None
        self.class_balancing = None
        self.classes_ = None
        self.pos_label = None
        self.transients_ = {}

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
        # logger.debug(f'data_pipeline:{self.data_pipeline}')
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

    @cache(arg_keys='X,y', attr_keys='data_cleaner_params,pipeline_signature',
           attrs_to_restore='data_cleaner,data_pipeline',
           transformer='transform_data')
    def fit_transform_data(self, X, y=None, verbose=0):
        starttime = time.time()

        if self.data_cleaner is not None:
            if verbose > 0:
                logger.info('clean data')
            X, y = self.data_cleaner.fit_transform(X, y)

        if verbose > 0:
            logger.info('fit and transform')
        X = self.data_pipeline.fit_transform(X, y)

        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')

        return X

    def transform_data(self, X, y=None, verbose=0):
        starttime = time.time()

        if self.data_cleaner is not None:
            if verbose > 0:
                logger.info('clean data')
            X = self.data_cleaner.transform(X)

        if verbose > 0:
            logger.info('transform')
        X = self.data_pipeline.transform(X)

        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')

        return X

    def fit_cross_validation(self, X, y, verbose=0, stratified=True, num_folds=3, pos_label=None,
                             shuffle=False, random_state=9527, metrics=None, **kwargs):

        random_state = get_random_state().get_state()[1][0]
        if dex.exist_dask_object(X, y):
            return self.fit_cross_validation_by_dask(X, y, verbose=verbose, pos_label=pos_label,
                                                     stratified=stratified, num_folds=num_folds,
                                                     shuffle=shuffle, random_state=random_state,
                                                     metrics=metrics, **kwargs)
        starttime = time.time()
        if verbose is None:
            verbose = 0
        if verbose > 0:
            logger.info('transforming the train set')

        pbar = self.transients_.get('pbar')
        if pbar is not None:
            pbar.reset()
            pbar.set_description('fit_transform_data')

        X = self.fit_transform_data(X, y, verbose=verbose)
        y = np.array(y)

        cross_validator = kwargs.pop('cross_validator', None)
        if cross_validator is not None:
            iterators = cross_validator
        else:
            if stratified and self.task == 'binary':
                iterators = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
            else:
                iterators = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

        kwargs = self.fit_kwargs
        if kwargs.get('verbose') is None:
            kwargs['verbose'] = verbose

        if metrics is None:
            metrics = ['accuracy']
        oof_ = None
        oof_scores = []
        self.pos_label = pos_label
        self.cv_gbm_models_ = []
        if pbar is not None:
            pbar.set_description('cross_validation')
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
                    x_train_fold, y_train_fold = sampler.fit_resample(x_train_fold, y_train_fold)
            kwargs['sample_weight'] = sample_weight

            fold_est = copy.deepcopy(self.gbm_model)
            fold_est.group_id = f'{fold_est.__class__.__name__}_cv_{n_fold}'
            fit_kwargs = {**kwargs, 'verbose': 0}
            if hasattr(fold_est, 'build_discriminator_callback'):
                callback = fold_est.build_discriminator_callback(self.discriminator)
                if callback:
                    callbacks = fit_kwargs.get('callbacks', [])
                    callbacks.append(callback)
                    fit_kwargs['callbacks'] = callbacks

            fold_est.fit(x_train_fold, y_train_fold, **fit_kwargs)
            # print(fold_est.__class__)
            # print(fold_est.evals_result_)
            # print(f'fold {n_fold}, est:{fold_est.__class__},  best_n_estimators:{fold_est.best_n_estimators}')
            if self.classes_ is None and hasattr(fold_est, 'classes_'):
                self.classes_ = fold_est.classes_
            if self.task == 'regression':
                proba = fold_est.predict(x_val_fold)
            else:
                proba = fold_est.predict_proba(x_val_fold)

            fold_scores = self.get_scores(y_val_fold, proba, metrics)
            oof_scores.append(fold_scores)
            if oof_ is None:
                if len(proba.shape) == 1:
                    oof_ = np.full(y.shape, np.nan, proba.dtype)
                else:
                    oof_ = np.full((y.shape[0], proba.shape[-1]), np.nan, proba.dtype)
            oof_[valid_idx] = proba
            self.cv_gbm_models_.append(fold_est)

            if pbar is not None:
                pbar.update(1)

        logger.info(f'oof_scores:{oof_scores}')
        scores = self.get_scores(y, oof_, metrics)
        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')
        return scores, oof_, oof_scores

    def get_scores(self, y, oof_, metrics):
        y, proba = select_valid_oof(y, oof_)
        if self.task == 'regression':
            preds = proba
            proba = None
        else:
            preds = self.proba2predict(proba)
            preds = np.array(self.classes_).take(preds, axis=0)
        scores = calc_score(y, preds, proba, metrics=metrics, task=self.task,
                            classes=self.classes_, pos_label=self.pos_label)
        return scores

    def get_iteration_scores(self):
        iteration_scores = {}

        def get_scores(gbm_model, iteration_scores, fold=None, ):
            if hasattr(gbm_model, 'iteration_scores'):
                if gbm_model.__dict__.get('group_id'):
                    group_id = gbm_model.group_id
                else:
                    if fold is not None:
                        group_id = f'{gbm_model.__class__.__name__}_cv_{i}'
                    else:
                        group_id = gbm_model.__class__.__name__
                iteration_scores[group_id] = gbm_model.iteration_scores

        if self.cv_gbm_models_:
            for i, gbm_model in enumerate(self.cv_gbm_models_):
                get_scores(gbm_model, iteration_scores, i)
        else:
            get_scores(self.gbm_model, iteration_scores)
        return iteration_scores

    def fit_cross_validation_by_dask(self, X, y, verbose=0, stratified=True, num_folds=3, pos_label=None,
                                     shuffle=False, random_state=9527, metrics=None, **kwargs):
        starttime = time.time()
        if verbose is None:
            verbose = 0
        if verbose > 0:
            logger.info('estimator is transforming the train set')

        pbar = self.transients_.get('pbar')
        if pbar is not None:
            pbar.reset()
            pbar.set_description('fit_transform_data')

        X = self.fit_transform_data(X, y, verbose=verbose)

        iterators = dm_sel.KFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)
        X_values = X.to_dask_array(lengths=True)
        y_values = y.to_dask_array(lengths=True)

        kwargs = self.fit_kwargs
        # if kwargs.get('verbose') is None and str(type(self.gbm_model)).find('dask') < 0:
        #     kwargs['verbose'] = verbose

        oof_ = []
        models = []
        if pbar is not None:
            pbar.set_description('cross_validation')
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
                    x_train_fold, y_train_fold = sampler.fit_resample(x_train_fold, y_train_fold)

            eval_set = [(x_val_fold, y_val_fold)]
            if self.task != 'regression' and \
                    len(da.unique(y_val_fold).compute()) != len(da.unique(y_train_fold).compute()):
                eval_set = None

            fold_est = copy.deepcopy(self.gbm_model)
            fit_kwargs = {**kwargs, 'sample_weight': sample_weight, 'eval_set': eval_set, 'verbose': 0,
                          'early_stopping_rounds': kwargs.get('early_stopping_rounds')
                          if eval_set is not None else None}
            fold_est.group_id = f'{fold_est.__class__.__name__}_cv_{n_fold}'
            if hasattr(fold_est, 'build_discriminator_callback'):
                callback = fold_est.build_discriminator_callback(self.discriminator)
                if callback:
                    callbacks = fit_kwargs.get('callbacks', [])
                    callbacks.append(callback)
                    fit_kwargs['callbacks'] = callbacks
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

            if pbar is not None:
                pbar.update(1)

        oof_ = dex.vstack_array(oof_)
        oof_df = dd.from_dask_array(oof_).set_index(0)
        oof_ = oof_df.to_dask_array(lengths=True)

        self.cv_gbm_models_ = models
        self.pos_label = pos_label

        if metrics is None:
            metrics = ['accuracy']
        if self.task == 'regression':
            proba = None
            preds = oof_
        else:
            proba = oof_
            preds = self.proba2predict(oof_)
            preds = da.take(np.array(self.classes_), preds, axis=0)
        scores = calc_score(y, preds, proba, metrics=metrics, task=self.task,
                            classes=self.classes_, pos_label=pos_label)
        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')
        return scores, oof_, None

    def fit(self, X, y, pos_label=None, verbose=0, **kwargs):
        starttime = time.time()
        if verbose is None:
            verbose = 0
        if verbose > 0:
            logger.info('estimator is transforming the train set')
        X = self.fit_transform_data(X, y, verbose=verbose)

        eval_set = kwargs.pop('eval_set', None)
        kwargs = self.fit_kwargs
        if eval_set is None:
            eval_set = kwargs.get('eval_set')
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                X_eval, y_eval = eval_set
                if verbose > 0:
                    logger.info('estimator is transforming the eval set')
                X_eval = self.transform_data(X_eval, verbose=verbose)
                kwargs['eval_set'] = [(X_eval, y_eval)]
            elif isinstance(eval_set, list):
                es = []
                for i, eval_set_ in enumerate(eval_set):
                    X_eval, y_eval = eval_set_
                    if verbose > 0:
                        logger.info(f'estimator is transforming the eval set({i})')
                    X_eval = self.transform_data(X_eval, verbose=verbose)
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
                X, y = sampler.fit_resample(X, y)

        if verbose > 0:
            logger.info('estimator is fitting the data')

        fit_kwargs = {**kwargs, 'verbose': 0}
        self.gbm_model.group_id = f'{self.gbm_model.__class__.__name__}'
        if hasattr(self.gbm_model, 'build_discriminator_callback'):
            callback = self.gbm_model.build_discriminator_callback(self.discriminator)
            if callback:
                callbacks = fit_kwargs.get('callbacks', [])
                callbacks.append(callback)
                fit_kwargs['callbacks'] = callbacks

        self.gbm_model.fit(X, y, **fit_kwargs)

        if self.classes_ is None and hasattr(self.gbm_model, 'classes_'):
            self.classes_ = self.gbm_model.classes_
        self.pos_label = pos_label

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

    def predict(self, X, verbose=0, **kwargs):
        starttime = time.time()
        if verbose is None:
            verbose = 0

        if self.cv_gbm_models_ is not None:
            if self.task == 'regression':
                pred_sum = None
                X = self.transform_data(X, verbose=verbose)
                for est in self.cv_gbm_models_:
                    pred = est.predict(X)
                    if pred_sum is None:
                        pred_sum = pred
                    else:
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
            X = self.transform_data(X, verbose=verbose)
            if verbose > 0:
                logger.info('estimator is predicting the data')
            preds = self.gbm_model.predict(X, **kwargs)

        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')
        return preds

    def predict_proba(self, X, verbose=0, **kwargs):
        starttime = time.time()

        if verbose is None:
            verbose = 0
        X = self.transform_data(X, verbose=verbose)
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
                    proba_sum = proba
                else:
                    proba_sum += proba
            proba = proba_sum / len(self.cv_gbm_models_)
        else:
            proba = getattr(self.gbm_model, method)(X)

        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')
        return proba

    def evaluate(self, X, y, metrics=None, verbose=0, **kwargs):
        if metrics is None:
            metrics = ['accuracy']
        if self.task != 'regression':
            proba = self.predict_proba(X, verbose=verbose)
        else:
            proba = None
        preds = self.predict(X, verbose=verbose)
        scores = calc_score(y, preds, proba, metrics=metrics, task=self.task,
                            pos_label=self.pos_label, classes=self.classes_)
        return scores

    def save(self, model_file):
        with fs.open(f'{model_file}', 'wb') as output:
            pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(model_file):
        with fs.open(f'{model_file}', 'rb') as input:
            model = pickle.load(input)
            return model

    def get_explainer(self, data=None):
        explainer = HyperGBMExplainer(self, data=data)
        return explainer

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()
        # Don't pickle eval_set and sample_weight
        state['transients_'] = {}

        fit_kwargs = state.get('fit_kwargs')
        if fit_kwargs is not None and 'eval_set' in fit_kwargs.keys():
            fit_kwargs = fit_kwargs.copy()
            fit_kwargs.pop('eval_set')
            state['fit_kwargs'] = fit_kwargs
        if fit_kwargs is not None and 'sample_weight' in fit_kwargs.keys():
            fit_kwargs = fit_kwargs.copy()
            fit_kwargs.pop('sample_weight')
            state['fit_kwargs'] = fit_kwargs

        return state


class HyperGBM(HyperModel):
    """
    HyperGBM
    """

    def __init__(self, searcher, dispatcher=None, callbacks=None, reward_metric='accuracy', task=None,
                 discriminator=None, data_cleaner_params=None, cache_dir=None, clear_cache=None):
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
        :param cache_dir: deprecated
        :param clear_cache: deprecated
        """
        self.data_cleaner_params = data_cleaner_params

        if callbacks is not None and any([isinstance(cb, ProgressiveCallback) for cb in callbacks]):
            callbacks = list(callbacks) + [FitCrossValidationCallback()]

        HyperModel.__init__(self, searcher, dispatcher=dispatcher, callbacks=callbacks, reward_metric=reward_metric,
                            task=task, discriminator=discriminator)

    def _get_estimator(self, space_sample):
        estimator = HyperGBMEstimator(task=self.task, space_sample=space_sample,
                                      data_cleaner_params=self.data_cleaner_params)

        cbs = self.callbacks
        if isinstance(cbs, list) and len(cbs) > 0 and isinstance(cbs[-1], FitCrossValidationCallback):
            estimator.transients_['pbar'] = cbs[-1].pbar

        return estimator

    def load_estimator(self, model_file):
        assert model_file is not None
        return HyperGBMEstimator.load(model_file)

    def export_trial_configuration(self, trial):
        return '`export_trial_configuration` does not implemented'
