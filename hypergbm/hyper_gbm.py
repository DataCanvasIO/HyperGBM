# -*- coding:utf-8 -*-
"""

"""
import copy
import hashlib
import numpy as np
import pandas as pd
import pickle
import re
import time
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
from sklearn import pipeline as sk_pipeline
from sklearn.inspection import permutation_importance as sk_pi
from sklearn.utils import Bunch
from tqdm.auto import tqdm

from hypergbm.gbm_callbacks import FileMonitorCallback
from hypernets.core import Callback, ProgressiveCallback
from hypernets.model.estimator import Estimator
from hypernets.model.hyper_model import HyperModel
from hypernets.pipeline.base import ComposeTransformer
from hypernets.tabular import get_tool_box
from hypernets.tabular.cache import cache
from hypernets.utils import logging, fs, const
from .cfg import HyperGBMCfg as cfg
from .estimators import HyperEstimator

try:
    import shap
    from shap import TreeExplainer

    has_shap = True
except:
    has_shap = False

logger = logging.get_logger(__name__)

GB = 1024 ** 3


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

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__

        state = state.copy()
        state['pbar'] = None

        return state


class HyperGBMExplainer:
    def __init__(self, hypergbm_estimator, data=None):
        if not has_shap:
            raise RuntimeError('Please install `shap` package first. command: pip install shap')
        self.hypergbm_estimator = hypergbm_estimator
        if data is not None:
            data = self.hypergbm_estimator.transform_data(data)
        self.explainer = TreeExplainer(self.hypergbm_estimator.model, data)

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
    def __init__(self, task, reward_metric, space_sample, data_cleaner_params=None):
        super(HyperGBMEstimator, self).__init__(space_sample=space_sample, task=task)
        self.data_cleaner_params = data_cleaner_params
        self.reward_metric = reward_metric

        # built
        # self.gbm_model = None
        self.class_balancing = None
        self.fit_kwargs = None
        self.data_pipeline = None
        self.pipeline_signature = None

        # fitted
        self.data_cleaner = None
        # self.cv_gbm_models_ = None
        self.classes_ = None
        self.pos_label = None
        self.transients_ = {}

        if space_sample is not None:
            self._build_model(space_sample)

    @property
    def gbm_model(self):
        return self.model

    @property
    def cv_gbm_models_(self):
        return self.cv_models_

    def _build_model(self, space_sample):
        space, _ = space_sample.compile_and_forward()

        outputs = space.get_outputs()
        assert len(outputs) == 1, 'The space can only contains 1 output.'
        assert isinstance(outputs[0], HyperEstimator), 'The output of space must be `HyperEstimator`.'
        if outputs[0].estimator is None:
            outputs[0].build_estimator(self.task)
        self.model = outputs[0].estimator
        self.class_balancing = outputs[0].class_balancing
        self.fit_kwargs = outputs[0].fit_kwargs

        pipeline_module = space.get_inputs(outputs[0])
        assert len(pipeline_module) == 1, 'The `HyperEstimator` can only contains 1 input.'
        assert isinstance(pipeline_module[0], ComposeTransformer), \
            'The upstream node of `HyperEstimator` must be `ComposeTransformer`.'
        # next, (name, p) = pipeline_module[0].compose()
        self.data_pipeline = self._build_pipeline(space, pipeline_module[0])
        if logger.is_debug_enabled():
            logger.debug(f'data_pipeline:{self.data_pipeline}')
        self.pipeline_signature = self.get_pipeline_signature(self.data_pipeline)

    @staticmethod
    def get_pipeline_signature(pipeline):
        repr = pipeline.__repr__(1000000)
        repr = re.sub(r'object at 0x(.*)>', "", repr)
        md5 = hashlib.md5(repr.encode('utf-8')).hexdigest()
        return md5

    def _build_pipeline(self, space, last_transformer):
        transformers = []
        while True:
            next, (name, p) = last_transformer.compose()
            transformers.insert(0, (name, p))
            inputs = space.get_inputs(next)
            if inputs == space.get_inputs():
                break
            assert len(inputs) == 1, 'The `ComposeTransformer` can only contains 1 input.'
            assert isinstance(inputs[0], ComposeTransformer), \
                'The upstream node of `ComposeTransformer` must be `ComposeTransformer`.'
            last_transformer = inputs[0]
        assert len(transformers) > 0
        if len(transformers) == 1:
            return transformers[0][1]
        else:
            pipeline = self._create_pipeline(transformers)
            return pipeline

    @staticmethod
    def _create_pipeline(steps):
        return sk_pipeline.Pipeline(steps=steps)

    def summary(self):
        s = f"{self.data_pipeline.__repr__(1000000)}"
        # s = f"{self.data_pipeline.__repr__(1000000)}\r\n{self.gbm_model.__repr__()}"
        return s

    @cache(arg_keys='X,y', attr_keys='data_cleaner_params,pipeline_signature',
           attrs_to_restore='data_cleaner,data_pipeline',
           transformer='transform_data')
    def fit_transform_data(self, X, y=None, verbose=0):
        starttime = time.time()

        if self.data_cleaner_params is not None:
            self.data_cleaner = get_tool_box(X).transformers['DataCleaner'](**self.data_cleaner_params)

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
                             shuffle=False, random_state=9527, metrics=None, skip_if_file=None, **kwargs):
        starttime = time.time()
        tb = get_tool_box(X, y)
        tb.gc()

        if verbose is None:
            verbose = 0
        if verbose > 0:
            logger.info(f'transforming the train set, memory free:{tb.memory_free() / GB:.3f}')

        pbar = self.transients_.get('pbar')
        if pbar is not None:
            pbar.reset()
            pbar.set_description('fit_transform_data')

        X = self.fit_transform_data(X, y, verbose=verbose)

        cross_validator = kwargs.pop('cross_validator', None)
        if cross_validator is not None:
            iterators = cross_validator
        else:
            if stratified and self.task == const.TASK_BINARY:
                iterators = tb.statified_kfold(n_splits=num_folds, shuffle=True, random_state=9527)
            else:
                iterators = tb.kfold(n_splits=num_folds, shuffle=True, random_state=9527)

        # kwargs = self.fit_kwargs.copy()
        # if kwargs.get('verbose') is None:
        #     kwargs['verbose'] = verbose

        if metrics is None:
            metrics = [self.reward_metric] if self.reward_metric is not None else ['accuracy']

        oof_ = []
        oof_scores = []
        cv_models_ = []
        x_vals = []
        y_vals = []
        X_trains = []
        y_trains = []
        self.pos_label = pos_label
        if pbar is not None:
            pbar.set_description('cross_validation')
        sel = tb.select_1d
        for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X, y)):
            if verbose > 0:
                logger.info(f'fold {n_fold} started, memory free:{tb.memory_free() / GB:.3f}')
            x_train_fold, y_train_fold = sel(X, train_idx), sel(y, train_idx)
            x_val_fold, y_val_fold = sel(X, valid_idx), sel(y, valid_idx)

            fit_kwargs = self.fit_kwargs.copy()
            fit_kwargs['eval_set'] = [(x_val_fold, y_val_fold)]
            if self.reward_metric is not None and 'eval_reward_metric' not in fit_kwargs.keys():
                fit_kwargs['eval_reward_metric'] = self.reward_metric

            sample_weight = None
            if self.task != const.TASK_REGRESSION and self.class_balancing is not None:
                sampler = get_sampler(self.class_balancing)
                if sampler is None:
                    if verbose > 0:
                        logger.info(f'fold {n_fold} compute_sample_weight')
                    sample_weight = tb.compute_sample_weight(y_train_fold)
                else:
                    if verbose > 0:
                        logger.info(f'fold {n_fold} fit_resample')
                    x_train_fold, y_train_fold = sampler.fit_resample(x_train_fold, y_train_fold)
            fit_kwargs['sample_weight'] = sample_weight

            fold_est = copy.deepcopy(self.model)
            fold_est.group_id = f'{fold_est.__class__.__name__}_cv_{n_fold}'

            fit_kwargs['verbose'] = 0
            self._prepare_callbacks(fit_kwargs, fold_est, self.discriminator, skip_if_file)

            fold_start_at = time.time()
            tb.gc()
            if verbose > 0:
                logger.info(f'fold {n_fold} fitting estimator')
            fold_est.fit(x_train_fold, y_train_fold, **fit_kwargs)
            # print(fold_est.__class__)
            # print(fold_est.evals_result_)
            # print(f'fold {n_fold}, est:{fold_est.__class__},  best_n_estimators:{fold_est.best_n_estimators}')
            if self.classes_ is None and hasattr(fold_est, 'classes_'):
                self.classes_ = np.array(tb.to_local(fold_est.classes_)[0])

            if verbose > 0:
                logger.info(f'fold {n_fold} predict x_val_fold')
            if self.task == const.TASK_REGRESSION:
                proba = fold_est.predict(x_val_fold)
            else:
                proba = fold_est.predict_proba(x_val_fold)

            if verbose > 0:
                logger.info(f'fold {n_fold} get scores')
            fold_scores = self.get_scores(y_val_fold, proba, metrics)
            oof_scores.append(fold_scores)
            oof_.append((valid_idx, proba))
            cv_models_.append(fold_est)
            x_vals.append(x_val_fold)
            y_vals.append(y_val_fold)

            X_trains.append(x_train_fold)
            y_trains.append(y_train_fold)

            del fit_kwargs, sample_weight
            # del x_train_fold, y_train_fold, proba
            del proba
            tb.gc()

            if verbose > 0:
                logger.info(f'fold {n_fold} done with {time.time() - fold_start_at} seconds')
            if pbar is not None:
                pbar.update(1)

        logger.info(f'oof_scores:{oof_scores}')

        if verbose > 0:
            logger.info(f'get total scores')
        oof_ = tb.merge_oof(oof_)
        scores = self.get_scores(y, oof_, metrics)
        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')

        self.cv_ = True
        self.cv_models_ = cv_models_

        return scores, oof_, oof_scores, X_trains, y_trains, x_vals, y_vals

    def get_scores(self, y, oof_, metrics):
        tb = get_tool_box(y)
        y, proba = tb.select_valid_oof(y, oof_)
        if self.task == const.TASK_REGRESSION:
            preds = proba
            proba = None
        else:
            preds = self.proba2predict(proba)
            preds = tb.take_array(self.classes_, preds, axis=0)
        scores = tb.metrics.calc_score(y, preds, proba, metrics=metrics, task=self.task,
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

        if self.cv_:
            assert self.cv_models_ is not None and len(self.cv_models_) > 0
            for i, model in enumerate(self.cv_models_):
                get_scores(model, iteration_scores, i)
        else:
            get_scores(self.model, iteration_scores)
        return iteration_scores

    def fit(self, X, y, pos_label=None, skip_if_file=None, verbose=0, **kwargs):
        starttime = time.time()
        tb = get_tool_box(X, y)
        tb.gc()

        if verbose is None:
            verbose = 0
        if verbose > 0:
            logger.info(f'transforming the train set, memory free:{tb.memory_free() / GB:.3f}')

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
            if self.reward_metric is not None and 'eval_reward_metric' not in kwargs.keys():
                kwargs['eval_reward_metric'] = self.reward_metric

        if kwargs.get('verbose') is None:
            kwargs['verbose'] = verbose

        if self.task != const.TASK_REGRESSION and self.class_balancing is not None:
            sampler = get_sampler(self.class_balancing)
            if sampler is None:
                if verbose > 0:
                    logger.info('setting sample weight')
                sample_weight = tb.compute_sample_weight(y)
                kwargs['sample_weight'] = sample_weight
            else:
                if verbose > 0:
                    logger.info(f'sample balancing:{self.class_balancing}')
                X, y = sampler.fit_resample(X, y)

        if verbose > 0:
            logger.info('estimator is fitting the data')

        fit_kwargs = {**kwargs, 'verbose': 0}
        self.model.group_id = f'{self.model.__class__.__name__}'
        self._prepare_callbacks(fit_kwargs, self.model, self.discriminator, skip_if_file)

        tb.gc()
        self.model.fit(X, y, **fit_kwargs)

        if self.classes_ is None and hasattr(self.model, 'classes_'):
            self.classes_ = self.model.classes_
        self.pos_label = pos_label
        self.cv_ = False

        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')

        return self

    @staticmethod
    def _prepare_callbacks(fit_kwargs, est, discriminator, skip_if_file):
        if hasattr(est, 'build_discriminator_callback'):
            discriminator_callback = est.build_discriminator_callback(discriminator)
            if discriminator_callback:
                callbacks = fit_kwargs.get('callbacks', [])
                callbacks.append(discriminator_callback)
                fit_kwargs['callbacks'] = callbacks
        else:
            discriminator_callback = None  # dose not support callback

        if skip_if_file and discriminator_callback is not None:
            if hasattr(est, 'build_filemonitor_callback'):
                fm_callback = est.build_filemonitor_callback(skip_if_file)
            else:
                fm_callback = FileMonitorCallback(skip_if_file)

            if fm_callback is not None:
                callbacks = fit_kwargs.get('callbacks', [])
                callbacks.append(fm_callback)
                fit_kwargs['callbacks'] = callbacks

    def predict(self, X, verbose=0, **kwargs):
        starttime = time.time()
        if verbose is None:
            verbose = 0

        if self.cv_:
            assert self.cv_models_ is not None and len(self.cv_models_) > 0
            if self.task == const.TASK_REGRESSION:
                pred_sum = None
                X = self.transform_data(X, verbose=verbose)
                for est in self.cv_models_:
                    pred = est.predict(X)
                    if pred_sum is None:
                        pred_sum = pred
                    else:
                        pred_sum += pred
                preds = pred_sum / len(self.cv_models_)
            else:
                proba = self.predict_proba(X)
                preds = self.proba2predict(proba)
                preds = get_tool_box(preds).take_array(np.array(self.classes_), preds, axis=0)
        else:
            X = self.transform_data(X, verbose=verbose)
            if verbose > 0:
                logger.info('estimator is predicting the data')
            preds = self.model.predict(X, **kwargs)

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
        if hasattr(self.model, 'predict_proba'):
            method = 'predict_proba'
        else:
            method = 'predict'

        if self.cv_:
            assert self.cv_models_ is not None and len(self.cv_models_) > 0
            proba_sum = None
            for est in self.cv_models_:
                proba = getattr(est, method)(X)
                if proba_sum is None:
                    proba_sum = proba
                else:
                    proba_sum += proba
            proba = proba_sum / len(self.cv_models_)
        else:
            proba = getattr(self.model, method)(X)

        if verbose > 0:
            logger.info(f'taken {time.time() - starttime}s')
        return proba

    def evaluate(self, X, y, metrics=None, verbose=0, **kwargs):
        if metrics is None:
            metrics = [self.reward_metric] if self.reward_metric is not None else ['accuracy']

        if self.task != const.TASK_REGRESSION:
            proba = self.predict_proba(X, verbose=verbose)
        else:
            proba = None
        preds = self.predict(X, verbose=verbose)
        scores = get_tool_box(X).metrics.calc_score(y, preds, proba, metrics=metrics, task=self.task,
                                                    pos_label=self.pos_label, classes=self.classes_)
        return scores

    def is_data_pipeline_straightforward(self):
        excluded = cfg.straightforward_excluded
        if excluded is None or len(excluded) == 0:
            return True

        r = self.data_pipeline.__repr__(1000000)
        return all(map(lambda s: r.find(s) < 0, excluded))

    def permutation_importance(self, X, y, *,
                               scoring=None, n_repeats=5, n_jobs=None,
                               random_state=None, sample_weight=None, max_samples=1.0):
        """
        see: sklearn.inspection.permutation_importance
        """
        tb = get_tool_box(X, y)
        if isinstance(scoring, str):
            scoring = tb.metrics.metric_to_scoring(scoring)

        # optimize 'n_jobs' option
        if type(self.model).__name__.lower().find('catboost') >= 0:
            if n_jobs is None:
                n_jobs = -1
        else:
            if n_jobs == -1:
                n_jobs = None

        options = dict(random_state=random_state)
        if sample_weight is not None:
            options['sample_weight'] = sample_weight
        if max_samples != 1.0:
            options['max_samples'] = max_samples

        if logger.is_info_enabled():
            logger.info(f'calculate permutation_importance, n_jobs:{n_jobs}, n_repeats:{n_repeats},'
                        f' model:{type(self.model).__name__}, '
                        f' datapipeline:{self.data_pipeline}')

        if not self.is_data_pipeline_straightforward():
            logger.info(f'datapipeline is not straightforward, redirect calculation to sklearn')
            return sk_pi(self, X, y, scoring=scoring, n_repeats=n_repeats, n_jobs=n_jobs, **options)

        # preprocessing data
        columns_in = X.columns.to_list()
        X = self.transform_data(X)
        columns_out = X.columns.to_list()
        assert set(columns_in).issuperset(set(columns_out))

        # compute permutation_importance
        if self.cv_:
            assert self.cv_models_ is not None and len(self.cv_models_) > 0
            importances = []
            for est in (self.cv_models_ * n_repeats)[:n_repeats]:
                est_pi = sk_pi(est, X, y, scoring=scoring, n_repeats=1, n_jobs=n_jobs, **options)
                importances.append(est_pi.importances)
            importances = tb.hstack_array(importances)
            result = Bunch(
                importances_mean=np.mean(importances, axis=1),
                importances_std=np.std(importances, axis=1),
                importances=importances,
            )
        else:
            result = sk_pi(self.model, X, y, scoring=scoring, n_repeats=n_repeats, n_jobs=n_jobs, **options)

        # fix the result
        if columns_out != columns_in:
            df_importances = pd.DataFrame(result.importances, index=columns_out)
            df_importances_fake = pd.DataFrame(np.zeros((len(columns_in), n_repeats)), index=columns_in)
            importances = (df_importances + df_importances_fake).fillna(0.0).values
            result = Bunch(
                importances_mean=np.mean(importances, axis=1),
                importances_std=np.std(importances, axis=1),
                importances=importances,
            )
        return result

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

        state = state.copy()
        state['transients_'] = {}

        if 'discriminator' in state.keys():
            state['discriminator'] = None

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

    def __repr__(self):
        cv = False if self.cv_ is None else self.cv_
        r = f'{type(self).__name__}(' \
            f'task={self.task}, reward_metric={self.reward_metric}, cv={cv},\n' \
            f'data_pipeline: {self.data_pipeline}\n' \
            f'gbm_model: {self.model if cv is False else self.cv_models_[0]}\n' \
            f')'
        return r


class HyperGBMShapExplainer:

    def __init__(self, hypergbm_estimator: HyperGBMEstimator, data=None, **kwargs):

        if not has_shap:
            raise RuntimeError('Please install `shap` package first. command: pip install shap')
        self.hypergbm_estimator = hypergbm_estimator
        if data is not None:
            data = self.hypergbm_estimator.transform_data(data)

        if hypergbm_estimator.cv_ is True:
            self._explainers = [TreeExplainer(m, data=data, **kwargs) for m in hypergbm_estimator.cv_gbm_models_]
        else:
            self._explainers = [TreeExplainer(self.hypergbm_estimator.gbm_model, data=data, **kwargs)]

    @property
    def expected_values(self):
        if self.hypergbm_estimator.cv_ is True:
            return [_.expected_value for _ in self._explainers]
        else:
            return self._explainers[0].expected_value

    def __call__(self, X, transform_kwargs=None, **kwargs):
        """Calc explanation of X using shap tree method.

        Parameters
        ----------
        X
        transform_kwargs
        kwargs

        Returns
        -------
            For cv training, output type is List[Explanation], length is num folds of CV.
            For train-test split training, output type is Explanation. if it's a LightGBM training one
                classification task the output shape is (Xt_n_rows, Xt_n_cols, n_classes), for other algorithms
                output shape is (Xt_n_rows, Xt_n_cols)
        """

        if transform_kwargs is None:
            transform_kwargs = {}

        Xt = self.hypergbm_estimator.transform_data(X, **transform_kwargs)

        def f(explainer):
            # TO FIX: CatBoostError: 'data' is numpy array of floating point numerical type,
            # it means no categorical features,
            # but 'cat_features' parameter specifies nonzero number of categorical features
            setattr(explainer, 'data_feature_names', Xt.columns.tolist())
            # pd.Dataframe.values would change the dtype to be a lower-common-denominator dtype (implicit upcasting);
            # see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.values.html
            return explainer(Xt.to_numpy(dtype='object'), **kwargs)

        if self.hypergbm_estimator.cv_ is True:
            return [f(explainer) for explainer in self._explainers]
        else:
            return f(self._explainers[0])

    def transform_data(self, X, **kwargs):
        X = self.hypergbm_estimator.transform_data(X, **kwargs)
        return X


class HyperGBM(HyperModel):
    """
    HyperGBM
    """
    estimator_cls = HyperGBMEstimator

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
        estimator = self.estimator_cls(task=self.task, reward_metric=self.reward_metric,
                                       space_sample=space_sample,
                                       data_cleaner_params=self.data_cleaner_params)

        cbs = self.callbacks
        if isinstance(cbs, list) and len(cbs) > 0 and isinstance(cbs[-1], FitCrossValidationCallback):
            estimator.transients_['pbar'] = cbs[-1].pbar

        return estimator

    def load_estimator(self, model_file):
        assert model_file is not None
        return self.estimator_cls.load(model_file)

    def export_trial_configuration(self, trial):
        return '`export_trial_configuration` does not implemented'
