# -*- coding:utf-8 -*-
"""

"""
from hypernets.model.estimator import Estimator
from hypernets.model.hyper_model import HyperModel
from hypergbm.sklearn_ex import DataCleaner
from .transformers import *
from .estimators import HyperEstimator
from sklearn import pipeline as sk_pipeline
from .metrics import calc_score
import pickle
import time

import re
import os
import shutil
import hashlib
import pandas as pd

try:
    import shap
    from shap import TreeExplainer

    has_shap = True
except:
    has_shap = False


class HyperGBMModel():
    def __init__(self, data_pipeline, pipeline_signature, estimator, task, cache_dir,
                 clear_cache=True, data_cleaner=None, fit_kwargs=None):
        self.data_pipeline = data_pipeline
        self.pipeline_signature = pipeline_signature
        self.estimator = estimator
        self.task = task
        self.data_cleaner = data_cleaner
        self.fit_kwargs = fit_kwargs
        self.cache_dir = self._prepare_cache_dir(cache_dir, clear_cache)

    def fit(self, X, y, **kwargs):
        X = self.transform_data(X, y, fit=True, **kwargs)
        if self.fit_kwargs is not None:
            kwargs = self.fit_kwargs

        starttime = time.time()
        print('Estimator is fitting the data')
        self.estimator.fit(X, y, **kwargs)
        print(f'Taken {time.time() - starttime}s')

    def transform_data(self, X, y=None, fit=False, **kwargs):
        use_cache = kwargs.get('use_cache')
        if use_cache is None:
            use_cache = True
        if use_cache:
            X_cache = self.get_X_from_cache(X, load_pipeline=True)
        else:
            X_cache = None

        if X_cache is None:
            starttime = time.time()
            if fit:
                if self.data_cleaner is not None:
                    print('Cleaning')
                    X, y = self.data_cleaner.fit_transform(X, y)
                print('Fitting and transforming')
                X = self.data_pipeline.fit_transform(X, y)
            else:
                if self.data_cleaner is not None:
                    print('Cleaning')
                    X, _ = self.data_cleaner.transform(X)
                print('Transforming')
                X = self.data_pipeline.transform(X)

            print(f'Taken {time.time() - starttime}s')
            if use_cache:
                self.save_X_to_cache(X, save_pipeline=True)
        else:
            X = X_cache

        return X

    def predict(self, X, **kwargs):
        X = self.transform_data(X, **kwargs)
        starttime = time.time()
        print('Estimator is predicting the data')
        preds = self.estimator.predict(X, **kwargs)
        print(f'Taken {time.time() - starttime}s')
        return preds

    def proba2predict(self, proba, proba_threshold=0.5):
        if self.task != 'classification':
            return proba
        if proba.shape[-1] > 2:
            predict = proba.argmax(axis=-1)
        elif proba.shape[-1] == 2:
            predict = (proba[:, 1] > proba_threshold).astype('int32')
        else:
            predict = (proba > proba_threshold).astype('int32')
        return predict

    def predict_proba(self, X, **kwargs):
        X = self.transform_data(X, **kwargs)
        starttime = time.time()
        print('Estimator is predicting probability')
        proba = self.estimator.predict_proba(X, **kwargs)
        print(f'Taken {time.time() - starttime}s')
        return proba

    def evaluate(self, X, y, metrics=None, **kwargs):
        if metrics is None:
            metrics = ['accuracy']
        proba = self.predict_proba(X, **kwargs)
        preds = self.predict(X, **kwargs)
        scores = calc_score(y, preds, proba, metrics, self.task)
        return scores

    def _prepare_cache_dir(self, cache_dir, clear_cache):
        if cache_dir is None:
            cache_dir = 'tmp/cache'
        if cache_dir[-1] == '/':
            cache_dir = cache_dir[:-1]

        cache_dir = os.path.expanduser(f'{cache_dir}')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        else:
            if clear_cache:
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir)
        return cache_dir

    def get_X_filepath(self, X):
        file_path = f'{self.cache_dir}/{X.shape[0]}_{X.shape[1]}_{self.pipeline_signature}.h5'
        return file_path

    def get_pipeline_filepath(self, X):
        file_path = f'{self.cache_dir}/{X.shape[0]}_{X.shape[1]}_pipeline_{self.pipeline_signature}.pkl'
        return file_path

    def get_X_from_cache(self, X, load_pipeline=False):
        file_path = self.get_X_filepath(X)
        if os.path.exists(file_path):
            df = self.load_df(file_path)
            if load_pipeline:
                pipeline_filepath = self.get_pipeline_filepath(X)
                try:
                    with open(f'{pipeline_filepath}', 'rb') as input:
                        self.data_pipeline, self.data_cleaner = pickle.load(input)
                except:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return None
            return df
        else:
            return None

    def save_X_to_cache(self, X, save_pipeline=False):
        file_path = self.get_X_filepath(X)
        self.save_df(file_path, X)
        if save_pipeline:
            pipeline_file_path = self.get_pipeline_filepath(X)
            try:
                with open(f'{pipeline_file_path}', 'wb') as output:
                    pickle.dump((self.data_pipeline, self.data_cleaner), output, protocol=2)
            except Exception as e:
                print(e)
                if os.path.exists(pipeline_file_path):
                    os.remove(pipeline_file_path)

    def load_df(self, filepath):
        global h5
        try:
            h5 = pd.HDFStore(filepath)
            df = h5['data']
            return df
        except:
            if os.path.exists(filepath):
                os.remove(filepath)
        finally:
            h5.close()

    def save_df(self, filepath, df):
        try:
            df.to_hdf(filepath, key='data', mode='w', format='t')
        except Exception as e:
            print(e)
            # traceback.print_exc()
            if os.path.exists(filepath):
                os.remove(filepath)

    def summary(self):
        s = f"{self.data_pipeline.__repr__(1000000)}\r\n{self.estimator.__repr__()}"
        return s

    def save_model(self, filepath):
        with open(f'{filepath}', 'wb') as output:
            pickle.dump(self, output, protocol=2)

    @staticmethod
    def load_model(filepath):
        with open(f'{filepath}', 'rb') as input:
            model = pickle.load(input)
            return model

    def get_explainer(self, data=None):
        explainer = HyperGBMExplainer(self, data=data)
        return explainer


class HyperGBMExplainer:
    def __init__(self, hypergbm_model, data=None):
        if not has_shap:
            raise RuntimeError('Please install `shap` package first. command: pip install shap')
        self.hypergbm_model = hypergbm_model
        if data is not None:
            data = self.hypergbm_model.transform_data(data)
        self.explainer = TreeExplainer(self.hypergbm_model.estimator, data)

    @property
    def expected_value(self):
        return self.explainer.expected_value

    def shap_values(self, X, y=None, tree_limit=None, approximate=False, check_additivity=True, from_call=False,
                    **kwargs):
        X = self.hypergbm_model.transform_data(X, **kwargs)
        return self.explainer.shap_values(X, y, tree_limit=tree_limit, approximate=approximate,
                                          check_additivity=check_additivity, from_call=from_call)

    def shap_interaction_values(self, X, y=None, tree_limit=None, **kwargs):
        X = self.hypergbm_model.transform_data(X, **kwargs)
        return self.explainer.shap_interaction_values(X, y, tree_limit)

    def transform_data(self, X, **kwargs):
        X = self.hypergbm_model.transform_data(X, **kwargs)
        return X


class HyperGBMEstimator(Estimator):
    def __init__(self, task, space_sample, data_cleaner_params=None, cache_dir=None, clear_cache=True):
        self.pipeline = None
        self.cache_dir = cache_dir
        self.clear_cache = clear_cache
        self.task = task
        self.data_cleaner_params = data_cleaner_params
        Estimator.__init__(self, space_sample=space_sample)

    def _build_model(self, space_sample):
        space, _ = space_sample.compile_and_forward()

        outputs = space.get_outputs()
        assert len(outputs) == 1, 'The space can only contains 1 output.'
        assert isinstance(outputs[0], HyperEstimator), 'The output of space must be `HyperEstimator`.'
        estimator = outputs[0].estimator
        fit_kwargs = outputs[0].fit_kwargs

        pipeline_module = space.get_inputs(outputs[0])
        assert len(pipeline_module) == 1, 'The `HyperEstimator` can only contains 1 input.'
        assert isinstance(pipeline_module[0],
                          ComposeTransformer), 'The upstream node of `HyperEstimator` must be `ComposeTransformer`.'
        # next, (name, p) = pipeline_module[0].compose()
        self.pipeline = self.build_pipeline(space, pipeline_module[0])
        pipeline_signature = self.get_pipeline_signature(self.pipeline)
        if self.data_cleaner_params is not None:
            data_cleaner = DataCleaner(**self.data_cleaner_params)
        else:
            data_cleaner = None
        model = HyperGBMModel(self.pipeline, pipeline_signature, estimator, self.task, self.cache_dir, self.clear_cache,
                              data_cleaner, fit_kwargs)
        return model

    def get_pipeline_signature(self, pipeline):
        repr = self.pipeline.__repr__(1000000)
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
        self.model.summary()

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def evaluate(self, X, y, **kwargs):
        scores = self.model.evaluate(X, y, **kwargs)
        return scores


class HyperGBM(HyperModel):
    def __init__(self, searcher, task='classification', dispatcher=None, callbacks=None, reward_metric='accuracy',
                 data_cleaner_params=None, cache_dir=None, clear_cache=True):
        if callbacks is None:
            callbacks = []
        self.task = task
        self.data_cleaner_params = data_cleaner_params
        self.cache_dir = cache_dir
        self.clear_cache = clear_cache
        HyperModel.__init__(self, searcher, dispatcher=dispatcher, callbacks=callbacks, reward_metric=reward_metric)

    def _get_estimator(self, space_sample):
        estimator = HyperGBMEstimator(task=self.task, space_sample=space_sample,
                                      data_cleaner_params=self.data_cleaner_params,
                                      cache_dir=self.cache_dir, clear_cache=self.clear_cache)
        return estimator

    def export_trail_configuration(self, trail):
        return '`export_trail_configuration` does not implemented'

    def blend_models(self, samples, X, y, **kwargs):
        models = []
        for sample in samples:
            estimator = self.final_train(sample, X, y, **kwargs)
            models.append(estimator.model)
        blends = BlendModel(models, self.task)
        return blends


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
        if self.task != 'classification':
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
        with open(f'{filepath}', 'wb') as output:
            pickle.dump(self, output)

    @staticmethod
    def load_model(filepath):
        with open(f'{filepath}', 'rb') as input:
            model = pickle.load(input)
            return model
