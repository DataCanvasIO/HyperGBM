# -*- coding:utf-8 -*-
__author__ = 'yangjian'

from sklearn.base import BaseEstimator

"""

"""
import copy

import numpy as np
import pandas as pd
from IPython.display import display, display_markdown
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from hypernets.experiment import Experiment
from hypernets.utils import logging
from hypernets.utils.common import isnotebook
from tabular_toolbox import dask_ex as dex
from tabular_toolbox import drift_detection as dd
from tabular_toolbox.data_cleaner import DataCleaner
from tabular_toolbox.ensemble import GreedyEnsemble, DaskGreedyEnsemble
from tabular_toolbox.feature_selection import select_by_multicollinearity
from tabular_toolbox.utils import load_data, infer_task_type
from .feature_importance import feature_importance_batch

logger = logging.get_logger(__name__)

DEFAULT_EVAL_SIZE = 0.3
DEFAULT_TARGET_SET = {'y', 'target'}

_is_notebook = isnotebook()


def _set_log_level(log_level):
    logging.set_level(log_level)

    from tabular_toolbox.utils import logging as tlogging
    tlogging.set_level(log_level)

    # if log_level >= logging.ERROR:
    #     import logging as pylogging
    #     pylogging.basicConfig(level=log_level)


class ExperimentStep(BaseEstimator):
    def __init__(self, experiment, name):
        super(ExperimentStep, self).__init__()

        self.name = name
        self.experiment = experiment

    def step_start(self, *args, **kwargs):
        if self.experiment is not None:
            self.experiment.step_start(*args, **kwargs)

    def step_end(self, *args, **kwargs):
        if self.experiment is not None:
            self.experiment.step_end(*args, **kwargs)

    def step_progress(self, *args, **kwargs):
        if self.experiment is not None:
            self.experiment.step_progress(*args, **kwargs)

    @property
    def task(self):
        return self.experiment.task if self.experiment is not None else None

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        raise NotImplemented()
        # return hyper_model, X_train, y_train, X_test, X_eval, y_eval,

    def transform(self, X, y=None, **kwargs):
        raise NotImplemented()
        # return X

    def is_transform_skipped(self):
        return False

    # override this to remove 'experiment' from estimator __expr__
    @classmethod
    def _get_param_names(cls):
        params = super()._get_param_names()
        return filter(lambda x: x != 'experiment', params)

    def __getstate__(self):
        state = super().__getstate__()
        # Don't pickle experiment
        if 'experiment' in state.keys():
            state['experiment'] = None
        return state


class FeatureSelectStep(ExperimentStep):

    def __init__(self, experiment, name):
        super().__init__(experiment, name)

        # fitted
        self.selected_features_ = None

    def transform(self, X, y=None, **kwargs):
        if self.selected_features_ is not None:
            if logger.is_debug_enabled():
                msg = f'{self.name} transform from {len(X.columns.tolist())} to {len(self.selected_features_)} features'
                logger.debug(msg)
            X = X[self.selected_features_]
        return X

    def is_transform_skipped(self):
        return self.selected_features_ is None


class DataCleanStep(ExperimentStep):
    def __init__(self, experiment, name, data_cleaner_args=None,
                 cv=False, train_test_split_strategy=None, random_state=None):
        super().__init__(experiment, name)

        self.data_cleaner_args = data_cleaner_args if data_cleaner_args is not None else {}
        self.cv = cv
        self.train_test_split_strategy = train_test_split_strategy
        self.random_state = random_state

        # fitted
        self.selected_features_ = None
        self.data_cleaner = None

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        self.step_start('clean and split data')
        # 1. Clean Data
        if self.cv and X_eval is not None and y_eval is not None:
            X_train = pd.concat([X_train, X_eval], axis=0)
            y_train = pd.concat([y_train, y_eval], axis=0)
            X_eval = None
            y_eval = None

        data_cleaner = DataCleaner(**self.data_cleaner_args)

        X_train, y_train = data_cleaner.fit_transform(X_train, y_train)
        self.step_progress('fit_transform train set')

        if X_test is not None:
            X_test = data_cleaner.transform(X_test)
            self.step_progress('transform X_test')

        if not self.cv:
            if X_eval is None or y_eval is None:
                eval_size = kwargs.get('eval_size', DEFAULT_EVAL_SIZE)
                if self.train_test_split_strategy == 'adversarial_validation' and X_test is not None:
                    logger.debug('DriftDetector.train_test_split')
                    detector = dd.DriftDetector()
                    detector.fit(X_train, X_test)
                    X_train, X_eval, y_train, y_eval = detector.train_test_split(X_train, y_train, test_size=eval_size)
                else:
                    if self.task == 'regression' or dex.is_dask_object(X_train):
                        X_train, X_eval, y_train, y_eval = dex.train_test_split(X_train, y_train, test_size=eval_size,
                                                                                random_state=self.random_state)
                    else:
                        X_train, X_eval, y_train, y_eval = dex.train_test_split(X_train, y_train, test_size=eval_size,
                                                                                random_state=self.random_state,
                                                                                stratify=y_train)
                if self.task != 'regression':
                    y_train_uniques = set(y_train.unique()) if hasattr(y_train, 'unique') else set(y_train)
                    y_eval_uniques = set(y_eval.unique()) if hasattr(y_eval, 'unique') else set(y_eval)
                    assert y_train_uniques == y_eval_uniques, \
                        'The classes of `y_train` and `y_eval` must be equal. Try to increase eval_size.'
                self.step_progress('split into train set and eval set')
            else:
                X_eval, y_eval = data_cleaner.transform(X_eval, y_eval)
                self.step_progress('transform eval set')

        self.step_end(output={'X_train.shape': X_train.shape,
                              'y_train.shape': y_train.shape,
                              'X_eval.shape': None if X_eval is None else X_eval.shape,
                              'y_eval.shape': None if y_eval is None else y_eval.shape,
                              'X_test.shape': None if X_test is None else X_test.shape})

        if _is_notebook:
            display_markdown('### Data Cleaner', raw=True)

            display(data_cleaner, display_id='output_cleaner_info1')
            display_markdown('### Train set & Eval set', raw=True)

            display_data = (X_train.shape,
                            y_train.shape,
                            X_eval.shape if X_eval is not None else None,
                            y_eval.shape if y_eval is not None else None,
                            X_test.shape if X_test is not None else None)
            if dex.exist_dask_object(X_train, y_train, X_eval, y_eval, X_test):
                display_data = [dex.compute(shape)[0] for shape in display_data]
            display(pd.DataFrame([display_data],
                                 columns=['X_train.shape',
                                          'y_train.shape',
                                          'X_eval.shape',
                                          'y_eval.shape',
                                          'X_test.shape']), display_id='output_cleaner_info2')
        original_features = X_train.columns.to_list()

        self.selected_features_ = original_features
        self.data_cleaner = data_cleaner

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def transform(self, X, y=None, **kwargs):
        return self.data_cleaner.transform(X, y, **kwargs)


class MulticollinearityDetectStep(FeatureSelectStep):

    def __init__(self, experiment, name, drop_feature_with_collinearity=True):
        super().__init__(experiment, name)

        self.drop_feature_with_collinearity = drop_feature_with_collinearity

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        if self.drop_feature_with_collinearity:
            if _is_notebook:
                display_markdown('### Drop features with collinearity', raw=True)

            self.step_start('drop features with multicollinearity')
            corr_linkage, remained, dropped = select_by_multicollinearity(X_train)
            self.output_multi_collinearity_ = {
                'corr_linkage': corr_linkage,
                'remained': remained,
                'dropped': dropped
            }
            self.step_progress('calc correlation')

            self.selected_features_ = remained

            X_train = X_train[self.selected_features_]
            if X_eval is not None:
                X_eval = X_eval[self.selected_features_]
            if X_test is not None:
                X_test = X_test[self.selected_features_]
            self.step_progress('drop features')
            self.step_end(output=self.output_multi_collinearity_)

            if _is_notebook:
                display(pd.DataFrame([(k, v)
                                      for k, v in self.output_multi_collinearity_.items()],
                                     columns=['key', 'value']),
                        display_id='output_drop_feature_with_collinearity')

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval


class DriftDetectStep(FeatureSelectStep):

    def __init__(self, experiment, name, drift_detection=True):
        super().__init__(experiment, name)

        self.drift_detection = drift_detection

        # fitted
        self.output_drift_detection_ = None

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        if self.drift_detection and self.experiment.X_test is not None:
            if _is_notebook:
                display_markdown('### Drift detection', raw=True)

            self.step_start('detect drifting')
            features, history, scores = dd.feature_selection(X_train, X_test)

            if set(X_train.columns.to_list()) - set(features):
                self.selected_features_ = features
                X_train = X_train[features]
                if X_eval is not None:
                    X_eval = X_eval[features]
                if X_test is not None:
                    X_test = X_test[features]
            else:
                self.selected_features_ = None

            self.output_drift_detection_ = {'no_drift_features': features, 'history': history}
            self.step_end(output=self.output_drift_detection_)

            if _is_notebook:
                display(pd.DataFrame((('no drift features', features), ('history', history), ('drift score', scores)),
                                     columns=['key', 'value']), display_id='output_drift_detection')

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval


class PermutationImportanceSelectionStep(FeatureSelectStep):

    def __init__(self, experiment, name, scorer, estimator_size, importance_threshold):
        super().__init__(experiment, name)

        self.scorer = scorer
        self.estimator_size = estimator_size
        self.importance_threshold = importance_threshold

        # fixed
        self.unselected_features_ = None
        self.importances_ = None

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        if _is_notebook:
            display_markdown('### Evaluate feature importance', raw=True)

        self.step_start('evaluate feature importance')

        best_trials = hyper_model.get_top_trials(self.estimator_size)
        estimators = [hyper_model.load_estimator(trial.model_file) for trial in best_trials]
        self.step_progress('load estimators')

        if X_eval is None or y_eval is None:
            importances = feature_importance_batch(estimators, X_train, y_train, self.scorer, n_repeats=5)
        else:
            importances = feature_importance_batch(estimators, X_eval, y_eval, self.scorer, n_repeats=5)

        if _is_notebook:
            display_markdown('#### importances', raw=True)
            display(pd.DataFrame(
                zip(importances['columns'], importances['importances_mean'], importances['importances_std']),
                columns=['feature', 'importance', 'std']))
            display_markdown('#### feature selection', raw=True)

        feature_index = np.argwhere(importances.importances_mean < self.importance_threshold)
        selected_features = [feat for i, feat in enumerate(X_train.columns.to_list()) if i not in feature_index]
        unselected_features = list(set(X_train.columns.to_list()) - set(selected_features))
        self.step_progress('calc importance')

        if unselected_features:
            X_train = X_train[selected_features]
            if X_eval is not None:
                X_eval = X_eval[selected_features]
            if X_test is not None:
                X_test = X_test[selected_features]

        output_feature_importances_ = {
            'importances': importances,
            'selected_features': selected_features,
            'unselected_features': unselected_features}
        self.step_progress('drop features')
        self.step_end(output=output_feature_importances_)

        if _is_notebook:
            display(pd.DataFrame([('Selected', selected_features), ('Unselected', unselected_features)],
                                 columns=['key', 'value']))

        self.selected_features_ = selected_features if len(unselected_features) > 0 else None
        self.unselected_features_ = unselected_features
        self.importances_ = importances

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval


class SpaceSearchStep(ExperimentStep):
    def __init__(self, experiment, name, cv=False, num_folds=3):
        super().__init__(experiment, name)

        self.cv = cv
        self.num_folds = num_folds

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        if _is_notebook:
            display_markdown('### Pipeline search', raw=True)

        self.step_start('first stage search')

        if not dex.is_dask_object(X_eval):
            kwargs['eval_set'] = (X_eval, y_eval)

        model = copy.deepcopy(self.experiment.hyper_model)  # copy from original hyper_model instance
        model.search(X_train, y_train, X_eval, y_eval, cv=self.cv, num_folds=self.num_folds, **kwargs)

        self.step_end(output={'best_reward': model.get_best_trial().reward})

        return model, X_train, y_train, X_test, X_eval, y_eval

    def transform(self, X, y=None, **kwargs):
        return X

    def is_transform_skipped(self):
        return True


class DaskSpaceSearchStep(SpaceSearchStep):

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        X_train, y_train, X_test, X_eval, y_eval = \
            [v.persist() if dex.is_dask_object(v) else v for v in (X_train, y_train, X_test, X_eval, y_eval)]

        return super().fit_transform(hyper_model, X_train, y_train, X_test, X_eval, y_eval, **kwargs)


class EstimatorBuilderStep(ExperimentStep):
    def __init__(self, experiment, name):
        super().__init__(experiment, name)

        # fitted
        self.estimator_ = None

    def transform(self, X, y=None, **kwargs):
        return X

    def is_transform_skipped(self):
        return True


class EnsembleStep(EstimatorBuilderStep):
    def __init__(self, experiment, name, scorer=None, ensemble_size=7):
        assert ensemble_size > 1
        super().__init__(experiment, name)

        self.scorer = scorer if scorer is not None else get_scorer('neg_log_loss')
        self.ensemble_size = ensemble_size

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        if _is_notebook:
            display_markdown('### Ensemble', raw=True)

        self.step_start('ensemble')

        best_trials = hyper_model.get_top_trials(self.ensemble_size)
        estimators = [hyper_model.load_estimator(trial.model_file) for trial in best_trials]
        ensemble = self.get_ensemble(estimators, X_train, y_train)

        if all(['oof' in trial.memo.keys() for trial in best_trials]):
            logger.info('ensemble with oofs')
            oofs = self.get_ensemble_predictions(best_trials, ensemble)
            assert oofs is not None
            ensemble.fit(None, y_train, oofs)
        else:
            ensemble.fit(X_eval, y_eval)

        self.estimator_ = ensemble
        self.step_end(output={'ensemble': ensemble})

        if _is_notebook:
            display(ensemble)

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def get_ensemble(self, estimators, X_train, y_train):
        return GreedyEnsemble(self.task, estimators, scoring=self.scorer, ensemble_size=self.ensemble_size)

    def get_ensemble_predictions(self, trials, ensemble):
        oofs = None
        for i, trial in enumerate(trials):
            if 'oof' in trial.memo.keys():
                oof = trial.memo['oof']
                if oofs is None:
                    if len(oof.shape) == 1:
                        oofs = np.zeros((oof.shape[0], len(trials)), dtype=np.float64)
                    else:
                        oofs = np.zeros((oof.shape[0], len(trials), oof.shape[-1]), dtype=np.float64)
                oofs[:, i] = oof

        return oofs


class DaskEnsembleStep(EnsembleStep):
    def get_ensemble(self, estimators, X_train, y_train):
        if dex.exist_dask_object(X_train, y_train):
            return DaskGreedyEnsemble(self.task, estimators, scoring=self.scorer,
                                      ensemble_size=self.ensemble_size,
                                      predict_kwargs={'use_cache': False})

        return super().get_ensemble(estimators, X_train, y_train)

    def get_ensemble_predictions(self, trials, ensemble):
        if isinstance(ensemble, DaskGreedyEnsemble):
            oofs = [trial.memo.get('oof') for trial in trials]
            return oofs if any([oof is not None for oof in oofs]) else None

        return super().get_ensemble_predictions(trials, ensemble)


class FinalTrainStep(EstimatorBuilderStep):
    def __init__(self, experiment, name, retrain_on_wholedata=False):
        super().__init__(experiment, name)

        self.retrain_on_wholedata = retrain_on_wholedata

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        if _is_notebook:
            display_markdown('### Load best estimator', raw=True)

        self.step_start('load estimator')
        if self.retrain_on_wholedata:
            if _is_notebook:
                display_markdown('#### retrain on whole data', raw=True)
            trial = hyper_model.get_best_trial()
            X_all = dex.concat_df([X_train, X_eval], axis=0)
            y_all = dex.concat_df([y_train, y_eval], axis=0)
            estimator = hyper_model.final_train(trial.space_sample, X_all, y_all, **kwargs)
        else:
            estimator = hyper_model.load_estimator(hyper_model.get_best_trial().model_file)

        self.estimator_ = estimator
        display(estimator)
        self.step_end()

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval


class PseudoLabelStep(ExperimentStep):
    def __init__(self, experiment, name, estimator_builder,
                 pseudo_labeling_proba_threshold=0.8, pseudo_labeling_resplit=False, random_state=None):
        super().__init__(experiment, name)
        assert hasattr(estimator_builder, 'estimator_')

        self.estimator_builder = estimator_builder
        self.pseudo_labeling_proba_threshold = pseudo_labeling_proba_threshold
        self.pseudo_labeling_resplit = pseudo_labeling_resplit
        self.random_state = random_state

    def transform(self, X, y=None, **kwargs):
        return X

    def is_transform_skipped(self):
        return True

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        # build estimator
        hyper_model, X_train, y_train, X_test, X_eval, y_eval = \
            self.estimator_builder.fit_transform(hyper_model, X_train, y_train, X_test=X_test,
                                                 X_eval=X_eval, y_eval=y_eval, **kwargs)
        estimator = self.estimator_builder.estimator_

        # start here
        if _is_notebook:
            display_markdown('### Pseudo_label', raw=True)

        self.step_start('pseudo_label')

        X_pseudo = None
        y_pseudo = None
        if self.task in ['binary', 'multiclass'] and X_test is not None:
            proba = estimator.predict_proba(X_test)
            if self.task == 'binary':
                proba = proba[:, 1]
                proba_threshold = self.pseudo_labeling_proba_threshold
                X_pseudo, y_pseudo = self.extract_pseudo_label(X_test, proba, proba_threshold, estimator.classes_)

                if _is_notebook:
                    display_markdown('### Pseudo label set', raw=True)
                    display(pd.DataFrame([(dex.compute(X_pseudo.shape)[0],
                                           dex.compute(y_pseudo.shape)[0],
                                           # len(positive),
                                           # len(negative),
                                           proba_threshold)],
                                         columns=['X_pseudo.shape',
                                                  'y_pseudo.shape',
                                                  # 'positive samples',
                                                  # 'negative samples',
                                                  'proba threshold']), display_id='output_presudo_labelings')
                try:
                    if _is_notebook:
                        import seaborn as sns
                        import matplotlib.pyplot as plt
                        # Draw Plot
                        plt.figure(figsize=(8, 4), dpi=80)
                        sns.kdeplot(proba, shade=True, color="g", label="Proba", alpha=.7, bw_adjust=0.01)
                        # Decoration
                        plt.title('Density Plot of Probability', fontsize=22)
                        plt.legend()
                        plt.show()
                    else:
                        print(proba)
                except:
                    print(proba)

        if X_pseudo is not None:
            X_train, y_train, X_eval, y_eval = \
                self.merge_pseudo_label(X_train, y_train, X_eval, y_eval, X_pseudo, y_pseudo)
            if _is_notebook:
                display_markdown('#### Pseudo labeled train set & eval set', raw=True)
                display(pd.DataFrame([(X_train.shape,
                                       y_train.shape,
                                       X_eval.shape if X_eval is not None else None,
                                       y_eval.shape if y_eval is not None else None,
                                       X_test.shape if X_test is not None else None)],
                                     columns=['X_train.shape',
                                              'y_train.shape',
                                              'X_eval.shape',
                                              'y_eval.shape',
                                              'X_test.shape']), display_id='output_cleaner_info2')

        self.step_end(output={'pseudo_label': 'done'})

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def extract_pseudo_label(self, X_test, proba, proba_threshold, classes):
        positive = np.argwhere(proba > proba_threshold).ravel()
        negative = np.argwhere(proba < 1 - proba_threshold).ravel()
        X_test_p1 = X_test.iloc[positive]
        X_test_p2 = X_test.iloc[negative]
        y_p1 = np.ones(positive.shape, dtype='int64')
        y_p2 = np.zeros(negative.shape, dtype='int64')
        X_pseudo = pd.concat([X_test_p1, X_test_p2], axis=0)
        y_pseudo = np.concatenate([y_p1, y_p2], axis=0)
        if classes is not None:
            y_pseudo = np.array(classes).take(y_pseudo, axis=0)

        return X_pseudo, y_pseudo

    def merge_pseudo_label(self, X_train, y_train, X_eval, y_eval, X_pseudo, y_pseudo, **kwargs):
        if self.pseudo_labeling_resplit:
            x_list = [X_train, X_pseudo]
            y_list = [y_train, pd.Series(y_pseudo)]
            if X_eval is not None and y_eval is not None:
                x_list.append(X_eval)
                y_list.append(y_eval)
            X_mix = pd.concat(x_list, axis=0, ignore_index=True)
            y_mix = pd.concat(y_list, axis=0, ignore_index=True)
            if y_mix.dtype != y_train.dtype:
                y_mix = y_mix.astype(y_train.dtype)
            if self.task == 'regression':
                stratify = None
            else:
                stratify = y_mix

            eval_size = kwargs.get('eval_size', DEFAULT_EVAL_SIZE)
            X_train, X_eval, y_train, y_eval = \
                train_test_split(X_mix, y_mix, test_size=eval_size,
                                 random_state=self.random_state, stratify=stratify)
        else:
            X_train = pd.concat([X_train, X_pseudo], axis=0)
            y_train = pd.concat([y_train, pd.Series(y_pseudo)], axis=0)

        return X_train, y_train, X_eval, y_eval


class DaskPseudoLabelStep(PseudoLabelStep):
    def extract_pseudo_label(self, X_test, proba, proba_threshold, classes):
        if not dex.exist_dask_object(X_test, proba):
            return super().extract_pseudo_label(X_test, proba, proba_threshold, classes)

        da = dex.da
        positive = da.argwhere(proba > proba_threshold)
        positive = dex.make_divisions_known(positive).ravel()

        negative = da.argwhere(proba < 1 - proba_threshold)
        negative = dex.make_divisions_known(negative).ravel()

        X_test_values = X_test.to_dask_array(lengths=True)
        X_test_p1 = dex.array_to_df(X_test_values[positive], meta=X_test)
        X_test_p2 = dex.array_to_df(X_test_values[negative], meta=X_test)

        y_p1 = da.ones_like(positive, dtype='int64')
        y_p2 = da.zeros_like(negative, dtype='int64')

        X_pseudo = dex.concat_df([X_test_p1, X_test_p2], axis=0)
        y_pseudo = dex.hstack_array([y_p1, y_p2])

        if classes is not None:
            y_pseudo = da.take(np.array(classes), y_pseudo, axis=0)

        return X_pseudo, y_pseudo

    def merge_pseudo_label(self, X_train, y_train, X_eval, y_eval, X_pseudo, y_pseudo, **kwargs):
        if not dex.exist_dask_object(X_train, y_train, X_eval, y_eval, X_pseudo, y_pseudo):
            return super().merge_pseudo_label(X_train, y_train, X_eval, y_eval, X_pseudo, y_pseudo, **kwargs)

        if self.pseudo_labeling_resplit:
            x_list = [X_train, X_pseudo]
            y_list = [y_train, y_pseudo]
            if X_eval is not None and y_eval is not None:
                x_list.append(X_eval)
                y_list.append(y_eval)
            X_mix = dex.concat_df(x_list, axis=0)
            y_mix = dex.concat_df(y_list, axis=0)
            # if self.task == 'regression':
            #     stratify = None
            # else:
            #     stratify = y_mix

            X_mix = dex.concat_df([X_mix, y_mix], axis=1).reset_index(drop=True)
            y_mix = X_mix.pop(y_mix.name)

            eval_size = kwargs.get('eval_size', DEFAULT_EVAL_SIZE)
            X_train, X_eval, y_train, y_eval = \
                dex.train_test_split(X_mix, y_mix, test_size=eval_size, random_state=self.random_state)
        else:
            X_train = dex.concat_df([X_train, X_pseudo], axis=0)
            y_train = dex.concat_df([y_train, y_pseudo], axis=0)

            # align divisions
            X_train = dex.concat_df([X_train, y_train], axis=1)
            y_train = X_train.pop(y_train.name)

        return X_train, y_train, X_eval, y_eval


class SteppedExperiment(Experiment):
    def __init__(self, steps, *args, **kwargs):
        assert isinstance(steps, (tuple, list)) and all([isinstance(step, ExperimentStep) for step in steps])
        super(SteppedExperiment, self).__init__(*args, **kwargs)

        if logger.is_info_enabled():
            names = [step.name for step in steps]
            logger.info(f'create experiment with {names}')
        self.steps = steps

    def train(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, **kwargs):
        for step in self.steps:
            if X_test is not None and X_train.columns.to_list() != X_test.columns.to_list():
                logger.warning(f'X_train{X_train.columns.to_list()} and X_test{X_test.columns.to_list()}'
                               f' have different columns before {step.name}, try fix it.')
                X_test = X_test[X_train.columns]
            if X_eval is not None and X_eval.columns.to_list() != X_eval.columns.to_list():
                logger.warning(f'X_train{X_train.columns.to_list()} and X_eval{X_eval.columns.to_list()}'
                               f' have different columns before {step.name}, try fix it.')
                X_eval = X_eval[X_train.columns]

            X_train, y_train, X_test, X_eval, y_eval = \
                [v.persist() if dex.is_dask_object(v) else v for v in (X_train, y_train, X_test, X_eval, y_eval)]

            logger.info(f'fit_transform {step.name}')
            hyper_model, X_train, y_train, X_test, X_eval, y_eval = \
                step.fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval, **kwargs)

        estimator = self.to_estimator(self.steps)
        self.hyper_model = hyper_model

        return estimator

    @staticmethod
    def to_estimator(steps):
        last_step = steps[-1]
        assert hasattr(last_step, 'estimator_')

        pipeline_steps = [(step.name, step) for step in steps if not step.is_transform_skipped()]

        if len(pipeline_steps) > 0:
            pipeline_steps += [('estimator', last_step.estimator_)]
            estimator = Pipeline(pipeline_steps)
            if logger.is_info_enabled():
                names = [step[0] for step in pipeline_steps]
                logger.info(f'trained experiment pipeline: {names}')
        else:
            estimator = last_step.estimator_
            if logger.is_info_enabled():
                logger.info(f'trained experiment estimator:\n{estimator}')

        return estimator


class CompeteExperiment(SteppedExperiment):
    """
    A powerful experiment strategy for AutoML with a set of advanced features.

    There are still many challenges in the machine learning modeling process for tabular data, such as imbalanced data,
    data drift, poor generalization ability, etc.  This challenges cannot be completely solved by pipeline search,
    so we introduced in HyperGBM a more powerful tool is `CompeteExperiment`. `CompteExperiment` is composed of a series
    of steps and *Pipeline Search* is just one step. It also includes advanced steps such as data cleaning,
    data drift handling, two-stage search, ensemble etc.
    """

    def __init__(self, hyper_model, X_train, y_train, X_eval=None, y_eval=None, X_test=None,
                 eval_size=DEFAULT_EVAL_SIZE,
                 train_test_split_strategy=None,
                 cv=True, num_folds=3,
                 task=None,
                 callbacks=None,
                 random_state=9527,
                 scorer=None,
                 data_cleaner_args=None,
                 collinearity_detection=False,
                 drift_detection=True,
                 ensemble_size=20,
                 feature_reselection=False,
                 feature_reselection_estimator_size=10,
                 feature_reselection_threshold=1e-5,
                 pseudo_labeling=False,
                 pseudo_labeling_proba_threshold=0.8,
                 pseudo_labeling_resplit=False,
                 retrain_on_wholedata=False,
                 log_level=None,
                 **kwargs):
        """
        Parameters
        ----------
        hyper_model : hypergbm.HyperGBM
            A `HyperGBM` instance
        X_train : Pandas or Dask DataFrame
            Feature data for training
        y_train : Pandas or Dask Series
            Target values for training
        X_eval : (Pandas or Dask DataFrame) or None
            (default=None), Feature data for evaluation
        y_eval : (Pandas or Dask Series) or None, (default=None)
            Target values for evaluation
        X_test : (Pandas or Dask Series) or None, (default=None)
            Unseen data without target values for semi-supervised learning
        eval_size : float or int, (default=None)
            Only valid when ``X_eval`` or ``y_eval`` is None. If float, should be between 0.0 and 1.0 and represent
            the proportion of the dataset to include in the eval split. If int, represents the absolute number of
            test samples. If None, the value is set to the complement of the train size.
        train_test_split_strategy : *'adversarial_validation'* or None, (default=None)
            Only valid when ``X_eval`` or ``y_eval`` is None. If None, use eval_size to split the dataset,
            otherwise use adversarial validation approach.
        cv : bool, (default=True)
            If True, use cross-validation instead of evaluation set reward to guide the search process
        num_folds : int, (default=3)
            Number of cross-validated folds, only valid when cv is true
        task : str or None, (default=None)
            Task type(*binary*, *multiclass* or *regression*).
            If None, inference the type of task automatically
        callbacks : list of callback functions or None, (default=None)
            List of callback functions that are applied at each experiment step. See `hypernets.experiment.ExperimentCallback`
            for more information.
        random_state : int or RandomState instance, (default=9527)
            Controls the shuffling applied to the data before applying the split
        scorer : str, callable or None, (default=None)
            Scorer to used for feature importance evaluation and ensemble. It can be a single string
            (see [get_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html))
            or a callable (see [make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)).
            If None, exception will occur.
        data_cleaner_args : dict, (default=None)
            dictionary of parameters to initialize the `DataCleaner` instance. If None, `DataCleaner` will initialized with
            default values.
        collinearity_detection :  bool, (default=False)
            Whether to clear multicollinearity features
        drift_detection : bool,(default=True)
            Whether to enable data drift detection and processing. Only valid when *X_test* is provided. Concept drift in
            the input data is one of the main challenges. Over time, it will worsen the performance of model on new data.
            We introduce an adversarial validation approach to concept drift problems in HyperGBM. This approach will detect
            concept drift and identify the drifted features and process them automatically.
        feature_reselection : bool, (default=True)
            Whether to enable two stage feature selection and searching
        feature_reselection_estimator_size : int, (default=10)
            The number of estimator to evaluate feature importance. Only valid when *feature_reselection* is True.
        feature_reselection_threshold : float, (default=1e-5)
            The threshold for feature selection. Features with importance below the threshold will be dropped.  Only valid when *feature_reselection* is True.
        ensemble_size : int, (default=20)
            The number of estimator to ensemble. During the AutoML process, a lot of models will be generated with different
            preprocessing pipelines, different models, and different hyperparameters. Usually selecting some of the models
            that perform well to ensemble can obtain better generalization ability than just selecting the single best model.
        pseudo_labeling : bool, (default=False)
            Whether to enable pseudo labeling. Pseudo labeling is a semi-supervised learning technique, instead of manually
            labeling the unlabelled data, we give approximate labels on the basis of the labelled data. Pseudo-labeling can
            sometimes improve the generalization capabilities of the model.
        pseudo_labeling_proba_threshold : float, (default=0.8)
            Confidence threshold of pseudo-label samples. Only valid when *pseudo_labeling* is True.
        pseudo_labeling_resplit : bool, (default=False)
            Whether to re-split the training set and evaluation set after adding pseudo-labeled data. If False, the
            pseudo-labeled data is only appended to the training set. Only valid when *pseudo_labeling* is True.
        retrain_on_wholedata : bool, (default=False)
            Whether to retrain the model with whole data after the search is completed.
        log_level : int, str, or None (default=None),
            Level of logging, possible values:
                -logging.CRITICAL
                -logging.FATAL
                -logging.ERROR
                -logging.WARNING
                -logging.WARN
                -logging.INFO
                -logging.DEBUG
                -logging.NOTSET
        kwargs :

        """
        steps = []
        two_stage = False
        enable_dask = dex.exist_dask_object(X_train, y_train, X_test, X_eval, y_eval)

        if enable_dask:
            search_cls, ensemble_cls, pseudo_cls = SpaceSearchStep, DaskEnsembleStep, DaskPseudoLabelStep
        else:
            search_cls, ensemble_cls, pseudo_cls = SpaceSearchStep, EnsembleStep, PseudoLabelStep

        # data clean
        steps.append(DataCleanStep(self, 'data_clean',
                                   data_cleaner_args=data_cleaner_args, cv=cv,
                                   train_test_split_strategy=train_test_split_strategy,
                                   random_state=random_state))

        # select by collinearity
        if collinearity_detection:
            steps.append(MulticollinearityDetectStep(self, 'collinearity_detection',
                                                     drop_feature_with_collinearity=collinearity_detection))
        # drift detection
        if drift_detection:
            steps.append(DriftDetectStep(self, 'drift_detection', drift_detection=drift_detection))

        # first-stage search
        steps.append(search_cls(self, 'space_search', cv=cv, num_folds=num_folds))

        # pseudo label
        if pseudo_labeling and task != 'regression':
            if ensemble_size is not None and ensemble_size > 1:
                estimator_builder = ensemble_cls(self, 'pseudo_ensemble', scorer=scorer, ensemble_size=ensemble_size)
            else:
                estimator_builder = FinalTrainStep(self, 'pseudo_train', retrain_on_wholedata=retrain_on_wholedata)
            step = pseudo_cls(self, 'pseudo_labeling',
                              estimator_builder=estimator_builder,
                              pseudo_labeling_resplit=pseudo_labeling_resplit,
                              pseudo_labeling_proba_threshold=pseudo_labeling_proba_threshold,
                              random_state=random_state)
            steps.append(step)
            two_stage = True

        # importance selection
        if feature_reselection:
            step = PermutationImportanceSelectionStep(self, 'feature_reselection',
                                                      scorer=scorer,
                                                      estimator_size=feature_reselection_estimator_size,
                                                      importance_threshold=feature_reselection_threshold)
            steps.append(step)
            two_stage = True

        # two-stage search
        if two_stage:
            steps.append(search_cls(self, 'two_stage_search', cv=cv, num_folds=num_folds))

        # final train
        if ensemble_size is not None and ensemble_size > 1:
            last_step = ensemble_cls(self, 'final_ensemble', scorer=scorer, ensemble_size=ensemble_size)
        else:
            last_step = FinalTrainStep(self, 'final_train', retrain_on_wholedata=retrain_on_wholedata)
        steps.append(last_step)

        # ignore warnings
        import warnings
        warnings.filterwarnings('ignore')

        if log_level is not None:
            _set_log_level(log_level)

        self.run_kwargs = kwargs
        super(CompeteExperiment, self).__init__(steps,
                                                hyper_model, X_train, y_train, X_eval=X_eval, y_eval=y_eval,
                                                X_test=X_test, eval_size=eval_size, task=task,
                                                callbacks=callbacks,
                                                random_state=random_state)

    def train(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, **kwargs):
        if _is_notebook:
            display_markdown('### Input Data', raw=True)

            if dex.exist_dask_object(X_train, y_train, X_test, X_eval, y_eval):
                display_data = (dex.compute(X_train.shape)[0],
                                dex.compute(y_train.shape)[0],
                                dex.compute(X_eval.shape)[0] if X_eval is not None else None,
                                dex.compute(y_eval.shape)[0] if y_eval is not None else None,
                                dex.compute(X_test.shape)[0] if X_test is not None else None,
                                self.task if self.task == 'regression'
                                else f'{self.task}({dex.compute(y_train.nunique())[0]})')
            else:
                display_data = (X_train.shape,
                                y_train.shape,
                                X_eval.shape if X_eval is not None else None,
                                y_eval.shape if y_eval is not None else None,
                                X_test.shape if X_test is not None else None,
                                self.task if self.task == 'regression'
                                else f'{self.task}({y_train.nunique()})')
            display(pd.DataFrame([display_data],
                                 columns=['X_train.shape',
                                          'y_train.shape',
                                          'X_eval.shape',
                                          'y_eval.shape',
                                          'X_test.shape',
                                          'Task', ]), display_id='output_intput')

            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y = le.fit_transform(y_train.dropna())
            # Draw Plot
            plt.figure(figsize=(8, 4), dpi=80)
            sns.distplot(y, kde=False, color="g", label="y")
            # Decoration
            plt.title('Distribution of y', fontsize=12)
            plt.legend()
            plt.show()

        return super().train(hyper_model, X_train, y_train, X_test, X_eval, y_eval, **kwargs)

    def run(self, **kwargs):
        run_kwargs = {**self.run_kwargs, **kwargs}
        return super().run(**run_kwargs)


def make_experiment(train_data,
                    target=None,
                    eval_data=None,
                    test_data=None,
                    task=None,
                    searcher=None,
                    search_space=None,
                    search_callbacks=None,
                    early_stopping_rounds=10,
                    early_stopping_time_limit=3600,
                    early_stopping_reward=None,
                    reward_metric='accuracy',
                    optimize_direction=None,
                    use_cache=None,
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
    reward_metric : str, callable, optional
        Hypernets search reward metric name or callable, default is 'accuracy'. Possible values:
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
    use_cache : bool, optional
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
    >>> from hypernets.utils import logging
    >>>
    >>> experiment = make_experiment('/opt/data01/test.csv', target='y', log_level=logging.INFO)
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
    _set_log_level(log_level)

    def find_target(df):
        columns = df.columns.to_list()
        for col in columns:
            if col.lower() in DEFAULT_TARGET_SET:
                return col
        raise ValueError(f'Not found one of {DEFAULT_TARGET_SET} from your data, implicit target must be specified.')

    def metric_to_scoring(metric):
        mapping = {
            'auc': 'roc_auc_ovo',
            'accuracy': 'accuracy',
            'recall': 'recall',
            'precision': 'precision',
            'f1': 'f1',
            'mse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'msle': 'neg_mean_squared_log_error',
            'rmse': 'neg_root_mean_squared_error',
            'rootmeansquarederror': 'neg_root_mean_squared_error',
            'r2': 'r2',
            'logloss': 'neg_log_loss',
        }
        if metric not in mapping.keys():
            raise ValueError(f'Not found matching scorer for {metric}, implicit scorer must be specified.')

        return mapping[metric]

    def default_search_space():
        args = {}
        if early_stopping_rounds is not None:
            args['early_stopping_rounds'] = early_stopping_rounds

        for key in ('n_esitimators', 'class_balancing'):
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
        from hypernets.searchers import EvolutionSearcher, RandomSearcher, MCTSSearcher

        search_space_fn = search_space if search_space is not None \
            else default_search_space()
        op = optimize_direction if optimize_direction is not None \
            else 'max' if scorer._sign > 0 else 'min'

        if cls == EvolutionSearcher:
            s = cls(search_space_fn, optimize_direction=op,
                    population_size=30, sample_size=10, candidates_size=10,
                    regularized=True, use_meta_learner=True)
        elif cls == MCTSSearcher:
            s = MCTSSearcher(search_space_fn, optimize_direction=op, max_node_space=10)
        elif cls == RandomSearcher:
            s = cls(search_space_fn, optimize_direction=op)
        else:
            s = cls(search_space_fn, optimize_direction=op)

        return s

    def to_search_object(sch):
        from hypernets.core.searcher import Searcher as SearcherSpec
        from hypernets.searchers import EvolutionSearcher, RandomSearcher, MCTSSearcher

        if sch is None:
            sch = default_searcher(EvolutionSearcher)
        elif isinstance(sch, type):
            sch = default_searcher(sch)
        elif isinstance(sch, str):
            name2cls = {'evolution': EvolutionSearcher,
                        'mcts': MCTSSearcher,
                        'random': RandomSearcher
                        }
            if sch.lower() not in name2cls.keys():
                raise ValueError(f'Unrecognized searcher "{sch}".')
            sch = default_searcher(name2cls[sch.lower()])
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

    scorer = metric_to_scoring(reward_metric) if kwargs.get('scorer') is None else kwargs.get('scorer')
    if isinstance(scorer, str):
        scorer = get_scorer(scorer)

    searcher = to_search_object(searcher)

    if search_callbacks is None:
        search_callbacks = default_search_callbacks()
    search_callbacks = append_early_stopping_callbacks(search_callbacks)

    from hypergbm.hyper_gbm import HyperGBM
    hm = HyperGBM(searcher, reward_metric=reward_metric, callbacks=search_callbacks,
                  cache_dir=kwargs.pop('cache_dir', 'hypergbm_cache'),
                  clear_cache=kwargs.pop('clear_cache', True))

    use_cache = not dex.exist_dask_object(X_train, X_test, X_eval) if use_cache is None else bool(use_cache)

    experiment = CompeteExperiment(hm, X_train, y_train, X_eval=X_eval, y_eval=y_eval, X_test=X_test,
                                   task=task, scorer=scorer, use_cache=use_cache,
                                   **kwargs)

    if logger.is_info_enabled():
        train_shape, test_shape, eval_shape = \
            dex.compute(X_train.shape,
                        X_eval.shape if X_eval is not None else None,
                        X_test.shape if X_test is not None else None,
                        traverse=True)
        logger.info(f'make_experiment with train data:{train_shape}, '
                    f'test data:{test_shape}, eval data:{eval_shape}, target:{target}')

    return experiment
