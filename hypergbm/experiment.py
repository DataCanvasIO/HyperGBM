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
from .feature_importance import feature_importance_batch

logger = logging.get_logger(__name__)

DEFAULT_EVAL_SIZE = 0.3
DEFAULT_TARGET_SET = {'y', 'target'}


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

        self.step_start = experiment.step_start
        self.step_end = experiment.step_end
        self.step_progress = experiment.step_progress

    @property
    def task(self):
        return self.experiment.task

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
                    assert set(y_train) == set(y_eval), \
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

        display_markdown('### Data Cleaner', raw=True)

        display(self.data_cleaner, display_id='output_cleaner_info1')
        display_markdown('### Train set & Eval set', raw=True)
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
        original_features = X_train.columns.to_list()

        self.selected_features_ = original_features
        self.data_cleaner = data_cleaner

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def transform(self, X, y=None, **kwargs):
        return self.data_cleaner.transform(X, y, **kwargs)


class SelectByMulticollinearityStep(FeatureSelectStep):

    def __init__(self, experiment, name, drop_feature_with_collinearity=True):
        super().__init__(experiment, name)

        self.drop_feature_with_collinearity = drop_feature_with_collinearity

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        if self.drop_feature_with_collinearity:
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
            # print(self.output_multi_collinearity_)
            display(
                pd.DataFrame([(k, v) for k, v in self.output_multi_collinearity_.items()], columns=['key', 'value']),
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
            display(pd.DataFrame((('no drift features', features), ('history', history), ('drift score', scores)),
                                 columns=['key', 'value']), display_id='output_drift_detection')
            self.step_end(output=self.output_drift_detection_)

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval


class PermutationImportanceSelectionStep(FeatureSelectStep):

    def __init__(self, experiment, name, scorer, n_est_feature_importance, importance_threshold):
        super().__init__(experiment, name)

        self.scorer = scorer
        self.n_est_feature_importance = n_est_feature_importance
        self.importance_threshold = importance_threshold

        # fixed
        self.unselected_features_ = None
        self.importances_ = None

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        self.step_start('evaluate feature importance')
        display_markdown('### Evaluate feature importance', raw=True)

        best_trials = hyper_model.get_top_trials(self.n_est_feature_importance)
        estimators = [hyper_model.load_estimator(trial.model_file) for trial in best_trials]
        self.step_progress('load estimators')

        if X_eval is None or y_eval is None:
            importances = feature_importance_batch(estimators, X_train, y_train, self.scorer, n_repeats=5)
        else:
            importances = feature_importance_batch(estimators, X_eval, y_eval, self.scorer, n_repeats=5)
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
        self.step_start('first stage search')
        display_markdown('### Pipeline search', raw=True)

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
        self.step_start('ensemble')
        display_markdown('### Ensemble', raw=True)

        best_trials = hyper_model.get_top_trials(self.ensemble_size)
        estimators = [hyper_model.load_estimator(trial.model_file) for trial in best_trials]
        ensemble = self.get_ensemble(estimators, X_train, y_train)

        if all(['oof' in trial.memo.keys() for trial in best_trials]):
            print('ensemble with oofs')
            oofs = self.get_ensemble_predictions(best_trials, ensemble)
            assert oofs is not None
            ensemble.fit(None, y_train, oofs)
        else:
            ensemble.fit(X_eval, y_eval)

        self.estimator_ = ensemble
        self.step_end(output={'ensemble': ensemble})
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
            return DaskGreedyEnsemble(self.task, estimators, scoring=self.scorer, ensemble_size=self.ensemble_size)

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
        display_markdown('### Load best estimator', raw=True)

        self.step_start('load estimator')
        if self.retrain_on_wholedata:
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
        self.step_start('pseudo_label')
        display_markdown('### Pseudo_label', raw=True)

        X_pseudo = None
        y_pseudo = None
        if self.task in ['binary', 'multiclass'] and X_test is not None:
            proba = estimator.predict_proba(X_test)
            if self.task == 'binary':
                proba = proba[:, 1]
                proba_threshold = self.pseudo_labeling_proba_threshold
                X_pseudo, y_pseudo = self.extract_pseudo_label(X_test, proba, proba_threshold, estimator.classes_)

                display_markdown('### Pseudo label set', raw=True)
                display(pd.DataFrame([(X_pseudo.shape,
                                       y_pseudo.shape,
                                       # len(positive),
                                       # len(negative),
                                       proba_threshold)],
                                     columns=['X_pseudo.shape',
                                              'y_pseudo.shape',
                                              # 'positive samples',
                                              # 'negative samples',
                                              'proba threshold']), display_id='output_presudo_labelings')
                try:
                    if isnotebook():
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
            y_pseudo = da.take(da.array(classes), y_pseudo, axis=0)

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
    def __init__(self, hyper_model, X_train, y_train, X_eval=None, y_eval=None, X_test=None,
                 eval_size=DEFAULT_EVAL_SIZE,
                 train_test_split_strategy=None,
                 cv=False, num_folds=3,
                 task=None,
                 callbacks=None,
                 random_state=9527,
                 scorer=None,
                 data_cleaner_args=None,
                 drop_feature_with_collinearity=False,
                 drift_detection=True,
                 two_stage_importance_selection=False,
                 n_est_feature_importance=10,
                 importance_threshold=1e-5,
                 ensemble_size=7,
                 pseudo_labeling=False,
                 pseudo_labeling_proba_threshold=0.8,
                 pseudo_labeling_resplit=False,
                 retrain_on_wholedata=False,
                 log_level=None,
                 **kwargs):
        steps = []
        two_stage = False
        enable_dask = dex.exist_dask_object(X_train, y_train, X_test, X_eval, y_eval)

        if enable_dask:
            search_cls, ensemble_cls, pseudo_cls = DaskSpaceSearchStep, DaskEnsembleStep, DaskPseudoLabelStep
        else:
            search_cls, ensemble_cls, pseudo_cls = SpaceSearchStep, EnsembleStep, PseudoLabelStep

        # data clean
        steps.append(DataCleanStep(self, 'data_clean',
                                   data_cleaner_args=data_cleaner_args, cv=cv,
                                   train_test_split_strategy=train_test_split_strategy,
                                   random_state=random_state))

        # select by collinearity
        if drop_feature_with_collinearity:
            steps.append(SelectByMulticollinearityStep(self, 'drop_feature_with_collinearity',
                                                       drop_feature_with_collinearity=drop_feature_with_collinearity))
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
        if two_stage_importance_selection:
            step = PermutationImportanceSelectionStep(self, 'two_stage_pi_select',
                                                      scorer=scorer,
                                                      n_est_feature_importance=n_est_feature_importance,
                                                      importance_threshold=importance_threshold)
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

        display_markdown('### Input Data', raw=True)
        display(pd.DataFrame([(X_train.shape,
                               y_train.shape,
                               X_eval.shape if X_eval is not None else None,
                               y_eval.shape if y_eval is not None else None,
                               X_test.shape if X_test is not None else None,
                               self.task if self.task == 'regression' else f'{self.task}({y_train.nunique()})')],
                             columns=['X_train.shape',
                                      'y_train.shape',
                                      'X_eval.shape',
                                      'y_eval.shape',
                                      'X_test.shape',
                                      'Task', ]), display_id='output_intput')

        if isnotebook():
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

