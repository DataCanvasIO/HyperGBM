# -*- coding:utf-8 -*-
__author__ = 'yangjian'
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


def _set_log_level(log_level):
    logging.set_level(log_level)

    from tabular_toolbox.utils import logging as tlogging
    tlogging.set_level(log_level)

    # import logging as pylogging
    # pylogging.basicConfig(level=log_level)


class ExperimentStep(object):
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

        return X_train, y_train, X_test, X_eval, y_eval

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

        return X_train, y_train, X_test, X_eval, y_eval


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

            self.output_drift_detection_ = {'no_drift_features': features, 'history': history}
            self.selected_features_ = features

            X_train = X_train[self.selected_features_]
            if X_eval is not None:
                X_eval = X_eval[self.selected_features_]
            if X_test is not None:
                X_test = X_test[self.selected_features_]
            self.step_end(output=self.output_drift_detection_)

            display(pd.DataFrame((('no drift features', features), ('history', history), ('drift score', scores)),
                                 columns=['key', 'value']), display_id='output_drift_detection')

        return X_train, y_train, X_test, X_eval, y_eval


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
        estimators = []
        for trial in best_trials:
            estimators.append(hyper_model.load_estimator(trial.model_file))
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

        self.selected_features_ = selected_features
        self.unselected_features_ = unselected_features
        self.importances_ = importances

        return X_train, y_train, X_test, X_eval, y_eval


class BaseSearchAndTrainStep(ExperimentStep):
    def __init__(self, experiment, name, scorer=None, cv=False, num_folds=3,
                 retrain_on_wholedata=False, ensemble_size=7):
        super().__init__(experiment, name)

        self.scorer = scorer if scorer is not None else get_scorer('neg_log_loss')
        self.cv = cv
        self.num_folds = num_folds
        self.retrain_on_wholedata = retrain_on_wholedata
        self.ensemble_size = ensemble_size

        # fitted
        self.estimator_ = None

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        X_train, y_train, X_test, X_eval, y_eval, searched_model = \
            self.search(hyper_model, X_train, y_train, X_test, X_eval, y_eval, **kwargs)

        estimator = self.final_train(searched_model, X_train, y_train, X_test, X_eval, y_eval, **kwargs)
        self.estimator_ = estimator

        return X_train, y_train, X_test, X_eval, y_eval

    def transform(self, X, y=None, **kwargs):
        return X

    def search(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        self.step_start('first stage search')
        display_markdown('### Pipeline search', raw=True)

        if not dex.is_dask_object(X_eval):
            kwargs['eval_set'] = (X_eval, y_eval)
        model = copy.deepcopy(hyper_model)
        model.search(X_train, y_train, X_eval, y_eval, cv=self.cv, num_folds=self.num_folds, **kwargs)

        self.step_end(output={'best_reward': model.get_best_trial().reward})

        return X_train, y_train, X_test, X_eval, y_eval, model

    def final_train(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        # 7. Ensemble
        if self.ensemble_size > 1:
            self.step_start('ensemble')
            display_markdown('### Ensemble', raw=True)

            best_trials = hyper_model.get_top_trials(self.ensemble_size)
            estimators = [hyper_model.load_estimator(trial.model_file) for trial in best_trials]
            ensemble = self.get_ensemble(estimators, X_train, y_train)
            if self.cv:
                print('fit on oofs')
                oofs = self.get_ensemble_predictions(best_trials, ensemble)
                assert oofs is not None
                ensemble.fit(None, y_train, oofs)
            else:
                ensemble.fit(X_eval, y_eval)

            estimator = ensemble
            self.step_end(output={'ensemble': ensemble})
            display(ensemble)
        else:
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
            self.step_end()

        return estimator

    def get_ensemble(self, estimators, X_train, y_train):
        return GreedyEnsemble(self.task, estimators, scoring=self.scorer, ensemble_size=self.ensemble_size)

    def get_ensemble_predictions(self, trials, ensemble):
        oofs = None
        for i, trial in enumerate(trials):
            if trial.memo.__contains__('oof'):
                oof = trial.memo['oof']
                if oofs is None:
                    if len(oof.shape) == 1:
                        oofs = np.zeros((oof.shape[0], len(trials)), dtype=np.float64)
                    else:
                        oofs = np.zeros((oof.shape[0], len(trials), oof.shape[-1]), dtype=np.float64)
                oofs[:, i] = oof

        return oofs


class TwoStageSearchAndTrainStep(BaseSearchAndTrainStep):
    def __init__(self, experiment, name, scorer=None, cv=False, num_folds=3, retrain_on_wholedata=False,
                 pseudo_labeling=False, pseudo_labeling_proba_threshold=0.8,
                 ensemble_size=7, pseudo_labeling_resplit=False,
                 two_stage_importance_selection=True, n_est_feature_importance=10, importance_threshold=1e-5,
                 random_state=None):
        super().__init__(experiment, name, scorer=scorer, cv=cv, num_folds=num_folds,
                         retrain_on_wholedata=retrain_on_wholedata, ensemble_size=ensemble_size)

        self.pseudo_labeling = pseudo_labeling
        self.pseudo_labeling_proba_threshold = pseudo_labeling_proba_threshold
        self.pseudo_labeling_resplit = pseudo_labeling_resplit
        self.random_state = random_state

        if two_stage_importance_selection:
            self.pi = PermutationImportanceSelectionStep(experiment, f'{name}_pi', scorer, n_est_feature_importance,
                                                         importance_threshold)
        else:
            self.pi = None

    def transform(self, X, y=None, **kwargs):
        if self.pi:
            X = self.pi.transform(X, y, **kwargs)

        return X

    def search(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        X_train, y_train, X_test, X_eval, y_eval, searched_model = \
            super().search(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval, **kwargs)

        X_pseudo, y_pseudo = \
            self.do_pseudo_label(searched_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval,
                                 **kwargs)

        if self.pi:
            X_train, y_train, X_test, X_eval, y_eval = \
                self.pi.fit_transform(searched_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval,
                                      **kwargs)
            selected_features, unselected_features = self.pi.selected_features_, self.pi.unselected_features_
        else:
            selected_features, unselected_features = X_train.columns.to_list(), []

        if len(unselected_features) > 0 or X_pseudo is not None:
            if self.pi and X_pseudo is not None:
                X_pseudo = X_pseudo[selected_features]
            X_train, y_train, X_test, X_eval, y_eval, searched_model = \
                self.do_two_stage_search(hyper_model, X_train, y_train, X_test, X_eval, y_eval,
                                         X_pseudo, y_pseudo, **kwargs)
        else:
            display_markdown('### Skip pipeline search stage 2', raw=True)

        return X_train, y_train, X_test, X_eval, y_eval, searched_model

    def do_pseudo_label(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        # 5. Pseudo Labeling
        X_pseudo = None
        y_pseudo = None
        if self.task in ['binary', 'multiclass'] and self.pseudo_labeling and X_test is not None:
            es = self.ensemble_size if self.ensemble_size > 0 else 10
            best_trials = hyper_model.get_top_trials(es)
            estimators = [hyper_model.load_estimator(trial.model_file) for trial in best_trials]
            ensemble = self.get_ensemble(estimators, X_train, y_train)
            oofs = self.get_ensemble_predictions(best_trials, ensemble)
            if oofs is not None:
                print('fit on oofs')
                ensemble.fit(None, y_train, oofs)
            else:
                ensemble.fit(X_eval, y_eval)
            proba = ensemble.predict_proba(X_test)[:, 1]
            if self.task == 'binary':
                proba_threshold = self.pseudo_labeling_proba_threshold
                X_pseudo, y_pseudo = self.extract_pseduo_label(X_test, proba, proba_threshold, ensemble.classes_)

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
        return X_pseudo, y_pseudo

    def extract_pseduo_label(self, X_test, proba, proba_threshold, classes):
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

    def do_two_stage_search(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None,
                            X_pseudo=None, y_pseudo=None, **kwargs):
        # 6. Final search
        self.step_start('two stage search')
        display_markdown('### Pipeline search stage 2', raw=True)

        # self.second_hyper_model = copy.deepcopy(hyper_model)
        second_hyper_model = copy.deepcopy(hyper_model)

        kwargs['eval_set'] = (X_eval, y_eval)
        if X_pseudo is not None:
            if self.pseudo_labeling_resplit:
                x_list = [X_train, X_pseudo]
                y_list = [y_train, pd.Series(y_pseudo)]
                if X_eval is not None and y_eval is not None:
                    x_list.append(X_eval)
                    y_list.append(y_eval)
                X_mix = pd.concat(x_list, axis=0)
                y_mix = pd.concat(y_list, axis=0)
                if self.task == 'regression':
                    stratify = None
                else:
                    stratify = y_mix

                eval_size = kwargs.get('eval_size', DEFAULT_EVAL_SIZE)
                X_train, X_eval, y_train, y_eval = train_test_split(X_mix, y_mix, test_size=eval_size,
                                                                    random_state=self.random_state,
                                                                    stratify=stratify)
            else:
                X_train = pd.concat([X_train, X_pseudo], axis=0)
                y_train = pd.concat([y_train, pd.Series(y_pseudo)], axis=0)

            display_markdown('#### Final train set & eval set', raw=True)
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

        second_hyper_model.search(X_train, y_train, X_eval, y_eval, cv=self.cv, num_folds=self.num_folds,
                                  **kwargs)

        self.step_end(output={'best_reward': second_hyper_model.get_best_trial().reward})

        return X_train, y_train, X_test, X_eval, y_eval, second_hyper_model


class DaskBaseSearchAndTrainStep(BaseSearchAndTrainStep):
    def get_ensemble(self, estimators, X_train, y_train):
        if dex.exist_dask_object(X_train, y_train):
            return DaskGreedyEnsemble(self.task, estimators, scoring=self.scorer, ensemble_size=self.ensemble_size)
        else:
            return super().get_ensemble(estimators, X_train, y_train)

    def get_ensemble_predictions(self, trials, ensemble):
        if isinstance(ensemble, DaskGreedyEnsemble):
            oofs = [trial.memo.get('oof') for trial in trials]
            if all([oof is None for oof in oofs]):
                oofs = None
            return oofs

        return super().get_ensemble_predictions(trials, ensemble)


class DaskTwoStageSearchAndTrainStep(TwoStageSearchAndTrainStep):
    def get_ensemble(self, estimators, X_train, y_train):
        if not dex.exist_dask_object(X_train, y_train):
            return super().get_ensemble(estimators, X_train, y_train)

        return DaskGreedyEnsemble(self.task, estimators, scoring=self.scorer, ensemble_size=self.ensemble_size)

    def get_ensemble_predictions(self, trials, ensemble):
        if not isinstance(ensemble, DaskGreedyEnsemble):
            return super().get_ensemble_predictions(trials, ensemble)

        oofs = [trial.memo.get('oof') for trial in trials]
        if all([oof is None for oof in oofs]):
            oofs = None
        return oofs

    def extract_pseduo_label(self, X_test, proba, proba_threshold, classes):
        if not dex.exist_dask_object(X_test, proba):
            return super().extract_pseduo_label(X_test, proba, proba_threshold, classes)

        da = dex.da
        dd = dex.dd
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

    def do_two_stage_search(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, X_pseudo=None,
                            y_pseudo=None, **kwargs):
        if not dex.exist_dask_object(X_train, y_train, X_test, X_eval, y_eval, X_pseudo, y_pseudo):
            return super().do_two_stage_search(hyper_model, X_train, y_train, X_test, X_eval, y_eval,
                                               X_pseudo, y_pseudo, **kwargs)

        self.step_start('two stage search')
        display_markdown('### Pipeline search stage 2', raw=True)

        # self.second_hyper_model = copy.deepcopy(hyper_model)
        second_hyper_model = copy.deepcopy(hyper_model)

        # kwargs['eval_set'] = (X_eval, y_eval)
        kwargs['eval_set'] = None
        if X_pseudo is not None:
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
                X_train, X_eval, y_train, y_eval = dex.train_test_split(X_mix, y_mix, test_size=eval_size,
                                                                        random_state=self.random_state)
                X_train, X_eval, y_train, y_eval = \
                    X_train.persist(), X_eval.persist(), y_train.persist(), y_eval.persist()
            else:
                X_train = dex.concat_df([X_train, X_pseudo], axis=0)
                y_train = dex.concat_df([y_train, y_pseudo], axis=0)

                X_train = dex.concat_df([X_train, y_train], axis=1)
                y_train = X_train.pop(y_train.name)

                X_train, y_train = X_train.persist(), y_train.persist()

            display_markdown('#### Final train set & eval set', raw=True)
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

        second_hyper_model.search(X_train, y_train, X_eval, y_eval, cv=self.cv, num_folds=self.num_folds,
                                  **kwargs)

        self.step_end(output={'best_reward': second_hyper_model.get_best_trial().reward})

        return X_train, y_train, X_test, X_eval, y_eval, second_hyper_model


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
                step.fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval, **kwargs)

        last_step = self.steps[-1]
        assert hasattr(last_step, 'estimator_')

        pipeline_steps = [(step.name, step) for step in self.steps]
        pipeline_steps += [('estimator', last_step.estimator_)]

        return Pipeline(pipeline_steps)


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
                 mode='one-stage',
                 two_stage_importance_selection=True,
                 n_est_feature_importance=10,
                 importance_threshold=1e-5,
                 ensemble_size=7,
                 pseudo_labeling=False,
                 pseudo_labeling_proba_threshold=0.8,
                 pseudo_labeling_resplit=False,
                 feature_generation=False,
                 retrain_on_wholedata=False,
                 enable_dask=False,
                 log_level=None):

        steps = [DataCleanStep(self, 'data_clean',
                               data_cleaner_args=data_cleaner_args, cv=cv,
                               train_test_split_strategy=train_test_split_strategy,
                               random_state=random_state),
                 ]
        if drop_feature_with_collinearity:
            steps.append(SelectByMulticollinearityStep(self, 'select_by_multicollinearity',
                                                       drop_feature_with_collinearity=drop_feature_with_collinearity))
        if drift_detection:
            steps.append(DriftDetectStep(self, 'drift_detected', drift_detection=drift_detection))

        if mode == 'two-stage':
            step_cls = TwoStageSearchAndTrainStep if not enable_dask else DaskTwoStageSearchAndTrainStep
            last_step = step_cls(self, 'two_stage_search_and_train',
                                 scorer=scorer, cv=cv, num_folds=num_folds,
                                 retrain_on_wholedata=retrain_on_wholedata,
                                 pseudo_labeling=pseudo_labeling,
                                 pseudo_labeling_proba_threshold=pseudo_labeling_proba_threshold,
                                 ensemble_size=ensemble_size,
                                 pseudo_labeling_resplit=pseudo_labeling_resplit,
                                 two_stage_importance_selection=two_stage_importance_selection,
                                 n_est_feature_importance=n_est_feature_importance,
                                 importance_threshold=importance_threshold)
        else:
            step_cls = BaseSearchAndTrainStep if not enable_dask else DaskBaseSearchAndTrainStep
            last_step = step_cls(self, 'base_search_and_train',
                                 scorer=scorer, cv=cv, num_folds=num_folds,
                                 ensemble_size=ensemble_size,
                                 retrain_on_wholedata=retrain_on_wholedata)
        steps.append(last_step)

        # ignore warnings
        import warnings
        warnings.filterwarnings('ignore')

        if log_level is not None:
            _set_log_level(log_level)

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
