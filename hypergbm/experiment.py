# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import copy

import numpy as np
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from hypernets.experiment import Experiment
from tabular_toolbox import drift_detection as dd
from tabular_toolbox.data_cleaner import DataCleaner
from tabular_toolbox.ensemble import GreedyEnsemble
from tabular_toolbox.feature_selection import select_by_multicollinearity
from .feature_importance import feature_importance_batch


class CompeteExperiment(Experiment):
    def __init__(self, task, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3,
                 callbacks=None,
                 random_state=9527,
                 scorer=None,
                 data_cleaner_args=None,
                 drop_feature_with_collinearity=False,
                 drift_detection=True,
                 mode='two-stage',
                 n_est_feature_importance=10,
                 importance_threshold=1e-5,
                 ensemble_size=7, ):
        super(CompeteExperiment, self).__init__(hyper_model, X_train, y_train, X_test, X_eval=X_eval, y_eval=y_eval,
                                                eval_size=eval_size, callbacks=callbacks, random_state=random_state)
        self.task = task
        self.data_cleaner_args = data_cleaner_args if data_cleaner_args is not None else {}
        self.drop_feature_with_collinearity = drop_feature_with_collinearity
        self.drift_detection = drift_detection
        self.mode = mode
        self.n_est_feature_importance = n_est_feature_importance
        if scorer is None:
            self.scorer = get_scorer('neg_log_loss')
        else:
            self.scorer = scorer
        self.importance_threshold = importance_threshold
        self.ensemble_size = ensemble_size
        self.selected_features_ = None
        self.output_drift_detection_ = None
        self.output_multi_collinearity_ = None
        self.output_feature_importances_ = None
        self.first_hyper_model = None
        self.second_hyper_model = None

    def data_split(self, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3):

        # support split data by model which trained to estimate whether a sample in train set is
        # similar with samples in test set.

        if X_eval or y_eval is None:
            X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=eval_size,
                                                                random_state=self.random_state, stratify=y_train)
        return X_train, y_train, X_test, X_eval, y_eval

    def train(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3, **kwargs):
        """Run an experiment

        Arguments
        ---------
        max_trails :

        """
        self.step_start('clean and split data')
        # 1. Clean Data
        self.data_cleaner = DataCleaner(**self.data_cleaner_args)

        X_train, y_train = self.data_cleaner.fit_transform(X_train, y_train)
        self.step_progress('fit_transform train set')

        if X_eval or y_eval is None:

            X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=eval_size,
                                                                random_state=self.random_state, stratify=y_train)
            self.step_progress('split into train set and eval set')
        else:
            X_eval, y_eval = self.data_cleaner.transform(X_eval, y_eval)
            self.step_progress('transform eval set')

        if X_test is not None:
            X_test, _ = self.data_cleaner.transform(X_test)
            self.step_progress('transform X_test')

        self.step_end()

        original_features = X_train.columns.to_list()

        # 2. Drop features with multicollinearity
        if self.drop_feature_with_collinearity:
            self.step_start('drop features with multicollinearity')
            corr_linkage, selected, unselected = select_by_multicollinearity(X_train)
            self.output_multi_collinearity_ = {
                'corr_linkage': corr_linkage,
                'selected': selected,
                'unselected': unselected
            }
            self.step_progress('calc correlation')

            self.selected_features_ = selected
            X_train = X_train[self.selected_features_]
            X_eval = X_eval[self.selected_features_]
            X_test = X_test[self.selected_features_]
            self.step_progress('drop features')
            self.step_end(output=self.output_multi_collinearity_)

        # 3. Shift detection
        if self.drift_detection and self.X_test is not None:
            self.step_start('detect drifting')
            features, history = dd.feature_selection(X_train, X_test)
            self.output_drift_detection_ = {'no_drift_features': features, 'history': history}
            self.selected_features_ = features
            X_train = X_train[self.selected_features_]
            X_eval = X_eval[self.selected_features_]
            X_test = X_test[self.selected_features_]
            self.step_end(output=self.output_drift_detection_)

        # 4. Baseline search
        self.first_hyper_model = copy.deepcopy(hyper_model)
        self.step_start('first stage search')
        self.first_hyper_model.search(X_train, y_train, X_eval, y_eval, **kwargs)
        self.hyper_model = self.first_hyper_model
        self.step_end(output={'best_reward':self.hyper_model.get_best_trail().reward})

        if self.mode == 'two-stage':
            # 5. Feature importance evaluation
            self.step_start('evaluate feature importance')
            best_trials = self.hyper_model.get_top_trails(self.n_est_feature_importance)
            estimators = []
            for trail in best_trials:
                estimators.append(self.hyper_model.load_estimator(trail.model_file))
            self.step_progress('load estimators')

            importances = feature_importance_batch(estimators, X_eval, y_eval, self.scorer,
                                                   n_repeats=5)
            feature_index = np.argwhere(importances.importances_mean < self.importance_threshold)
            selected_features = [feat for i, feat in enumerate(X_train.columns.to_list()) if i not in feature_index]
            unselected_features = list(set(X_train.columns.to_list()) - set(selected_features))
            self.output_feature_importances_ = {
                'importances': importances,
                'selected_features': selected_features,
                'unselected_features': unselected_features}
            self.selected_features_ = selected_features
            self.step_progress('calc importance')

            X_train = X_train[self.selected_features_]
            X_eval = X_eval[self.selected_features_]
            X_test = X_test[self.selected_features_]
            self.step_progress('drop features')
            self.step_end(output=self.output_feature_importances_)

            # 6. Final search
            self.step_start('two stage search')
            self.second_hyper_model = copy.deepcopy(hyper_model)
            self.second_hyper_model.search(X_train, y_train, X_eval, y_eval, **kwargs)
            self.hyper_model = self.second_hyper_model
            self.step_end(output={'best_reward': self.hyper_model.get_best_trail().reward})
        # 7. Ensemble
        if self.ensemble_size > 1:
            self.step_start('ensemble')
            best_trials = self.hyper_model.get_top_trails(self.ensemble_size)
            estimators = []
            for trail in best_trials:
                estimators.append(self.hyper_model.load_estimator(trail.model_file))
            ensemble = GreedyEnsemble(self.task, estimators, scoring=self.scorer, ensemble_size=self.ensemble_size)
            ensemble.fit(X_eval, y_eval)
            self.estimator = ensemble
            self.step_end(output={'ensemble': ensemble})
        else:
            self.step_start('load estimator')
            self.estimator = self.hyper_model.load_estimator(self.hyper_model.get_best_trail().model_file)
            self.step_end()

        droped_features = set(original_features) - set(self.selected_features_)
        self.data_cleaner.append_drop_columns(droped_features)
        # 8. Compose pipeline
        self.step_start('compose pipeline')
        pipeline = Pipeline([('data_cleaner', self.data_cleaner), ('estimator', self.estimator)])
        self.step_end()
        # 9. Save model
        return pipeline
