# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import numpy as np
from hypernets.experiment import Experiment
from sklearn.model_selection import train_test_split
from tabular_toolbox.data_cleaner import DataCleaner
from tabular_toolbox import drift_detection as dd
from tabular_toolbox.feature_selection import select_by_multicollinearity
from .feature_importance import feature_importance_batch
from sklearn.metrics import get_scorer
from tabular_toolbox.ensemble import GreedyEnsemble


class CompeteExperiment(Experiment):
    def __init__(self, task, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3,
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
                                                eval_size=eval_size,
                                                random_state=random_state)
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

        self.output_drift_detection_ = None
        self.output_multi_collinearity_ = None
        self.output_feature_importances_ = None

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
        max_trails = kwargs.get('max_trails')
        if max_trails is None:
            max_trails = 10

        # 1. Clean Data
        self.data_cleaner = DataCleaner(**self.data_cleaner_args)
        X_train, y_train = self.data_cleaner.fit_transform(X_train, y_train)

        if X_eval or y_eval is None:
            X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=eval_size,
                                                                random_state=self.random_state, stratify=y_train)
        else:
            X_eval, y_eval = self.data_cleaner.transform(X_eval, y_eval)

        if X_test is not None:
            X_test, _ = self.data_cleaner.transform(X_test)

        # 2. Drop feature with multicollinearity
        if self.drop_feature_with_collinearity:
            corr_linkage, selected, unselected = select_by_multicollinearity(X_train)
            self.output_multi_collinearity_ = (corr_linkage, selected, unselected)
            X_train = X_train[selected]
            X_eval = X_eval[selected]
            X_test = X_test[selected]

        # 3. Shift detection
        if self.drift_detection and self.X_test is not None:
            features, history = dd.feature_selection(X_train, X_test)
            self.output_drift_detection_ = (features, history)
            X_train = X_train[features]
            X_eval = X_eval[features]
            X_test = X_test[features]

        # 4. Baseline search
        hyper_model.search(X_train, y_train, X_eval, y_eval, max_trails=max_trails, **kwargs)

        if self.mode == 'two-stage':
            # 5. Feature importance evaluation
            best_trials = hyper_model.get_top_trails(self.n_est_feature_importance)
            estimators = []
            for trail in best_trials:
                estimators.append(hyper_model.load_estimator(trail.model_file))

            importances = feature_importance_batch(estimators, X_eval, y_eval, self.scorer,
                                                   n_repeats=5)
            feature_index = np.argwhere(importances.importances_mean < self.importance_threshold)
            selected_features = [feat for i, feat in enumerate(X_train.columns.to_list()) if i not in feature_index]
            unselected_features = list(set(X_train.columns.to_list()) - set(selected_features))
            self.output_feature_importances_ = (importances, selected_features, unselected_features)
            X_train = X_train[selected_features]
            X_eval = X_eval[selected_features]
            X_test = X_test[selected_features]

            # 6. Final search
            hyper_model.search(X_train, y_train, X_eval, y_eval, max_trails=max_trails, **kwargs)

        # 7. Ensemble
        if self.ensemble_size > 1:
            best_trials = hyper_model.get_top_trails(self.ensemble_size)
            estimators = []
            for trail in best_trials:
                estimators.append(hyper_model.load_estimator(trail.model_file))
            ensemble = GreedyEnsemble(self.task, estimators, scoring=self.scorer, ensemble_size=self.ensemble_size)
            ensemble.fit(X_eval, y_eval)
            self.estimator = ensemble
        else:
            self.estimator = hyper_model.load_estimator(hyper_model.get_best_trail().model_file)

        # 8. Compose pipeline

        # 9. Save model
        return self.estimator
