# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from hypernets.experiment import Experiment
from tabular_toolbox import drift_detection as dd
from tabular_toolbox.data_cleaner import DataCleaner
from tabular_toolbox.ensemble import GreedyEnsemble
from tabular_toolbox.feature_selection import select_by_multicollinearity
from .feature_importance import feature_importance_batch
from IPython.display import display, clear_output, update_display, display_markdown


class CompeteExperiment(Experiment):
    def __init__(self, hyper_model, X_train, y_train, X_eval=None, y_eval=None, X_test=None, eval_size=0.3,
                 train_test_split_strategy=None,
                 task=None,
                 callbacks=None,
                 random_state=9527,
                 scorer=None,
                 data_cleaner_args=None,
                 drop_feature_with_collinearity=False,
                 drift_detection=True,
                 mode='one-stage',
                 n_est_feature_importance=10,
                 importance_threshold=1e-5,
                 ensemble_size=7,
                 feature_generation=False,
                 retrain_on_wholedata=False,
                 log_level=None):
        super(CompeteExperiment, self).__init__(hyper_model, X_train, y_train, X_eval=X_eval, y_eval=y_eval,
                                                X_test=X_test, eval_size=eval_size, task=task,
                                                callbacks=callbacks,
                                                random_state=random_state)
        self.data_cleaner_args = data_cleaner_args if data_cleaner_args is not None else {}
        self.drop_feature_with_collinearity = drop_feature_with_collinearity
        self.train_test_split_strategy = train_test_split_strategy
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
        self.feature_generation = feature_generation
        self.retrain_on_wholedata = retrain_on_wholedata

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
        # ignore warnings
        import warnings
        warnings.filterwarnings('ignore')

        display_markdown('### Input Data', raw=True)
        display(pd.DataFrame([(X_train.shape,
                               y_train.shape,
                               X_eval.shape if X_eval is not None else None,
                               y_eval.shape if y_eval is not None else None,
                               X_test.shape if X_test is not None else None)],
                             columns=['X_train.shape',
                                      'y_train.shape',
                                      'X_eval.shape',
                                      'y_eval.shape',
                                      'X_test.shape']), display_id='output_intput')

        self.step_start('clean and split data')
        # 1. Clean Data
        self.data_cleaner = DataCleaner(**self.data_cleaner_args)

        X_train, y_train = self.data_cleaner.fit_transform(X_train, y_train)
        self.step_progress('fit_transform train set')

        if X_test is not None:
            X_test = self.data_cleaner.transform(X_test)
            self.step_progress('transform X_test')

        if X_eval is None or y_eval is None:
            stratify = y_train
            if self.train_test_split_strategy == 'adversarial_validation' and X_test is not None:
                print('DriftDetector.train_test_split')
                detector = dd.DriftDetector()
                detector.fit(X_train, X_test)
                X_train, X_eval, y_train, y_eval = detector.train_test_split(X_train, y_train, test_size=eval_size)
            else:
                if self.task == 'regression':
                    stratify = None
                X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=eval_size,
                                                                    random_state=self.random_state, stratify=stratify)
            self.step_progress('split into train set and eval set')
        else:
            X_eval, y_eval = self.data_cleaner.transform(X_eval, y_eval)
            self.step_progress('transform eval set')

        self.step_end(output={'X_train.shape': X_train.shape,
                              'y_train.shape': y_train.shape,
                              'X_eval.shape': X_eval.shape,
                              'y_eval.shape': y_eval.shape,
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

        # 2. Drop features with multicollinearity
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
            X_eval = X_eval[self.selected_features_]
            X_test = X_test[self.selected_features_]
            self.step_progress('drop features')
            self.step_end(output=self.output_multi_collinearity_)
            # print(self.output_multi_collinearity_)
            display(
                pd.DataFrame([(k, v) for k, v in self.output_multi_collinearity_.items()], columns=['key', 'value']),
                display_id='output_drop_feature_with_collinearity')

        # 3. Shift detection
        if self.drift_detection and self.X_test is not None:
            display_markdown('### Drift detection', raw=True)

            self.step_start('detect drifting')
            features, history, scores = dd.feature_selection(X_train, X_test)

            self.output_drift_detection_ = {'no_drift_features': features, 'history': history}
            self.selected_features_ = features
            X_train = X_train[self.selected_features_]
            X_eval = X_eval[self.selected_features_]
            X_test = X_test[self.selected_features_]
            self.step_end(output=self.output_drift_detection_)

            display(pd.DataFrame((('no drift features', features), ('history', history), ('drift score', scores)),
                                 columns=['key', 'value']), display_id='output_drift_detection')
        # 4. Baseline search
        self.first_hyper_model = copy.deepcopy(hyper_model)
        self.step_start('first stage search')
        display_markdown('### Pipeline search', raw=True)

        kwargs['eval_set'] = (X_eval, y_eval)

        self.first_hyper_model.search(X_train, y_train, X_eval, y_eval, **kwargs)
        self.hyper_model = self.first_hyper_model
        self.step_end(output={'best_reward': self.hyper_model.get_best_trail().reward})

        if self.mode == 'two-stage':
            # 5. Feature importance evaluation
            self.step_start('evaluate feature importance')
            display_markdown('### Evaluate feature importance', raw=True)

            best_trials = self.hyper_model.get_top_trails(self.n_est_feature_importance)
            estimators = []
            for trail in best_trials:
                estimators.append(self.hyper_model.load_estimator(trail.model_file))
            self.step_progress('load estimators')

            importances = feature_importance_batch(estimators, X_eval, y_eval, self.scorer,
                                                   n_repeats=5)
            display_markdown('#### importances', raw=True)

            display(pd.DataFrame(
                zip(importances['columns'], importances['importances_mean'], importances['importances_std']),
                columns=['feature', 'importance', 'std']))
            display_markdown('#### feature selection', raw=True)

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

            display(pd.DataFrame([('Selected', selected_features), ('Unselected', unselected_features)],
                                 columns=['key', 'value']))
            if len(unselected_features) > 0:
                # 6. Final search
                self.step_start('two stage search')
                display_markdown('### Pipeline search stage 2', raw=True)

                self.second_hyper_model = copy.deepcopy(hyper_model)

                kwargs['eval_set'] = (X_eval, y_eval)
                self.second_hyper_model.search(X_train, y_train, X_eval, y_eval, **kwargs)
                self.hyper_model = self.second_hyper_model
                self.step_end(output={'best_reward': self.hyper_model.get_best_trail().reward})
            else:
                display_markdown('### Skip pipeline search stage 2', raw=True)

        # 7. Ensemble
        if self.ensemble_size > 1:
            self.step_start('ensemble')
            display_markdown('### Ensemble', raw=True)

            best_trials = self.hyper_model.get_top_trails(self.ensemble_size)
            estimators = []
            if self.retrain_on_wholedata:
                display_markdown('#### retrain on whole data', raw=True)
                for trail in best_trials:
                    X_all = pd.concat([X_train, X_eval], axis=0)
                    y_all = pd.concat([y_train, y_eval], axis=0)
                    estimator = self.hyper_model.final_train(trail.space_sample, X_all, y_all, **kwargs)
                    estimators.append(estimator)
            else:
                for trail in best_trials:
                    estimators.append(self.hyper_model.load_estimator(trail.model_file))
            ensemble = GreedyEnsemble(self.task, estimators, scoring=self.scorer, ensemble_size=self.ensemble_size)
            ensemble.fit(X_eval, y_eval)
            self.estimator = ensemble
            self.step_end(output={'ensemble': ensemble})
            display(ensemble)
        else:
            display_markdown('### Load best estimator', raw=True)

            self.step_start('load estimator')
            if self.retrain_on_wholedata:
                display_markdown('#### retrain on whole data', raw=True)
                trail = self.hyper_model.get_best_trail()
                X_all = pd.concat([X_train, X_eval], axis=0)
                y_all = pd.concat([y_train, y_eval], axis=0)
                self.estimator = self.hyper_model.final_train(trail.space_sample, X_all, y_all, **kwargs)
            else:
                self.estimator = self.hyper_model.load_estimator(self.hyper_model.get_best_trail().model_file)
            self.step_end()

        droped_features = set(original_features) - set(self.selected_features_)
        self.data_cleaner.append_drop_columns(droped_features)
        # 8. Compose pipeline
        self.step_start('compose pipeline')

        display_markdown('### Compose pipeline', raw=True)

        pipeline = Pipeline([('data_cleaner', self.data_cleaner), ('estimator', self.estimator)])
        self.step_end()
        print(pipeline)
        display_markdown('### Finished', raw=True)
        # 9. Save model
        return pipeline
