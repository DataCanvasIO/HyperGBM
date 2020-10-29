# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.experiment.experiment import Experiment
from sklearn.model_selection import train_test_split



class CompeteExperiment(Experiment):
    def __init__(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3,
                 random_state=9527):
        super().__init__(hyper_model, X_train, y_train, X_test, X_eval=X_eval, y_eval=y_eval, eval_size=eval_size,
                         random_state=random_state)

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


        #1. Clean data

        #2. Shift detection

        #3. Feature generation

        #4. Baseline search

        #5. Feature importance evaluation

        #6. Final search

        #7. Ensemble

        #8. Compose pipeline

        #9. Save model

        hyper_model.search(X_train, y_train, X_eval, y_eval, max_trails=max_trails)
        best_trial = hyper_model.get_best_trail()

        self.estimator = hyper_model.final_train(best_trial.space_sample, X_train, y_train)
        return self.estimator
