# -*- encoding: utf-8 -*-

from hypergbm.callbacks import HyperGBMLogEventHyperModelCallback, HyperGBMLogEventExperimentCallback
from hypergbm import make_experiment
from hypernets.utils import logging

from hypernets.tests.experiment import experiment_factory

logger = logging.get_logger(__name__)


def _run_vis_experiment(creator):
    model_callback = HyperGBMLogEventHyperModelCallback()
    callback = HyperGBMLogEventExperimentCallback()
    exp = creator(maker=make_experiment,
                  callbacks=[callback],
                  search_callbacks=[model_callback])
    estimator = exp.run(max_trials=10)
    return estimator


def run_data_clean():
    _run_vis_experiment(experiment_factory.create_data_clean_experiment)


def run_drift_detection():
    _run_vis_experiment(experiment_factory.create_drift_detection_experiment)


def run_multicollinearity_detect():
    _run_vis_experiment(experiment_factory.create_multicollinearity_detect_experiment)


def run_feature_generation():
    _run_vis_experiment(experiment_factory.create_feature_generation_experiment)


def run_feature_reselection_experiment():
    _run_vis_experiment(experiment_factory.create_feature_reselection_experiment)


def run_feature_selection_experiment():
    _run_vis_experiment(experiment_factory.create_feature_selection_experiment)


def run_pseudo_labeling_experiment():
    _run_vis_experiment(experiment_factory.create_pseudo_labeling_experiment)


if __name__ == '__main__':
    pass
    # run_data_clean()
    # run_drift_detection()
    # run_multicollinearity_detect()
    # run_feature_generation()
    # run_feature_selection_experiment()
    # run_feature_reselection_experiment()
    # run_pseudo_labeling_experiment()
