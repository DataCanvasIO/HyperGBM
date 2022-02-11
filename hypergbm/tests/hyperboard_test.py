# -*- encoding: utf-8 -*-
import json
from hypergbm.callbacks import HyperGBMLogEventHyperModelCallback, HyperGBMLogEventExperimentCallback, \
    HyperGBMNotebookHyperModelCallback, HyperGBMNotebookExperimentCallback
from hypergbm import make_experiment
from hypernets.utils import logging
from hypernets.tests.experiment import experiment_factory

logger = logging.get_logger(__name__)


def _run_experiment(creator):
    webui_model_callback = HyperGBMLogEventHyperModelCallback()
    webui_callback = HyperGBMLogEventExperimentCallback(exit_web_server_on_finish=True)
    nb_model_callback = HyperGBMNotebookHyperModelCallback()
    nb_callback = HyperGBMNotebookExperimentCallback()

    exp = creator(maker=make_experiment,
                  callbacks=[webui_callback, nb_callback],
                  search_callbacks=[webui_model_callback, nb_model_callback])
    estimator = exp.run(max_trials=10)
    assert estimator
    
    logfile = webui_callback.internal_callback.get_log_file(exp)
    assert logfile
    with open(logfile, 'r', newline='\n') as f:
        events = [json.loads(line) for line in f.readlines()]

    assert events
    assert len(events) > 0
    return events


def test_data_clean():
    _run_experiment(experiment_factory.create_data_clean_experiment)


def test_drift_detection():
    _run_experiment(experiment_factory.create_drift_detection_experiment)


def test_multicollinearity_detect():
    _run_experiment(experiment_factory.create_multicollinearity_detect_experiment)


def test_feature_generation():
    _run_experiment(experiment_factory.create_feature_generation_experiment)


def test_feature_reselection_experiment():
    _run_experiment(experiment_factory.create_feature_reselection_experiment)


def test_feature_selection_experiment():
    _run_experiment(experiment_factory.create_feature_selection_experiment)


def test_pseudo_labeling_experiment():
    _run_experiment(experiment_factory.create_pseudo_labeling_experiment)
