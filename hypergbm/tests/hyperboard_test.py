# -*- encoding: utf-8 -*-
import json

import pytest

from hypernets.utils import logging
from hypernets.tests.experiment import experiment_factory

from hypergbm.experiment_callbacks import create_web_vis_experiment_callback, create_notebook_experiment_callback, \
    create_notebook_hyper_model_callback, create_web_vis_hyper_model_callback
from hypergbm import make_experiment

logger = logging.get_logger(__name__)


def _notebook_widget_and_web_app_ready():
    try:
        import hboard
        import hboard_widget
        return True
    except:
        return False


need_hboard = pytest.mark.skipif(not _notebook_widget_and_web_app_ready(),
                                 reason='hboard or hboard-widget not installed')


def _run_experiment(creator):
    webui_model_callback = create_web_vis_hyper_model_callback()
    webui_callback = create_web_vis_experiment_callback(exit_web_server_on_finish=True)
    nb_model_callback = create_notebook_hyper_model_callback()
    nb_callback = create_notebook_experiment_callback()

    exp = creator(maker=make_experiment,
                  callbacks=[webui_callback, nb_callback],
                  search_callbacks=[webui_model_callback, nb_model_callback])
    estimator = exp.run(max_trials=3)
    assert estimator
    logfile = webui_callback.get_log_file(exp)
    assert logfile
    with open(logfile, 'r', newline='\n') as f:
        events = [json.loads(line) for line in f.readlines()]

    assert events
    assert len(events) > 0
    return events


@need_hboard
def test_data_clean():
    _run_experiment(experiment_factory.create_data_clean_experiment)


@need_hboard
def test_drift_detection():
    _run_experiment(experiment_factory.create_drift_detection_experiment)


@need_hboard
def test_multicollinearity_detect():
    _run_experiment(experiment_factory.create_multicollinearity_detect_experiment)


@need_hboard
def test_feature_generation():
    _run_experiment(experiment_factory.create_feature_generation_experiment)


@need_hboard
def test_feature_reselection_experiment():
    _run_experiment(experiment_factory.create_feature_reselection_experiment)


@need_hboard
def test_feature_selection_experiment():
    _run_experiment(experiment_factory.create_feature_selection_experiment)


@need_hboard
def test_pseudo_labeling_experiment():
    _run_experiment(experiment_factory.create_pseudo_labeling_experiment)


@need_hboard
def test_disable_cv():
    _run_experiment(experiment_factory.create_disable_cv_experiment)


@need_hboard
def test_custom_metric_func():
    _run_experiment(experiment_factory.create_custom_reward_metric_func_experiment)


@need_hboard
def test_custom_metric_class():
    _run_experiment(experiment_factory.create_custom_reward_metric_class_experiment)
