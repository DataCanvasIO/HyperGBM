import abc
import pickle
import time
import json
import os
import datetime
from pathlib import Path

from hypernets.core.callbacks import Callback, EarlyStoppingCallback
from hypernets.experiment import EarlyStoppingStatusMeta, ActionType
from hypernets.experiment import ExperimentCallback, \
    ExperimentExtractor, StepMeta, ExperimentMeta
from hypernets.utils import fs, logging as hyn_logging
from hypernets.utils import get_tree_importances
from hypernets.experiment import ExperimentCallback

logger = hyn_logging.get_logger(__name__)


def _prompt_installing():
    logger.warning("No visualization module detected, please install by command:"
                   "pip install experiment-visualization ")


class ParseTrailEventHyperModelCallback(Callback):

    def __init__(self, **kwargs):
        super(ParseTrailEventHyperModelCallback, self).__init__()
        self.max_trials = None
        self.current_running_step_index = None
        self.exp_id = None

    def set_exp_id(self, exp_id):
        self.exp_id = exp_id

    def set_current_running_step_index(self, value):
        self.current_running_step_index = value

    def assert_ready(self):
        assert self.exp_id is not None
        assert self.current_running_step_index is not None

    @staticmethod
    def sort_imp(imp_dict, sort_imp_dict, n_features=10):
        sort_imps = []
        for k in sort_imp_dict:
            sort_imps.append({
                'name': k,
                'imp': sort_imp_dict[k]
            })

        top_features = list(
            map(lambda x: x['name'], sorted(sort_imps, key=lambda v: v['imp'], reverse=True)[: n_features]))

        imps = []
        for f in top_features:
            imps.append({
                'name': f,
                'imp': imp_dict[f]
            })
        return imps

    def on_search_start(self, hyper_model, X, y, X_eval, y_eval, cv,
                        num_folds, max_trials, dataset_id, trial_store, **fit_kwargs):
        self.max_trials = max_trials  # to record trail summary info

    @staticmethod
    def get_early_stopping_status_data(hyper_model):
        """ Return early stopping if triggered
        :param hyper_model:
        :return:
        """
        # check whether end cause by early stopping
        for c in hyper_model.callbacks:
            if isinstance(c, EarlyStoppingCallback):
                if c.triggered:
                    if c.start_time is not None:
                        elapsed_time = time.time() - c.start_time
                    else:
                        elapsed_time = None
                    ess = EarlyStoppingStatusMeta(c.best_reward, c.best_trial_no, c.counter_no_improvement_trials,
                                                  c.triggered, c.triggered_reason, elapsed_time)
                    return ess
        return None

    def on_search_end(self, hyper_model):
        self.assert_ready()
        early_stopping_data = self.get_early_stopping_status_data(hyper_model)
        self.on_search_end_(hyper_model, early_stopping_data)

    def on_search_end_(self, hyper_model, early_stopping_data):
        pass

    @staticmethod
    def get_space_params(space):
        params_dict = {}
        for hyper_param in space.get_assigned_params():
            # param_name = hyper_param.alias[len(list(hyper_param.references)[0].name) + 1:]
            param_name = hyper_param.alias
            param_value = hyper_param.value
            # only show number param
            # if isinstance(param_value, int) or isinstance(param_value, float):
            #     if not isinstance(param_value, bool):
            #         params_dict[param_name] = param_value
            if param_name is not None and param_value is not None:
                # params_dict[param_name.split('.')[-1]] = str(param_value)
                params_dict[param_name] = str(param_value)
        return params_dict

    @staticmethod
    def assert_int_param(value, var_name):
        if value is None:
            raise ValueError(f"Var {var_name} can not be None.")
        else:
            if not isinstance(value, float) and not isinstance(value, int):
                raise ValueError(f"Var {var_name} = {value} not a number.")

    @staticmethod
    def get_trail_by_no(hyper_model, trial_no):
        for t in hyper_model.history.trials:
            if t.trial_no == trial_no:
                return t
        return None

    def on_trial_end(self, hyper_model, space, trial_no, reward, improved, elapsed):
        self.assert_ready()

        self.assert_int_param(reward, 'reward')
        self.assert_int_param(trial_no, 'trail_no')
        self.assert_int_param(elapsed, 'elapsed')

        # pass
        trial = self.get_trail_by_no(hyper_model, trial_no)

        if trial is None:
            raise Exception(f"Trial no {trial_no} is not in history")

        model_file = trial.model_file
        with fs.open(model_file, 'rb') as f:
            model = pickle.load(f)

        cv_models = model.cv_gbm_models_
        models_json = []
        is_cv = cv_models is not None and len(cv_models) > 0
        if is_cv:
            # cv is opening
            imps = []
            for m in cv_models:
                imps.append(get_tree_importances(m))

            imps_avg = {}
            for k in imps[0]:
                imps_avg[k] = sum([imp.get(k, 0) for imp in imps]) / 3

            for fold, m in enumerate(cv_models):
                models_json.append({
                    'fold': fold,
                    'importances': self.sort_imp(get_tree_importances(m), imps_avg)
                })
        else:
            gbm_model = model.gbm_model
            if gbm_model is None:
                raise Exception("Both cv_models or gbm_model is None ")
            imp_dict = get_tree_importances(gbm_model)
            models_json.append({
                'fold': None,
                'importances': self.sort_imp(imp_dict, imp_dict)
            })
        early_stopping_status = None
        for c in hyper_model.callbacks:
            if isinstance(c, EarlyStoppingCallback):
                early_stopping_status = EarlyStoppingStatusMeta(c.best_reward, c.best_trial_no,
                                                                c.counter_no_improvement_trials,
                                                                c.triggered,
                                                                c.triggered_reason,
                                                                time.time() - c.start_time).to_dict()
                break
        trial_data = {
            "trialNo": trial_no,
            "maxTrials": self.max_trials,
            "hyperParams": self.get_space_params(space),
            "models": models_json,
            "reward": reward,
            "elapsed": elapsed,
            "is_cv": is_cv,
            "metricName": hyper_model.reward_metric,
            "earlyStopping": early_stopping_status
        }
        data = {
            'stepIndex': self.current_running_step_index,
            'trialData': trial_data
        }
        self.on_trial_end_(hyper_model, space, trial_no, reward, improved, elapsed, data)

    def on_trial_end_(self, hyper_model, space, trial_no, reward, improved, elapsed, trial_data):
        pass


class HyperGBMLogEventHyperModelCallback(ParseTrailEventHyperModelCallback):

    def __init__(self):
        super(HyperGBMLogEventHyperModelCallback, self).__init__()
        self.log_file = None

    def set_log_file(self, log_file):
        self.log_file = log_file

    def assert_ready(self):
        super(HyperGBMLogEventHyperModelCallback, self).assert_ready()
        assert self.log_file is not None

    def on_search_end_(self, hyper_model, early_stopping_data):
        from experiment_visualization.callbacks import append_event_to_file
        if early_stopping_data is not None:
            payload = early_stopping_data.to_dict()
            append_event_to_file(self.log_file, ActionType.EarlyStopped, payload)

    def on_trial_end_(self, hyper_model, space, trial_no, reward, improved, elapsed, trial_data):
        from experiment_visualization.callbacks import append_event_to_file
        append_event_to_file(self.log_file, ActionType.TrialEnd, trial_data)


class ExperimentCallbackProxy(ExperimentCallback):

    def __init__(self):
        self.internal_callback = None

    def experiment_start(self, exp):
        if self.internal_callback is not None:
            self.internal_callback.experiment_start(exp)

    def experiment_end(self, exp, elapsed):
        if self.internal_callback is not None:
            self.internal_callback.experiment_end(exp, elapsed)

    def experiment_break(self, exp, error):
        if self.internal_callback is not None:
            self.internal_callback.experiment_break(exp, error)

    def step_start(self, exp, step):
        if self.internal_callback is not None:
            self.internal_callback.step_start(exp, step)

    def step_progress(self, exp, step, progress, elapsed, eta=None):
        if self.internal_callback is not None:
            self.internal_callback.step_progress(exp, step, progress, elapsed, eta=None)

    def step_end(self, exp, step, output, elapsed):
        if self.internal_callback is not None:
            self.internal_callback.step_end(exp, step, output, elapsed)

    def step_break(self, exp, step, error):
        if self.internal_callback is not None:
            self.internal_callback.step_break(exp, step, error)


class HyperGBMLogEventExperimentCallback(ExperimentCallbackProxy):
    
    def __init__(self, **kwargs):
        super(HyperGBMLogEventExperimentCallback, self).__init__()
        try:
            from experiment_visualization.callbacks import LogEventExperimentCallback
            self.internal_callback = LogEventExperimentCallback(HyperGBMLogEventHyperModelCallback, **kwargs)
        except Exception as e:
            logger.warning("No visualization module detected, please install by command:"
                           "pip install experiment-notebook-widget")
            logger.exception(e)


class HyperGBMNotebookHyperModelCallback(ParseTrailEventHyperModelCallback):

    def send_action(self, action_type, payload):
        from experiment_notebook_widget.callbacks import NotebookExperimentCallback
        self.assert_ready()
        NotebookExperimentCallback.send_action(self.exp_id, action_type, payload)

    def on_search_end_(self, hyper_model, early_stopping_data):
        if early_stopping_data is not None:  # early stopping triggered
            payload = {
                'stepIndex': self.current_running_step_index,
                'data': early_stopping_data.to_dict()
            }
            self.send_action(ActionType.EarlyStopped, payload)

    def on_trial_end_(self, hyper_model, space, trial_no, reward, improved, elapsed, trial_data):
        self.send_action(ActionType.TrialEnd, trial_data)


class HyperGBMNotebookExperimentCallback(ExperimentCallbackProxy):

    def __init__(self):
        super(HyperGBMNotebookExperimentCallback, self).__init__()
        try:
            from experiment_notebook_widget.callbacks import NotebookExperimentCallback
            self.internal_callback = NotebookExperimentCallback(HyperGBMNotebookHyperModelCallback)
        except Exception as e:
            logger.warning("No visualization module detected, please install by command:"
                           "pip install experiment-notebook-widget ")
            logger.exception(e)
