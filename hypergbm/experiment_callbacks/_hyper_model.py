import pickle

from hypernets.experiment import ABSExpVisHyperModelCallback
from hypernets.experiment import ActionType
from hypernets.utils import fs, logging as hyn_logging
from hypernets.utils import get_tree_importances

logger = hyn_logging.get_logger(__name__)


def _parse_trial_end_event(hyper_model, space, trial_no, reward, improved,
                           elapsed, max_trials, step_index):

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

    def assert_int_param(value, var_name):
        if value is None:
            raise ValueError(f"Var {var_name} can not be None.")
        else:
            if not isinstance(value, float) and not isinstance(value, int):
                raise ValueError(f"Var {var_name} = {value} not a number.")

    trial = None
    for t in hyper_model.history.trials:
        if t.trial_no == trial_no:
            trial = t

    if trial is None:
        raise Exception(f"Trial no {trial_no} is not in history")

    assert_int_param(reward, 'reward')
    assert_int_param(trial_no, 'trail_no')
    assert_int_param(elapsed, 'elapsed')

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
                'importances': sort_imp(get_tree_importances(m), imps_avg)
            })
    else:
        gbm_model = model.gbm_model
        if gbm_model is None:
            raise Exception("Both cv_models or gbm_model is None ")
        imp_dict = get_tree_importances(gbm_model)
        models_json.append({
            'fold': None,
            'importances': sort_imp(imp_dict, imp_dict)
        })

    early_stopping_status = ABSExpVisHyperModelCallback.get_early_stopping_status_data(hyper_model)
    hyper_params = ABSExpVisHyperModelCallback.get_space_params(space)

    def get_reward_metric_name(reward_metric):  # reward_metric maybe a function/str/instance of custom class(callable)
        if isinstance(reward_metric, str):
            return reward_metric
        else:
            return str(reward_metric.__name__)  # custom metrics or function

    trial_data = {
        "trialNo": trial_no,
        "maxTrials": max_trials,
        "hyperParams": hyper_params,
        "models": models_json,
        "reward": reward,
        "elapsed": elapsed,
        "is_cv": is_cv,
        "metricName": get_reward_metric_name(hyper_model.reward_metric),
        "earlyStopping": early_stopping_status
    }

    data = {
        'stepIndex': step_index,
        'trialData': trial_data
    }

    return data


class HyperGBMNotebookHyperModelCallback(ABSExpVisHyperModelCallback):

    def send_action(self, action_type, payload):
        from hboard_widget.callbacks import NotebookExperimentCallback
        self.assert_ready()
        NotebookExperimentCallback.send_action(self.exp_id, action_type, payload)

    def on_search_end_(self, hyper_model, early_stopping_data):
        if early_stopping_data is not None:  # early stopping triggered
            payload = {
                'stepIndex': self.current_running_step_index,
                'data': early_stopping_data.to_dict()
            }
            self.send_action(ActionType.EarlyStopped, payload)

    def on_trial_end(self, hyper_model, space, trial_no, reward, improved, elapsed):
        reward = reward[0]
        self.assert_ready()
        trial_event_data = _parse_trial_end_event(hyper_model, space, trial_no, reward, improved, elapsed,
                                                  self.max_trials, self.current_running_step_index)
        self.send_action(ActionType.TrialEnd, trial_event_data)


def create_notebook_hyper_model_callback():
    return HyperGBMNotebookHyperModelCallback()


def create_web_vis_hyper_model_callback():
    from hboard.callbacks import WebVisHyperModelCallback
    return WebVisHyperModelCallback(_parse_trial_end_event)
