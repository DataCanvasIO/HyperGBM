from hypernets.experiment import ExperimentCallback
from hypernets.utils import logging as hyn_logging

from ._hyper_model import HyperGBMWebVisHyperModelCallback, HyperGBMNotebookHyperModelCallback

logger = hyn_logging.get_logger(__name__)


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


class HyperGBMWebVisExperimentCallback(ExperimentCallbackProxy):

    def __init__(self, **kwargs):
        super(HyperGBMWebVisExperimentCallback, self).__init__()
        try:
            from experiment_visualization.callbacks import WebVisExperimentCallback
            self.internal_callback = WebVisExperimentCallback(HyperGBMWebVisHyperModelCallback, **kwargs)
        except Exception as e:
            logger.warning("No visualization module detected, please install by command:"
                           "pip install experiment-notebook-widget")
            logger.exception(e)

