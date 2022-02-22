from hypernets.utils import logging as hyn_logging

from ._hyper_model import HyperGBMNotebookHyperModelCallback

logger = hyn_logging.get_logger(__name__)


def create_notebook_experiment_callback():
    try:
        from experiment_notebook_widget.callbacks import NotebookExperimentCallback
        return NotebookExperimentCallback(HyperGBMNotebookHyperModelCallback)
    except Exception as e:
        logger.warning("No notebook visualization module detected, please install by command:"
                       "\"pip install experiment-notebook-widget\"")
        logger.exception(e)
        return None


def create_web_vis_experiment_callback(**kwargs):
    from experiment_visualization.callbacks import WebVisExperimentCallback
    return WebVisExperimentCallback(**kwargs)
