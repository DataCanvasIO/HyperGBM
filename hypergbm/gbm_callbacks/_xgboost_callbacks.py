# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import xgboost

from hypernets.utils import Version
from ._base import BaseDiscriminationCallback, FileMonitorCallback

if Version(xgboost.__version__) < Version('1.6'):
    class XGBoostDiscriminationCallback(BaseDiscriminationCallback):
        def __init__(self, discriminator, group_id, n_estimator):
            super().__init__(discriminator, group_id)

        def _get_score(self, env):
            if len(env.evaluation_result_list) > 0:
                score = env.evaluation_result_list[0][1]
                return score
            else:
                raise ValueError('Evaluation result not found.')


    class XGBoostFileMonitorCallback(FileMonitorCallback):
        pass

else:

    from xgboost.callback import TrainingCallback


    class XGBoostDiscriminationCallback(BaseDiscriminationCallback, TrainingCallback):
        def __init__(self, discriminator, group_id, n_estimator):
            super().__init__(discriminator, group_id)

            self.n_estimator = n_estimator

        def _get_score(self, env):
            for _, ev in env.items():
                for _, s in ev.items():
                    score = s[0]
                    return score

            raise ValueError('Evaluation result not found.')

        def after_iteration(self, model, epoch, evals_log) -> bool:
            score = self._get_score(evals_log)
            self.iteration(score, self.n_estimator)
            return False


    class XGBoostFileMonitorCallback(FileMonitorCallback, TrainingCallback):
        def after_iteration(self, model, epoch, evals_log) -> bool:
            self.__call__(None)
            return False
