# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from ._base import BaseDiscriminationCallback
from hypernets.discriminators import UnPromisingTrial


class CatboostDiscriminationCallback(BaseDiscriminationCallback):

    def after_iteration(self, info):
        score = self._get_score(info)
        self.iteration_trajectory.append(score)
        self.is_promising_ = self.discriminator.is_promising(self.iteration_trajectory, self.group_id, -1)
        return self.is_promising_

    def _get_score(self, info):
        if len(list(info.metrics['validation'].items())) > 0:
            score = list(info.metrics['validation'].items())[0][1][-1]
            return score
        else:
            raise ValueError('Evaluation result not found.')
