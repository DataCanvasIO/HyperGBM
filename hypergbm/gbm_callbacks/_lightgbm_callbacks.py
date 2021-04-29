# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from ._base import BaseDiscriminationCallback

class LightGBMDiscriminationCallback(BaseDiscriminationCallback):
    def _get_score(self, env):
        if len(env.evaluation_result_list)>0:
            score = env.evaluation_result_list[0][2]
            return score
        else:
            raise ValueError('Evaluation result not found.')