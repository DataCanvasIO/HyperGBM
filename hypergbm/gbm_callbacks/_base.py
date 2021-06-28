# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.discriminators import UnPromisingTrial


class BaseDiscriminationCallback(object):
    def __init__(self, discriminator, group_id):
        self.discriminator = discriminator
        self.group_id = group_id
        self.iteration_trajectory = []
        self.is_promising_ = True

    def iteration(self, score):
        self.iteration_trajectory.append(score)
        promising = self.discriminator.is_promising(self.iteration_trajectory, self.group_id)
        self.is_promising_ = promising
        if not promising:
            raise UnPromisingTrial(f'unpromising trial:{self.iteration_trajectory}')

    def _get_score(self, env):
        raise NotImplementedError

    def __call__(self, env):
        score = self._get_score(env)
        self.iteration(score)
