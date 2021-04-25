# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

class BaseDiscriminationCallback(object):
    def __init__(self, discriminator, group_id):
        self.discriminator = discriminator
        self.group_id = group_id
        self.iteration_trajectory = []

    def iteration(self, score):
        self.iteration_trajectory.append(score)
        promising = self.discriminator.is_promising(self.iteration_trajectory, self.group_id)
        if not promising:
            raise ValueError('unpromising trial')

    def _get_score(self, env):
        raise NotImplementedError

    def __call__(self, env):
        score = self._get_score(env)
        self.iteration(score)
