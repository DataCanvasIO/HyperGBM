# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import os.path

from hypernets.discriminators import UnPromisingTrial

class BaseDiscriminationCallback(object):
    def __init__(self, discriminator, group_id):
        self.discriminator = discriminator
        self.group_id = group_id
        self.iteration_trajectory = []
        self.is_promising_ = True
        self.show_iteration_trajectory_len=6

    def iteration(self, score, end_iteration):
        self.iteration_trajectory.append(score)
        promising = self.discriminator.is_promising(self.iteration_trajectory, self.group_id, end_iteration)
        self.is_promising_ = promising
        if not promising:
            raise UnPromisingTrial(f'unpromising trial:{self.iteration_trajectory[-self.show_iteration_trajectory_len:]}')

    def _get_score(self, env):
        raise NotImplementedError

    def __call__(self, env):
        score = self._get_score(env)
        self.iteration(score, env.end_iteration)


class FileMonitorCallback(object):
    def __init__(self, file_path):
        assert isinstance(file_path, str) and len(file_path) > 0

        if os.path.exists(file_path):
            os.remove(file_path)

        self.file_path = file_path

    def __call__(self, env):
        if os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
                removed = True
            except Exception as e:
                removed = False
            if removed:
                raise UnPromisingTrial(f'found file {self.file_path}, skip trial')

    def after_iteration(self, *args, **kwargs):
        return True
