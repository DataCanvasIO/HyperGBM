# -*- coding:utf-8 -*-
"""

"""
from hypernets.tabular.cuml_ex import CumlToolBox, Localizable, copy_attrs_as_local
from ..hyper_gbm import HyperGBMEstimator, HyperGBM


class CumlHyperGBMEstimator(HyperGBMEstimator, Localizable):
    @staticmethod
    def _create_pipeline(steps):
        return CumlToolBox.transformers['Pipeline'](steps=steps)

    def as_local(self):
        # create target instance with "space_sample=None" to disable building
        target = HyperGBMEstimator(self.task, self.reward_metric, None, self.data_cleaner_params)

        attrs = [# 'space_sample',  # init args
                 'model', 'class_balancing', 'fit_kwargs', 'data_pipeline',  # built
                 'data_cleaner', 'classes_', 'pos_label', 'transients_', 'cv_models_', 'cv_',  # fitted
                 ]
        copy_attrs_as_local(self, target, *attrs)
        return target


class CumlHyperGBM(HyperGBM):
    estimator_cls = CumlHyperGBMEstimator
