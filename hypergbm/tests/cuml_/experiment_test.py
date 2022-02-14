# -*- coding:utf-8 -*-
"""

"""
from . import is_cuml_installed, if_cuml_ready

if is_cuml_installed:
    from .run_experiment_cuml import main, main_with_adaption


@if_cuml_ready
class TestCumlExperiment:
    def test_binary_default(self):
        exp, est = main(target='y', max_trials=3, verbose=0, pos_label='yes', )
        assert est is not None

    def test_binary_no_cv(self):
        exp, est = main(target='y', max_trials=3, cv=False, ensemble_size=0, verbose=0, pos_label='yes', )
        assert est is not None

    def test_binary_no_ensemble(self):
        exp, est = main(target='y', max_trials=3, ensemble_size=0, verbose=0, pos_label='yes', )
        assert est is not None

    def test_binary_metric_f1(self):
        exp, est = main(target='y', reward_metric='auc', max_trials=3, verbose=0, pos_label='yes')
        assert est is not None

    def test_binary_feature_selection(self):
        exp, est = main(target='y', reward_metric='auc', max_trials=3, verbose=0, pos_label='yes',
                        feature_selection=True)
        assert est is not None

    def test_binary_drift_detection(self):
        exp, est = main(target='y', reward_metric='auc', max_trials=3, verbose=0, pos_label='yes',
                        drift_detection=True)
        assert est is not None

    def test_binary_pseudo_labeling(self):
        exp, est = main(target='y', reward_metric='auc', max_trials=3, verbose=0, pos_label='yes',
                        pseudo_labeling=True, random_state=31772, log_level='info')
        assert est is not None

    def test_binary_feature_reselection(self):
        exp, est = main(target='y', reward_metric='auc', max_trials=3, verbose=0, pos_label='yes',
                        feature_reselection=True)
        assert est is not None

    def test_multiclass_default(self):
        exp, est = main(target='day', max_trials=3, verbose=0, )
        assert est is not None

    def test_regression_default(self):
        exp, est = main(target='age', dtype='float', max_trials=3, verbose=0, )
        assert est is not None

    def test_binary_with_data_adaption(self):
        exp, est = main_with_adaption(target='y', max_trials=3, verbose=0, pos_label='yes', )
        assert est is not None
