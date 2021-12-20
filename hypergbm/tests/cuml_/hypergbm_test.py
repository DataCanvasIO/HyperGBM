# -*- coding:utf-8 -*-
"""

"""
from . import is_cuml_installed, if_cuml_ready

if is_cuml_installed:
    from .run_hypergbm_cuml import main


@if_cuml_ready
class TestSearchWithHyperGBM:
    def test_binary(self):
        est = main(target='y', max_trials=3, verbose=0, pos_label='yes', )
        assert est is not None

    def test_binary_class_balancing(self):
        est = main(class_balancing=True, target='y', max_trials=3, verbose=0, pos_label='yes', )
        assert est is not None

    def test_multiclass(self):
        est = main(target='day', max_trials=3, verbose=0, )
        assert est is not None

    def test_regression(self):
        est = main(target='age', dtype='float', max_trials=3, verbose=0, )
        assert est is not None

    def test_regression_histgb(self):
        est = main(enable_catboost=False, enable_lightgbm=False, enable_xgb=False, enable_histgb=True,
                   target='age', dtype='float', max_trials=3, verbose=0, )
        assert est is not None

    def test_binary_histgb(self):
        est = main(enable_catboost=False, enable_lightgbm=False, enable_xgb=False, enable_histgb=True,
                   target='y', max_trials=3, verbose=0, pos_label='yes', )
        assert est is not None

    def test_complex_pipeline(self):
        est = main(num_pipeline_mode='complex', cat_pipeline_mode='complex',
                   target='y', max_trials=10, verbose=0, pos_label='yes', )
        assert est is not None
