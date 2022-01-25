from hypernets.tests.tabular.tb_dask import if_dask_ready


@if_dask_ready
def test_experiment():
    from .run_hypergbm_dask import main
    est = main(max_trials=3, log_level='warn')
    assert est is not None
