from .run_hypergbm_dask import main


def test_experiment():
    est = main(max_trials=3, log_level='warn')
    assert est is not None
