from hypergbm.job import HyperGBMExperimentJobCreator
from hypernets.experiment.job import ExperimentJobCreator

class GPUHyperGBMExperimentJobCreator(HyperGBMExperimentJobCreator):

    @staticmethod
    def _read_file(file_path):
        df = ExperimentJobCreator._read_file(file_path)
        import cudf
        return cudf.from_pandas(df)


if __name__ == "__main__":
    GPUHyperGBMExperimentJobCreator().create_and_run_experiment()
