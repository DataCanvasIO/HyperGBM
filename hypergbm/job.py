from hypernets.experiment.job import CompeteExperimentJobCreator


class HyperGBMExperimentJobCreator(CompeteExperimentJobCreator):

    def _create_experiment(self, make_options):
        from hypergbm.experiment import make_experiment

        train_data = make_options.get('train_data')
        assert train_data
        assert isinstance(train_data, str)

        make_options['train_data'] = self._read_file(train_data)

        eval_data = make_options.get('eval_data')
        if eval_data is not None:
            assert isinstance(eval_data, str)
            make_options['eval_data'] = self._read_file(make_options.get('eval_data'))

        test_data = make_options.get('test_data')
        if test_data is not None:
            assert isinstance(test_data, str)
            make_options['test_data'] = self._read_file(test_data)

        exp = make_experiment(**make_options)
        return exp


if __name__ == "__main__":
    HyperGBMExperimentJobCreator().create_and_run_experiment()
