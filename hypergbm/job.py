from hypernets.experiment.job import DatasetConf, CompeteExperimentJobEngine


class HyperGBMJobEngine(CompeteExperimentJobEngine):

    def _create_experiment(self, make_options):
        from hypergbm.experiment import make_experiment
        dateset_conf: DatasetConf = self.job_conf.dataset

        make_options['train_data'] = self._read_file(dateset_conf.train_file)

        if dateset_conf.target is not None:
            make_options['target'] = dateset_conf.target

        if dateset_conf.eval_file is not None:
            make_options['eval_data'] = self._read_file(dateset_conf.eval_file)

        if dateset_conf.test_file is not None:
            make_options['test_data'] = self._read_file(dateset_conf.test_file)

        if dateset_conf.task is not None:
            make_options['task'] = dateset_conf.task

        exp = make_experiment(**make_options)
        return exp
