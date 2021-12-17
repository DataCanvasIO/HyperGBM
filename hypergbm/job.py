from hypernets.experiment._job import DatasetConf, JobEngine


class HyperGBMJobEngine(JobEngine):

    @staticmethod
    def name():
        return "hypergbm"

    def create_experiment(self):
        from hypergbm.experiment import make_experiment

        dateset_conf: DatasetConf = self.job_conf.dataset

        assert dateset_conf.train_file is not None, "train_file can not be None"
        assert dateset_conf.target is not None, "target can not be None"

        experiment_conf: dict = self.job_conf.experiment
        make_kwargs = self._flat_compete_experiment_conf(experiment_conf)

        train_data = self._read_file(dateset_conf.train_file)
        make_kwargs['train_data'] = train_data
        make_kwargs['target'] = dateset_conf.target

        if dateset_conf.eval_file is not None:
            make_kwargs['eval_data'] = self._read_file(dateset_conf.eval_file)

        if dateset_conf.test_file is not None:
            make_kwargs['test_data'] = self._read_file(dateset_conf.test_file)

        if dateset_conf.task is not None:
            make_kwargs['task'] = dateset_conf.task

        # replace report file path
        if make_kwargs.get('report_render') == 'excel':
            report_render_options = make_kwargs.get('report_render_options')
            default_excel_report_path = f"{self.job_conf.working_dir}/report.xlsx"
            if report_render_options is not None:
                if report_render_options.get('file_path') is None:
                    report_render_options['file_path'] = default_excel_report_path
                    make_kwargs['report_render_options'] = report_render_options
            else:
                report_render_options = {'file_path': default_excel_report_path}
                make_kwargs['report_render_options'] = report_render_options

        job_working_dir = f"{self.job_conf.working_dir}/{self.job_conf.name}"

        # set default prediction dir if enable persist
        if make_kwargs.get('evaluation_persist_prediction') is True:
            if make_kwargs.get('evaluation_persist_prediction_dir') is None:
                make_kwargs['evaluation_persist_prediction_dir'] = f"{job_working_dir}/prediction"

        # set default report file path
        if make_kwargs.get('report_render') == 'excel':
            report_render_options = make_kwargs.get('report_render_options')
            default_excel_report_path = f"{job_working_dir}/report.xlsx"
            if report_render_options is not None:
                if report_render_options.get('file_path') is None:
                    report_render_options['file_path'] = default_excel_report_path
                    make_kwargs['report_render_options'] = report_render_options
                else:
                    pass  # use user setting
            else:
                make_kwargs['report_render_options'] = {'file_path': default_excel_report_path}

        exp = make_experiment(**make_kwargs)
        return exp
