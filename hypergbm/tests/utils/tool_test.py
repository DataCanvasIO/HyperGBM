import os
import os.path as path

from hypergbm.utils.tool import main
from hypernets.tabular.datasets import dsutils


class TestTool:
    @classmethod
    def setup_class(cls):
        cls.data_dir = path.split(dsutils.__file__)[0]

    def test_version(self):
        argv = ['-version', ]
        main(argv)

    def test_train(self):
        if path.exists('model.pkl'):
            os.remove('model.pkl')

        data_file = f'{self.data_dir}/blood.csv'
        argv = [
            # '-info','-v',
            'train',
            '--train-data', data_file,
            '--target', 'Class',
        ]
        main(argv)
        assert path.exists('model.pkl')

    def test_predict(self):
        if not path.exists('model.pkl'):
            self.test_train()
        if path.exists('prediction.csv'):
            os.remove('prediction.csv')

        data_file = f'{self.data_dir}/blood.csv'
        argv = [
            # '-info', '-v',
            'predict',
            '--data', data_file,
        ]
        main(argv)
        assert path.exists('prediction.csv')

    def test_evaluate(self):
        if not path.exists('model.pkl'):
            self.test_train()

        data_file = f'{self.data_dir}/blood.csv'
        argv = [
            # '-info', '-v',
            'evaluate',
            '--data', data_file,
            '--target', 'Class',
        ]
        main(argv)
