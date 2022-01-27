import os
import os.path as path
import pickle

import pandas as pd

from hypergbm.utils.tool import main
from hypernets.core import TrialHistory
from hypernets.tabular.datasets import dsutils


class TestTool:
    @classmethod
    def setup_class(cls):
        cls.data_dir = path.split(dsutils.__file__)[0]

    @classmethod
    def teardown_class(cls):
        for f in ['model.pkl', 'prediction.csv', 'perf.csv', 'history.pkl']:
            try:
                if path.exists(f):
                    os.remove(f)
            except:
                pass

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

    def test_train_with_perf(self):
        if path.exists('model.pkl'):
            os.remove('model.pkl')

        data_file = f'{self.data_dir}/blood.csv'
        argv = [
            # '-info','-v',
            '--perf-file', 'perf.csv',
            'train',
            '--train-data', data_file,
            '--target', 'Class',
        ]
        main(argv)
        assert path.exists('model.pkl')
        assert path.exists('perf.csv')

        perf = pd.read_csv('perf.csv')
        assert perf is not None

    def test_train_with_history(self):
        if path.exists('model.pkl'):
            os.remove('model.pkl')
        if path.exists('history.pkl'):
            os.remove('history.pkl')

        data_file = f'{self.data_dir}/blood.csv'
        argv = [
            # '-info','-v',
            'train',
            '--train-data', data_file,
            '--target', 'Class',
            '--history', 'history.pkl',
        ]
        main(argv)
        assert path.exists('model.pkl')
        assert path.exists('history.pkl')

        with open('history.pkl', 'rb') as f:
            history = pickle.load(f)
            assert isinstance(history, TrialHistory)

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
