# -*- coding:utf-8 -*-
"""

"""

from sklearn.model_selection import train_test_split

from hypergbm import make_experiment
from hypergbm.search_space import search_space_general_gpu
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.metrics import evaluate


def main():
    print('search space:', search_space_general_gpu)
    print('>' * 30)

    df = dsutils.load_bank()
    df_train, df_test = train_test_split(df, test_size=0.8, random_state=42)

    experiment = make_experiment(df_train, target='y',
                                 search_space=search_space_general_gpu,
                                 max_trials=50, log_level='info', verbose=1, skip_if_file='/tmp/skip.tag'
                                 )
    estimator = experiment.run()

    X_test = df_test
    y_test = X_test.pop('y')
    pred = estimator.predict(X_test)
    result = evaluate(estimator, X_test, y_test, metrics=['auc', 'accuracy'])
    print(f'final result:{result}')


if __name__ == '__main__':
    main()
