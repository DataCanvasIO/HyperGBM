# -*- coding:utf-8 -*-
"""

"""

from sklearn.model_selection import train_test_split

from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.metrics import evaluate
import pandas as pd


def main():

    # target = 'price'
    target = 'Total Compensation'
    df_train = pd.read_csv('C:/Users/wuhf/Desktop/san-francisco-employee-salary-compensation/train.csv').sample(2000)
    df_test = pd.read_csv('C:/Users/wuhf/Desktop/san-francisco-employee-salary-compensation/test.csv').sample(1000)

    df_test.dropna(axis=0, how='any', subset=[target], inplace=True)

    experiment = make_experiment(df_train, eval_data=df_test, target=target,
                                 max_trials=1, log_level='info', verbose=1, report_render='excel', ensemble_size=2)
    estimator = experiment.run()


if __name__ == '__main__':
    main()
