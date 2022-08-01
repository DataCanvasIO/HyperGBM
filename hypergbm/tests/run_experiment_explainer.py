# -*- coding:utf-8 -*-
"""

"""

from sklearn.model_selection import train_test_split

from hypergbm import make_experiment
from hypergbm.experiment import PipelineSHAPExplainer
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.metrics import evaluate
import time


def main():
    df = dsutils.load_bank()
    df_train, df_test = train_test_split(df, test_size=0.8, random_state=42)

    experiment = make_experiment(df_train, target='y', max_trials=3, log_level='info', verbose=1, skip_if_file='/tmp/skip.tag')
    estimator = experiment.run()

    explainer = PipelineSHAPExplainer(estimator)
    print(explainer)
    values_list = explainer(df_test)
    print(values_list)


def final_train():
    df = dsutils.load_bank()
    df_train, df_test = train_test_split(df, test_size=0.8, random_state=42)

    experiment = make_experiment(df_train, target='y', max_trials=3, log_level='info', cv=False,
                                 verbose=1, skip_if_file='/tmp/skip.tag', ensemble_size=None)
    estimator = experiment.run()
    explainer = PipelineSHAPExplainer(estimator)
    print(explainer)
    values_list = explainer(df_test)
    print(values_list)


def kernel_test():
    df = dsutils.load_bank()
    df_train, df_test = train_test_split(df, test_size=0.8, random_state=42)

    experiment = make_experiment(df_train, target='y', max_trials=3, log_level='info', cv=False,
                                 verbose=1, skip_if_file='/tmp/skip.tag', ensemble_size=None)
    estimator = experiment.run()

    explainer = PipelineSHAPExplainer(estimator, data=df_train.sample(n=200), method='kernel')
    print(explainer)
    values_list1 = explainer(df_test.head(n=5))
    print(values_list1)
    values_list2 = explainer(df_test.head(n=20))
    print(values_list2)
    print(values_list2)


if __name__ == '__main__':
    t1 = time.time()
    kernel_test()

    print(time.time() - t1)
    # kernel_test()
    # final_train()

# final_train
