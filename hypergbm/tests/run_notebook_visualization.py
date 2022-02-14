# -*- encoding: utf-8 -*-
from hypergbm.callbacks import HyperGBMNotebookExperimentCallback, HyperGBMNotebookHyperModelCallback
from hypernets.utils import logging

from sklearn.model_selection import train_test_split

from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.metrics import evaluate


logger = logging.get_logger(__name__)

model_callback = HyperGBMNotebookHyperModelCallback()
callback = HyperGBMNotebookExperimentCallback()


df = dsutils.load_bank()

df_train, df_test = train_test_split(df, test_size=0.8, random_state=42)

experiment = make_experiment(df_train, target='y',
                             max_trials=50,
                             log_level='info',
                             verbose=1, callbacks=[callback], search_callbacks=[model_callback])

estimator = experiment.run(max_trials=20)

assert estimator
X_test = df_test
y_test = X_test.pop('y')
pred = estimator.predict(X_test)
result = evaluate(estimator, X_test, y_test, metrics=['auc', 'accuracy'])
print(f'final result:{result}')
