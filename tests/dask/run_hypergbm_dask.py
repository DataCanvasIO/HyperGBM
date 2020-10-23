# -*- coding:utf-8 -*-
"""

"""
import dask.dataframe as dd
from dask.distributed import Client
from dask_ml.model_selection import train_test_split
from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers.random_searcher import RandomSearcher
from sklearn import preprocessing as sk_pre

from hypergbm.dask.dask_ops import get_space_num_cat_pipeline_complex
from hypergbm.datasets import dsutils
from hypergbm.hyper_gbm import HyperGBM
from tests import test_output_dir

# client = Client("tcp://127.0.0.1:64958")
client = Client(processes=False, threads_per_worker=1, n_workers=1, memory_limit='4GB')
print(client)


rs = RandomSearcher(get_space_num_cat_pipeline_complex, optimize_direction=OptimizeDirection.Maximize)
hk = HyperGBM(rs, task='classification', reward_metric='accuracy',
              cache_dir=f'{test_output_dir}/hypergbm_cache',
              callbacks=[SummaryCallback(), FileLoggingCallback(rs, output_dir=f'{test_output_dir}/hyn_logs')])

df = dsutils.load_bank()
# df = df.sample(n=10000)
df.drop(['id'], axis=1, inplace=True)

le = sk_pre.LabelEncoder()
df['y'] = le.fit_transform(df['y'])

df = dd.from_pandas(df, npartitions=2)
# object_columns = [i for i, v in df.dtypes.items() if v == 'object']
# for c in object_columns:
#     df[c] = df[c].astype('category')
# df = df.categorize(object_columns)


X_train, X_test = train_test_split(df, test_size=0.8, random_state=42)
y_train = X_train.pop('y')
y_test = X_test.pop('y')

hk.search(X_train, y_train, X_test, y_test, max_trails=50)
assert hk.best_model
print('-' * 30)

best_trial = hk.get_best_trail()
print(f'best_train:{best_trial}')
estimator = hk.final_train(best_trial.space_sample, X_train, y_train)
score = estimator.predict(X_test)
result = estimator.evaluate(X_test, y_test, metrics=['accuracy', 'auc', 'logloss'])
print(f'final result:{result}')
