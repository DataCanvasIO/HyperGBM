## Quick Start

The purpose of this guide is to illustrate some of the main features that hypergbm provides.
It assumes a basic working knowledge of machine learning practices (dataset, model fitting, predicting, cross-validation, etc.). 
Please refer to [installation](installation.md) instructions for installing hypergbm; You can use hypergbm through the python API and command line tools

This section will show you how to train a binary model using hypergbm.

You can use `hypernets.tabular` utility to read [Bank Marketing](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing) dataset： 
```pydocstring
>>> from hypernets.tabular.datasets import dsutils
>>> df = dsutils.load_bank()
>>> df[:3]
   id  age         job  marital  education default  balance housing loan   contact  day month  duration  campaign  pdays  previous poutcome   y
0   0   30  unemployed  married    primary      no     1787      no   no  cellular   19   oct        79         1     -1         0  unknown  no
1   1   33    services  married  secondary      no     4789     yes  yes  cellular   11   may       220         1    339         4  failure  no
2   2   35  management   single   tertiary      no     1350     yes   no  cellular   16   apr       185         1    330         1  failure  no
```

### Training with `make_experiment`

Firstly,  we load and split the data into training set and test set to train and evaluate the model: 
```pydocstring
>>> from sklearn.model_selection import train_test_split
>>> from hypernets.tabular.datasets import dsutils
>>> df = dsutils.load_bank()
>>> train_data,test_data = train_test_split(df, test_size=0.3, random_state=9527)
```

Then, create experiment instance and run it:
```pydocstring
>>> from hypergbm import make_experiment
>>> experiment=make_experiment(train_data,target='y')
>>> pipeline=experiment.run(max_trials=10)
>>> pipeline
Pipeline(steps=[('data_clean',
                 DataCleanStep(cv=True, data_cleaner_args={}, name='data_clean', random_state=9527)),
                ('estimator',
                 GreedyEnsemble(weight=[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]))])
```
The `pipeline` is trained model。

We can use sklearn metrics to evaluate the trained model：
```pydocstring
>>> from sklearn import metrics
>>> X_test=test_data.copy()
>>> y_test=X_test.pop('y')
>>> y_proba = pipeline.predict_proba(X_test)
>>> metrics.roc_auc_score(y_test, y_proba[:, 1])
0.9659882829799505
```



### Training with CompeteExperiment

Load and  split the data into training set and test set to train and evaluate the model: 
```pydocstring
>>> from sklearn.model_selection import train_test_split
>>> from hypernets.tabular.datasets import dsutils
>>> df = dsutils.load_bank()
>>> y = df.pop('y')  # target col is "y"
>>> X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=9527)
```

Hypergbm provides a variety of search strategies.
Here, the random search strategy is used to train in the built-in search space:

```pydocstring
>>> from hypernets.searchers import RandomSearcher
>>> from hypernets.core import OptimizeDirection
>>> from hypergbm.search_space import search_space_general
>>> rs = RandomSearcher(space_fn=search_space_general,
...                     optimize_direction=OptimizeDirection.Maximize)
>>> rs
<hypernets.searchers.random_searcher.RandomSearcher object at 0x10e5b9850>
```

Parameters `space_fn` is used to specify the search space;
Meric AUC is used here, and set `optimize_direction=OptimizeDirection.Maximize` means the larger the value of the metric, the better .

Then use the Experiment API to train the model:
```
>>> from hypergbm import HyperGBM, CompeteExperiment
>>> hk = HyperGBM(rs, reward_metric='auc', cache_dir=f'hypergbm_cache', callbacks=[])
>>> experiment = CompeteExperiment(hk, X_train, y_train, X_test=X_test)
19:19:31 I hypergbm.experiment.py 714 - create experiment with ['data_clean', 'drift_detected', 'base_search_and_train']
>>> pipeline = experiment.run(use_cache=True, max_trials=2)
...
   Trial No.    Reward   Elapsed                      Space Vector
0          1  0.994731  8.490173             [2, 0, 1, 2, 0, 0, 0]
1          2  0.983054  4.980630  [1, 2, 1, 2, 215, 3, 0, 0, 4, 3]
>>> pipeline
Pipeline(steps=[('data_clean',
                 DataCleanStep(cv=True, data_cleaner_args={}, name='data_clean', random_state=9527)),
                ('estimator', GreedyEnsemble(weight=[1. 0.]))])


```

After the training experiment, let's evaluate the model：
```pydocstring
>>> from sklearn import metrics
>>> y_proba = pipeline.predict_proba(X_test)
>>> metrics.roc_auc_score(y_test, y_proba[:, 1])
0.9956872713648863
```

### Training with command-line tools

HyperGBM also provides command-line tools to train model and predict data, view the help doc:
```
hypergm -h

usage: hypergbm [-h] --train_file TRAIN_FILE [--eval_file EVAL_FILE]
                [--eval_size EVAL_SIZE] [--test_file TEST_FILE] --target
                TARGET [--pos_label POS_LABEL] [--max_trials MAX_TRIALS]
                [--model_output MODEL_OUTPUT]
                [--prediction_output PREDICTION_OUTPUT] [--searcher SEARCHER]
...
```

Similarly, taking the training Bank Marketing as an example, we first split the data set into training set and test set and generate the CSV file for command-line tools:
```pydocstring
>>> from hypernets.tabular.datasets import dsutils
>>> from sklearn.model_selection import train_test_split
>>> df = dsutils.load_bank()
>>> df_train, df_test = train_test_split(df, test_size=0.3, random_state=9527)
>>> df_train.to_csv('bank_train.csv', index=None)
>>> df_test.to_csv('bank_test.csv', index=None)
```

The generated CSV files is used as the training command parameters then execute the command:：
```shell script
hypergbm --train_file=bank_train.csv --test_file=bank_test.csv --target=y --pos_label=yes --model_output=model.pkl prediction_output=bank_predict.csv

...
   Trial No.    Reward    Elapsed                       Space Vector
0         10  1.000000  64.206514  [0, 0, 1, 3, 2, 1, 2, 1, 2, 2, 3]
1          7  0.999990   2.433192   [1, 1, 1, 2, 215, 0, 2, 3, 0, 4]
2          4  0.999950  37.057761  [0, 3, 1, 0, 2, 1, 3, 1, 3, 4, 3]
3          9  0.967292   9.977973   [1, 0, 1, 1, 485, 2, 2, 5, 3, 0]
4          1  0.965844   4.304114    [1, 2, 1, 1, 60, 2, 2, 5, 0, 1]
```

After the training, the model will be persisted to file `model.pkl` and the prediction results will be saved to `bank_predict.csv`.
