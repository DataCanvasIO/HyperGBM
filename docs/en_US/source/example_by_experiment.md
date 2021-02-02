### Experiment Examples 

#### Basic Usages

In this chapter we'll show how to train models with HyperGBM experiment,  we'll use the `blood` dataset in the following examples，`Class` is the target feature.

```csv
Recency,Frequency,Monetary,Time,Class
2,50,12500,98,1
0,13,3250,28,1
1,16,4000,35,1
2,20,5000,45,1
1,24,6000,77,0
4,4,1000,4,0

...

```



##### Use experiment with default settings

User can create experiment instance with the python tool `make_experiment` and run it quickly。`train_data` is the only required parameter, all others are optional.   The `target` is also required if your target feature name isn't `y`。

Codes:
```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class')
estimator = experiment.run()
print(estimator)

```

Outputs：
```
Pipeline(steps=[('data_clean',
                 DataCleanStep(...),
                ('estimator',
                 GreedyEnsemble(...)])

Process finished with exit code 0

```

As the console output, the trained model is a pipeline object，the estimator is ensembled by several other models。



If your training data files are  .csv or .parquet files，user can  call `make_experiment` with the file path directly，like the following：

```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = '/path/to/mydata.csv'
experiment = make_experiment(train_data, target='my_target')
estimator = experiment.run()
print(estimator)

```



##### Cross Validation

`make_experiment` enable cross validation as default,  user can disable it by set `cv= False`.  Use can change cross fold number with `num_folds`,  just like this:

```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', cv=True, num_folds=5)
estimator = experiment.run()
print(estimator)

```



##### Setup evaluate data (eval_data)

Experiment split evaluate data from `train_data` by default if cross validation is disabled,  user can customize it with `eval_data` like this:

```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils
from sklearn.model_selection import train_test_split

train_data = dsutils.load_blood()
train_data,eval_data=train_test_split(train_data,test_size=0.3)
experiment = make_experiment(train_data, target='Class', eval_data=eval_data, cv=False)
estimator = experiment.run()
print(estimator)

```



If `eval_data` is None and `cv` is False,  the experiment will split evaluation data from `train_data`,  user can change evaluation data size with `eval_size`,  like this:

```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', cv=False, eval_size=0.2)
estimator = experiment.run()
print(estimator)

```



##### Setup search reward metric

The default search reward metric is `accuracy`，user can change it with `reward_metric`,  like this:

```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', reward_metric='auc')
estimator = experiment.run()
print(estimator)

```



##### Change search trial number and setup early stopping

User can limit search trial number with `max_trials`，and setup search early stopping with `early_stopping_round`, `early_stopping_time_limit`, `early_stopping_reward`. like this:

```pytyon
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', max_trials=30, early_stopping_time_limit=3600 * 3)
estimator = experiment.run()
print(estimator)

```



##### Drift detection

To enable the feature drift detection, set `drift_detection=True`, and set `test_data` with the testing data, like this：

```python
from io import StringIO
import pandas as pd
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

test_data = """
Recency,Frequency,Monetary,Time
2,10,2500,64
4,5,1250,23
4,9,2250,46
4,5,1250,23
4,8,2000,40
2,12,3000,82
11,24,6000,64
2,7,1750,46
4,11,2750,61
1,7,1750,57
2,11,2750,79
2,3,750,16
4,5,1250,26
2,6,1500,41
"""

train_data = dsutils.load_blood()
test_df = pd.read_csv(StringIO(test_data))
experiment = make_experiment(train_data, test_data=test_df, target='Class', drift_detection=True)
estimator = experiment.run()
print(estimator)


```



##### Multicollinearity detection

To enable multicollinearity detection, set `collinearity_detection=True`, like this:

```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', `collinearity_detection=True)
estimator = experiment.run()
print(estimator)

```



##### Pseudo labeling

To enable pseudo labeling with two stage searching,  set `pseudo_labeling=True`, like this:
```python
train_data=...
experiment = make_experiment(train_data, pseudo_labeling=True, ...)

```



##### Permutation importance feature selection

To enable feature selection by permutation importance with two stage searching,  set `feature_reselection=True`, like this:
```python
train_data=...
experiment = make_experiment(train_data, feature_reselection=True, ...)

```



##### Ensemble

To change estimator number for ensemble, set `ensemble_size` to expected number.  Or set `ensemble_size=0`  to disable ensemble.

```python
train_data = ...
experiment = make_experiment(train_data, ensemble_size=10, ...)

```



##### Logging settings

To change logging level, set `log_level` with log level defined in  python logging utility.

```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', log_level='INFO', verbose=1)
estimator = experiment.run()
print(estimator)

```

Outputs：
```
14:24:33 I tabular_toolbox.u._common.py 30 - 2 class detected, {0, 1}, so inferred as a [binary classification] task
14:24:33 I hypergbm.experiment.py 699 - create experiment with ['data_clean', 'drift_detection', 'space_search', 'final_ensemble']
14:24:33 I hypergbm.experiment.py 1262 - make_experiment with train data:(748, 4), test data:None, eval data:None, target:Class
14:24:33 I hypergbm.experiment.py 716 - fit_transform data_clean
14:24:33 I hypergbm.experiment.py 716 - fit_transform drift_detection
14:24:33 I hypergbm.experiment.py 716 - fit_transform space_search
14:24:33 I hypernets.c.meta_learner.py 22 - Initialize Meta Learner: dataset_id:7123e0d8c8bbbac8797ed9e42352dc59
14:24:33 I hypernets.c.callbacks.py 192 - 
Trial No:1
--------------------------------------------------------------
(0) estimator_options.hp_or:                                0
(1) numeric_imputer_0.strategy:                 most_frequent
(2) numeric_scaler_optional_0.hp_opt:                    True


...

14:24:35 I hypergbm.experiment.py 716 - fit_transform final_ensemble
14:24:35 I hypergbm.experiment.py 737 - trained experiment pipeline: ['data_clean', 'estimator']
Pipeline(steps=[('data_clean',
                 DataCleanStep(...),
                ('estimator',
                 GreedyEnsemble(...)

Process finished with exit code 0

```



#### Advanced Usages



##### Customize Searcher and Search Space

User can customize searcher and search space with `searcher` and `search_space`, like this:

```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils
from hypergbm.search_space import search_space_general


def my_search_space():
    return search_space_general(n_esitimators=100)


train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', searcher='random', search_space=my_search_space)
estimator = experiment.run()
print(estimator)

```



Or like this：

```python
from hypergbm import make_experiment
from hypergbm.search_space import search_space_general
from hypernets.searchers import MCTSSearcher
from tabular_toolbox.datasets import dsutils

my_searcher = MCTSSearcher(lambda: search_space_general(n_esitimators=100),
                           max_node_space=20,
                           optimize_direction='max')

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', searcher=my_searcher)
estimator = experiment.run()
print(estimator)

```



##### Use CompeteExperiment

Use can create  experiment with class `CompeteExperiment` for more details.

```python
from hypergbm import HyperGBM, CompeteExperiment
from hypergbm.search_space import search_space_general
from hypernets.core.callbacks import EarlyStoppingCallback, SummaryCallback
from hypernets.searchers import EvolutionSearcher
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()


def my_search_space():
    return search_space_general(early_stopping_rounds=10, verbose=0, cat_pipeline_mode='complex')


searcher = EvolutionSearcher(my_search_space,
                             optimize_direction='max', population_size=30, sample_size=10,
                             regularized=True, candidates_size=10)

es = EarlyStoppingCallback(time_limit=3600 * 3, mode='max')
hm = HyperGBM(searcher, reward_metric='auc', cache_dir=f'hypergbm_cache', clear_cache=True,
              callbacks=[es, SummaryCallback()])

X = train_data
y = train_data.pop('Class')
experiment = CompeteExperiment(hm, X, y, eval_size=0.2,
                               cv=True, pseudo_labeling=False,
                               max_trials=20, use_cache=True)
estimator = experiment.run()
print(estimator)

```



#### Distribution with Dask



##### Quick Start

To run  HyperGBM experiment with Dask cluster, use need to setup the  default Dask client before call `make_experiment`, like this:

```python
from dask.distributed import LocalCluster, Client

from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils


def train():
    cluster = LocalCluster(processes=True)
    client = Client(cluster)

    train_data = '/opt/data/my_data.csv'

    experiment = make_experiment(train_data, target='...')
    estimator = experiment.run()
    print(estimator)


if __name__ == '__main__':
    train()

```



User can also use `dask.dataframe` load training data set Dask DataFrame to create experiment:

```python
from dask import dataframe as dd
from dask.distributed import LocalCluster, Client

from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils


def train():
    cluster = LocalCluster(processes=False)
    client = Client(cluster)

    train_data = dd.from_pandas(dsutils.load_blood(), npartitions=1)

    experiment = make_experiment(train_data, target='Class')
    estimator = experiment.run()
    print(estimator)


if __name__ == '__main__':
    train()


```

Reference Dask  [Create DataFrames](https://docs.dask.org/en/latest/dataframe-api.html#create-dataframes) for more details



##### Customize Search Space

To run experiment with Dask cluster, all transformers and estimators must support Dask objects, reference `hypergbm.dask.search_space.search_space_general`  for more details to customize search space pls。


```python
from dask import dataframe as dd
from dask.distributed import LocalCluster, Client

from hypergbm import make_experiment
from hypergbm.dask.search_space import search_space_general
from tabular_toolbox.datasets import dsutils


def my_search_space():
    return search_space_general(n_esitimators=100)


def train():
    cluster = LocalCluster(processes=False)
    client = Client(cluster)

    train_data = dd.from_pandas(dsutils.load_blood(), npartitions=1)

    experiment = make_experiment(train_data, target='Class', searcher='mcts', search_space=my_search_space)
    estimator = experiment.run()
    print(estimator)


if __name__ == '__main__':
    train()


```


