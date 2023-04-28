## Distributed training

### Quick Experiment

HyperGBM supports performing distributed training with Dask. Before training, the Dask collections should be deployed and `Client` object of Dask should be initialized. 


Suppose that your training data file is '/opt/data/my_data.csv', the following code shows how to load data for a single node:

```python
from dask import dataframe as dd
from dask.distributed import LocalCluster, Client

from hypergbm import make_experiment


def train():
    cluster = LocalCluster(processes=True)
    client = Client(cluster)

    train_data_file = '/opt/data/my_data.csv'
    train_data = dd.read_csv(train_data_file)
    train_data = train_data.persist()
    
    experiment = make_experiment(train_data, target='...')
    estimator = experiment.run()
    print(estimator)


if __name__ == '__main__':
    train()

```



We recommend splitting the data to multiple files and save them in a single location such as '/opt/data/my_data' for large-scale data to speed up the loading process. After doing this, one can create an experiment with these files:

```python
from dask import dataframe as dd
from dask.distributed import LocalCluster, Client

from hypergbm import make_experiment


def train():
    cluster = LocalCluster(processes=True)
    client = Client(cluster)

    train_data_files = '/opt/data/my_data/*.parquet'
    train_data = dd.read_parquet(train_data_files)
    train_data = train_data.persist()
    
    experiment = make_experiment(train_data, target='...')
    estimator = experiment.run()
    print(estimator)


if __name__ == '__main__':
    train()

```



Please also refer to the official documents of Dask [Create DataFrames](https://docs.dask.org/en/latest/dataframe-api.html#create-dataframes) for further details on how to use Dask DataFrame.



### Define Search Space

When running an experiment in the Dask environment, the Transformer and Estimator used in the search space need to support Dask data type. Users can define new search space based on the default search space of HyperGBM which supports Dask. 

An example code:


```python
from dask import dataframe as dd
from dask.distributed import LocalCluster, Client

from hypergbm import make_experiment
from hypergbm.dask import search_space_general
from hypernets.tabular.datasets import dsutils


def my_search_space():
    return search_space_general(n_estimators=100)


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


