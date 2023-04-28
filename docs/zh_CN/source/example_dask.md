## 分布式训练



### 快速实验

HyperGBM支持使用Dask进行分布式训练，在运行实验之前您需要部署Dask集群并初始化Dask客户端`Client`对象； 然后使用`dask`提供的方法将数据加载为Dask DataFrame然后创建实验。

示例代码（以单节点为例，假设您的训练数据文件是‘/opt/data/my_data.csv’）：

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



对于大规模数据，为了加速数据的加载过程，建议您将数据拆分为多个文件并保存在一个目录下（如：/opt/data/my_data/），然后使用目录下的文件创建实验，如：

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


关于使用Dask DataFrame的更多信息请参考Dask官方文档中的 [Create DataFrames](https://docs.dask.org/en/latest/dataframe-api.html#create-dataframes)



### 自定义SearchSpace

在Dask环境下运行实验时，搜索空间中使用的Transformer和Estimator必须都支持Dask数据类型，您可以参考或基于HyperGBM内置的支持Dask的搜索空间定义自己的搜索空间。


```python
from dask import dataframe as dd
from dask.distributed import LocalCluster, Client

from hypergbm import make_experiment
from hypergbm.dask.search_space import search_space_general
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


