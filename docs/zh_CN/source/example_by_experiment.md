## 实验示例

### 基础使用

本节通过示例讲解如何使用实验进行模型训练，示例中使用数据集`blood`，该数据集中的目标列`Class`，数据如下：
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
本节示例从`tabular_toolbox`中加载该数据集。


#### 以默认配置创建并运行实验

利用工具`make_experiment`可快速创建一个可运行的实验对象，执行该实验对象的`run`方法即可开始训练并得到模型。使用该工具时只有实验数据`train_data`是必须的，其它都是可选项。数据的目标列如果不是`y`的话，需要通过参数`target`设置。

示例代码：
```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class')
estimator = experiment.run()
print(estimator)

```

输出：
```
Pipeline(steps=[('data_clean',
                 DataCleanStep(...),
                ('estimator',
                 GreedyEnsemble(...)])

Process finished with exit code 0

```

可以看出，训练得到的是一个Pipeline，最终模型是由多个模型构成的融合模型。



如果您的训练数据是cvs或parquet格式，而且数据文件的扩展名是“.csv”或“.parquet”的话，可以直接使用文件路径创建实验，make_experiment回自动将数据加载为data_frame，如：

```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = '/path/to/mydata.csv'
experiment = make_experiment(train_data, target='my_target')
estimator = experiment.run()
print(estimator)

```



#### 交叉验证

未来获取较好的模型效果，`make_experiment`创建实验时默认开启了交叉验证的特性 ，可通过参数`cv`指定是否启用交叉验证。当`cv`设置为`False`时表示禁用交叉验证并使用经典的train_test_split方式进行模型训练；当`cv`设置为`True`时表示开启交叉验证，折数可通过参数`num_folds`设置（默认：3）；


启用交叉验证的示例代码：
```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', cv=True, num_folds=5)
estimator = experiment.run()
print(estimator)

```

#### 指定验证数据集(eval_data)

在禁用交叉验证时，模型训练处除了需要训练数据集，还需要评估数据集，您可在`make_experiment`时通过eval_data指定评估集，如：

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

在禁用交叉验证时，如果您未指定`eval_data`，实验对象将从`train_data`中拆分部分数据作为评估集，拆分大小可通过`eval_size`设置，如：

```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', cv=False, eval_size=0.2)
estimator = experiment.run()
print(estimator)

```



#### 指定模型的评价指标

使用`make_experiment`创建实验的默认的模型评价指标是`accuracy`，可通过参数`reward_metric`指定，如：

```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', reward_metric='auc')
estimator = experiment.run()
print(estimator)

```



#### 设置搜索次数和早停（Early Stopping）策略

使用`make_experiment`时，可通过参数`max_trials`设置最大搜索次数，通过参数`early_stopping_round`、`early_stopping_time_limit`、`early_stopping_reward`设置实验的早停策略。

将搜索时间设置为最多3小时的示例代码：
```pytyon
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', max_trials=30, early_stopping_time_limit=3600 * 3)
estimator = experiment.run()
print(estimator)

```

#### 漂移检测

数据漂移是建模过程中的一个主要挑战。当数据的分布随着时间在不断的发现变化时，模型的表现会越来越差，我们在HyperGBM中引入了对抗验证的方法专门处理数据漂移问题。这个方法会自动的检测是否发生漂移，并且找出发生漂移的特征并删除他们，以保证模型在真实数据上保持良好的状态。

如果要开启飘逸检测，使用`make_experiment`创建实验时需要设置`drift_detection=True`并提供测试集`test_data`。需要注意的是，测试集不应该包含目标列。

启用漂移检测的示例代码：
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

#### 共线性检测

有时训练数据中会出现一些相关度很高的特征，这些并没有提供太多的信息量，相反，数据集拥有更多的特征意味着更容易收到噪声的影响，更容易收到特征偏移的影响等等。
HyperGBM中提供了删除发生共线性的特征的能力， 在通过`make_experiment`创建实验时设置`collinearity_detection=True`即可。

启用共线性检测的示例代码：
```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', `collinearity_detection=True)
estimator = experiment.run()
print(estimator)

```

#### 伪标签技术

伪标签是一种半监督学习技术，将测试集中未观测标签列的特征数据通过一阶段训练的模型预测标签后，将置信度高于一定阈值的样本添加到训练数据中重新训练模型，有时候可以进一步提升模型在新数据上的拟合效果。在通过`make_experiment`创建实验时设置`pseudo_labeling=True`可开启伪标签训练，与之匹配的配置项包括：

* pseudo_labeling_proba_threshold：float(default=0.8), 伪标签的置信度阈值。
* pseudo_labeling_resplit：bool(default=False), 添加新的伪标签数据后是否重新分割训练集和评估集. 如果为False, 直接把所有伪标签数据添加到训练集中重新训练模型，否则把训练集、评估集及伪标签数据合并后重新分割。

启用伪标签技术的示例代码：
```python
train_data=...
experiment = make_experiment(train_data, pseudo_labeling=True, ...)

```

#### 二阶段特征筛选

在通过`make_experiment`创建实验时设置`feature_reselection=True`可开启二阶段特征筛选，与之匹配的配置项包括：
* feature_reselection_estimator_size：int, (default=10), 用于评估特征重要性的estimator数量（在一阶段搜索中表现最好的n个模型。
* feature_reselection_threshold：float, (default=1e-5), 二阶搜索是特征选择重要性阈值，重要性低于该阈值的特征会被删除。

启用伪标签技术的示例代码：
```python
train_data=...
experiment = make_experiment(train_data, feature_reselection=True, ...)

```



#### 模型融合

未来获取较好的模型效果，`make_experiment`创建实验时默认开启了模型融合的特性，并使用效果最好的20个模型进行融合，可通过参数`ensemble_size`指定参与融合的模型的数量，当`ensemble_size`设置为`0`时则表示禁用模型融合。

调整参与融合的模型数量的示例代码：
```python
train_data = ...
experiment = make_experiment(train_data, ensemble_size=10, ...)

```



#### 调整实验运行的日志级别

如果希望对在训练过程中看到使用进度信息的话，可通过log_level指定日志级别，可以就`str`或`int`。关于日志级别的详细定义可参考python的`logging`包。 另外，如果将`verbose`设置为`1`的话，可以得到更详细的信息。


将日志级别设置为`INFO`的示例代码如下：
```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', log_level='INFO', verbose=1)
estimator = experiment.run()
print(estimator)

```

输出：
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



### 高级应用

#### 自定义搜索空间(SearchSpace)
`make_experiment`缺省使用的搜索空间是`search_space_general`，定义如下：

```
search_space_general = GeneralSearchSpaceGenerator(n_estimators=200)
```


在调用`make_experiment`时可通过参数`search_space`指定自定义的搜索空间。如：指定`xgboost`的`max_depth=20`：

```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils
from hypergbm.search_space import GeneralSearchSpaceGenerator

my_search_space = \
    GeneralSearchSpaceGenerator(n_estimators=200, xgb_init_kwargs={'max_depth': 20})

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', search_space=my_search_space)
estimator = experiment.run()
print(estimator)
```



如果您希望自定义的参数是可搜索的，则需要定义`GeneralSearchSpaceGenerator`的一个子类，比如：指定`xgboost`的`max_depth`在10、20、30之间搜索：

```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils
from hypergbm.search_space import GeneralSearchSpaceGenerator
from hypernets.core.search_space import Choice

class MySearchSpace(GeneralSearchSpaceGenerator):
    @property
    def default_xgb_init_kwargs(self):
        return { **super().default_xgb_init_kwargs,
                'max_depth': Choice([10, 20 ,30]),
        }

my_search_space = MySearchSpace()
train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', search_space=my_search_space)
estimator = experiment.run()
print(estimator)
```



#### 自定义搜索算法（Searcher）

HyperGBM内置的搜索算法包括：EvolutionSearcher（默认）、MCTSSearcher、RandomSearch，在`make_experiment`时可通过参数`searcher`指定，可以指定搜索算法的类名(class)、搜索算法的名称（str）。


示例代码：
```python
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', searcher='random')
estimator = experiment.run()
print(estimator)

```



您也可以自己创建searcher对象，然后用所创建的对象创建实验，如：

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



#### 自定义CompeteExperiment

如果你希望控制更多实验的细节，您可以直接创建CompeteExperiment对象并运行。



示例代码：

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



### 分布式训练



#### 快速实验

HyperGBM支持使用Dask进行分布式训练，在运行实验之前您需要部署Dask集群并初始化Dask客户端`Client`对象；如果您的训练数据是csv或parquet格式，而且数据文件的扩展名是“.csv”或“.parquet”的话，可以直接使用文件路径创建实验，make_experiment在检测到Dask环境是会自动将数据加载为Dask的DataFrame对象并进行搜索和训练。

示例代码（以单节点为例）：

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



对于大规模数据，为了加速数据的加载过程，建议您将数据拆分为多个文件并保存在一个目录下，然后使用目录下的文件创建实验，如：

```python
from dask.distributed import LocalCluster, Client

from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils


def train():
    cluster = LocalCluster(processes=True)
    client = Client(cluster)

    train_data = '/opt/data/my_data/*.parquet'

    experiment = make_experiment(train_data, target='...')
    estimator = experiment.run()
    print(estimator)


if __name__ == '__main__':
    train()

```



您也可以直接使用`dask.dataframe`提供的方法将数据加载为Dask DataFrame然后创建实验，如：

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

关于使用Dask DataFrame的更多信息请参考Dask官方文档中的 [Create DataFrames](https://docs.dask.org/en/latest/dataframe-api.html#create-dataframes)



#### 自定义SearchSpace

在Dask环境下运行实验时，搜索空间中使用的Transformer和Estimator必须都支持Dask数据类型，您可以参考或基于HyperGBM内置的支持Dask的搜索空间定义自己的搜索空间。


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


