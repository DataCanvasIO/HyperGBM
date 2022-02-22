## GPU加速

为了使用 NVIDIA GPU 加速 HyperGBM 训练过程，您需要安装 NVIDIA RAPIDS 中的 cuML 和 cuDF , 同时也需要安装支持 GPU 的LightGBM、XGBoost、CatBoost, 关于软件安装的信息请参考 [安装HyperGBM](installation.html)。 

### 加速实验

为了在HyperGBM训练中启用GPU加速，您只需要将数据加载为`cudf`的DataFrame，然后将将其作为参数`train_data`、`eval_data`、`test_data`传递给工具方法 `make_experiment`，该方法会自动检测数据类型并配置适合使用GPU训练的实验。

示例:

```python
import cudf

from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils


def train():
    train_data = cudf.from_pandas(dsutils.load_blood())

    experiment = make_experiment(train_data, target='Class')
    estimator = experiment.run()
    print(estimator)


if __name__ == '__main__':
    train()

```

输出:
```
LocalizablePipeline(steps=[('data_clean',
                            DataCleanStep(cv=True,
                                          name='data_cle...
                            CumlGreedyEnsemble(weight=[...]))])


```

需要注意的是，此时训练得到的estimator是一个 `LocalizablePipeline` 而不是常见的sklearn Pipeline。`LocalizablePipeline`支持使用cudf DataFrame作为输入数据进行 predict 和 predict_proba。同时在生产环境部署 `LocalizablePipeline` 时需要安装与训练环境相同的软件，包括cuML、cuDF等。

如果您希望在没有cuML、cuDF的环境中部署模型的话，可调用 `LocalizablePipeline` 的`as_local`方法将其转化为sklearn Pipeline，示例：

```python
import cudf

from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils


def train():
    train_data = cudf.from_pandas(dsutils.load_blood())

    experiment = make_experiment(train_data, target='Class')
    estimator = experiment.run()
    print(estimator)

    print('-' * 20)
    estimator = estimator.as_local()
    print('localized estimator:\n', estimator)


if __name__ == '__main__':
    train()

```

输出：

```
LocalizablePipeline(steps=[('data_clean',
                            DataCleanStep(cv=True,
                                          name='data_cle...
                            CumlGreedyEnsemble(weight=[...]))])
--------------------
localized estimator:
 Pipeline(steps=[('data_clean',
                 DataCleanStep(cv=True,
                               name='data_clean')),
                ('est...
                 GreedyEnsemble(weight=[...]))])


```



### 自定义搜索空间

在启用GPU加速时，自定义的搜索空间中所有Transformer和Estimator都要求能够同时支持pandas和cudf的DataFrame，您可以以hypergbm.cuml中的 `earch_space_general` 和 `CumlGeneralSearchSpaceGenerator` 为基础定义自己的搜索空间。

示例：

```python
import cudf

from hypergbm import make_experiment
from hypergbm.cuml import search_space_general
from hypernets.tabular.datasets import dsutils


def my_search_space():
    return search_space_general(n_estimators=100)


def train():
    train_data = cudf.from_pandas(dsutils.load_blood())

    experiment = make_experiment(train_data, target='Class', searcher='mcts', search_space=my_search_space)
    estimator = experiment.run()
    print(estimator)


if __name__ == '__main__':
    train()

```


