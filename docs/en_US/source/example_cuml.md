## GPU acceleration

To accelerate HyperGBM with NVIDIA GPU devices, you must install NVIDIA RAPIDS cuML and cuDF, and enable GPU support of all estimators, see [Installation Guide](installation.html) for more details. 

### Accelerate the experiment

To accelerate the experiment with GPU, you should load dataset as `cudf.DataFrame` and use them as `train_data`/`eval_data`/`test_data` arguments to call the utility `make_experiment`, the utility will set experiment to run on GPU device.

Example:

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

Outputs:
```
LocalizablePipeline(steps=[('data_clean',
                            DataCleanStep(cv=True,
                                          name='data_cle...
                            CumlGreedyEnsemble(weight=[...]))])


```

It should be noted that the trained estimator is a `LocalizablePipeline` rather than a sklearn Pipeline. The `Localizablepipeline` accepts cudf DataFrame as input X for prediction. When you deploy the `LocalizablePipeline` in a production environment, you need to install the same software as the training environment, including cuML, cuDF, etc. 

If you want to deploy the trained estimator in an environment without cuML and cuDF, please call the `estimator.as_local()` to converts it into a sklearn Pipeline. An example:

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

Outputs：

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



### Customize Search Space

When running an experiment on GPU, all Transformers and Estimators used in the search space need to support both pandas/numpy data types and cuDF/cupy data types. Users can define new search space based on the `search_space_general` and `CumlGeneralSearchSpaceGenerator` from `hypergbm.cuml`. 

An example code:


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


