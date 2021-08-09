## 自定义SearchSpace

`make_experiment`缺省使用的搜索空间是`search_space_general`，定义如下：

```python
search_space_general = GeneralSearchSpaceGenerator(n_estimators=200)
```


### 自定义搜索空间(SearchSpace)


在调用`make_experiment`时可通过参数`search_space`指定自定义的搜索空间。如：指定`xgboost`的`max_depth=20`：

```python
from hypergbm import make_experiment
from hypergbm.search_space import GeneralSearchSpaceGenerator

my_search_space = \
    GeneralSearchSpaceGenerator(n_estimators=200, xgb_init_kwargs={'max_depth': 20})

train_data = ...

experiment = make_experiment(train_data,
                             search_space=my_search_space,
                             ...)

```



如果您希望自定义的参数是可搜索的，推荐定义`GeneralSearchSpaceGenerator`的一个子类，比如：指定`xgboost`的`max_depth`在10、20、30之间搜索：

```python
from hypergbm import make_experiment
from hypergbm.search_space import GeneralSearchSpaceGenerator
from hypernets.core.search_space import Choice

class MySearchSpace(GeneralSearchSpaceGenerator):
    @property
    def default_xgb_init_kwargs(self):
        return { **super().default_xgb_init_kwargs,
                'max_depth': Choice([10, 20 ,30]),
        }

my_search_space = MySearchSpace()
train_data = ...

experiment = make_experiment(train_data, 
                             search_space=my_search_space,
                             ...)

```



### 自定义建模算法
 
HyperGBM实现了对XGBoost、LightGBM、CatBoost和HistGridientBoosting的支持，并作为SearchSpace的一部分在建模优化时进行搜索。如果您需要增加对其他算法的支持，则需要：
* 将您选择的算法封装为HyperEstimator的子类
* 将封装后的算法增加到SearchSpace中，并定义搜索参数
* 在make_experiment中年使用您自定义的SearchSpace

示例：

```python
from sklearn import svm

from hypergbm import make_experiment
from hypergbm.estimators import HyperEstimator
from hypergbm.search_space import GeneralSearchSpaceGenerator
from hypernets.core.search_space import Choice, Int, Real
from hypernets.tabular.datasets import dsutils


class SVMEstimator(HyperEstimator):
    def __init__(self, fit_kwargs, C=1.0, kernel='rbf', gamma='auto', degree=3, random_state=666, probability=True,
                 decision_function_shape=None, space=None, name=None, **kwargs):
        if C is not None:
            kwargs['C'] = C
        if kernel is not None:
            kwargs['kernel'] = kernel
        if gamma is not None:
            kwargs['gamma'] = gamma
        if degree is not None:
            kwargs['degree'] = degree
        if random_state is not None:
            kwargs['random_state'] = random_state
        if decision_function_shape is not None:
            kwargs['decision_function_shape'] = decision_function_shape
        kwargs['probability'] = probability
        HyperEstimator.__init__(self, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, task, kwargs):
        if task == 'regression':
            hsvm = SVMRegressorWrapper(**kwargs)
        else:
            hsvm = SVMClassifierWrapper(**kwargs)
        hsvm.__dict__['task'] = task
        return hsvm


class SVMClassifierWrapper(svm.SVC):
    def fit(self, X, y=None, **kwargs):
        return super().fit(X, y)


class SVMRegressorWrapper(svm.SVC):
    def fit(self, X, y=None, **kwargs):
        return super().fit(X, y)


class GeneralSearchSpaceGeneratorPlusSVM(GeneralSearchSpaceGenerator):
    def __init__(self, enable_svm=True, **kwargs):
        super(GeneralSearchSpaceGeneratorPlusSVM, self).__init__(**kwargs)
        self.enable_svm = enable_svm

    @property
    def default_svm_init_kwargs(self):
        return {
            'C': Real(0.1, 5, 0.1),
            'kernel': Choice(['rbf', 'poly', 'sigmoid']),
            'degree': Int(1, 5),
            'gamma': Real(0.0001, 5, 0.0002)
        }

    @property
    def default_svm_fit_kwargs(self):
        return {}

    @property
    def estimators(self):
        r = super().estimators
        if self.enable_svm:
            r['svm'] = (SVMEstimator, self.default_svm_init_kwargs, self.default_svm_fit_kwargs)
        return r


my_search_space = GeneralSearchSpaceGeneratorPlusSVM()

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class',
                             search_space=my_search_space)
estimator = experiment.run()
print(estimator)

```

