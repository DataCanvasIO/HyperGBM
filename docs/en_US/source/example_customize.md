## Search Space

When not defined explicitly, `make_experiment` will use `search_space_general` as its search space, which is defined as follows

```python
search_space_general = GeneralSearchSpaceGenerator(n_estimators=200)
```



### Define Search Space


To use a specific search space, one can change the parameter `search_space` when calling `make_experiment`. Taking defining the `max_depth` as 20 for `xgboost` as an example:

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



If you want to use searchable parameters, we recommend doing this by defining a subclass of `GeneralSearchSpaceGenerator`. For example, if we want the algorithm to search among 3 choices of the `max_depth` for `xgboost`:

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



### Support Machine Learning Models
HyperGBM has already supported XGBoost, LightGBM, CatBoost, and HistGradientBoosting. They are taken as components of the Search Space to be searched for training a model. Supporting other machine learning algorithms can be done by following 3 steps:

* encapsulating your algorithms as a subclass of HyperEstimator
* Add the encapsulated algorithms to the search sapce and define the search parameters
* Use your Search Space in `make_experiment`

Please see the following example:
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
