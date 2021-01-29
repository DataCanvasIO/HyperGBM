## Using HyperGBM

#### Basic examples
##### Cross-validation

HyperGBM supports cross-validation to evaluate the model, specify `cv=True` to enable it and param `num_fold` used to set folds: 
```python
...
hk = HyperGBM(rs, task='multiclass', reward_metric='accuracy', callbacks=[])
hk.search(X_train, y_train, X_eval=None, y_eval=None, cv=True, num_folds=3)  # 3 folds
...
```

Evaluation data should be a fold of  `X_train` and `y_train`, so set `X_eval=None` and `y_eval=None`.
Here is an example :
```pydocstring
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> 
>>> X, y = load_iris(return_X_y=True, as_frame=True)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> 
>>> from hypergbm.search_space import search_space_general
>>> from hypergbm import HyperGBM
>>> from hypernets.searchers import MCTSSearcher
>>> 
>>> rs = MCTSSearcher(search_space_general, max_node_space=10, optimize_direction='max')
>>> hk = HyperGBM(rs, task='multiclass', reward_metric='accuracy', callbacks=[])
>>> hk.search(X_train, y_train, X_eval=None, y_eval=None, cv=True, num_folds=3)  # using Cross Validation
   Trial No.    Reward   Elapsed                       Space Vector
0          4  0.941667  0.331012   [1, 3, 1, 1, 370, 3, 2, 3, 4, 0]
1          7  0.933333  0.290077  [0, 0, 1, 0, 3, 1, 1, 2, 1, 2, 3]
2          1  0.925000  0.472835     [0, 0, 0, 3, 0, 1, 0, 2, 0, 4]
3          3  0.925000  0.422006     [0, 1, 0, 1, 1, 1, 1, 0, 0, 1]
4          8  0.925000  0.228165     [0, 1, 0, 3, 2, 0, 2, 0, 2, 0]
>>> estimator = hk.load_estimator(hk.get_best_trial().model_file)
>>> 
>>> estimator.cv_gbm_models_
[LGBMClassifierWrapper(boosting_type='dart', learning_rate=0.5, max_depth=10,
                      n_estimators=200, num_leaves=370, reg_alpha=1,
                      reg_lambda=1), LGBMClassifierWrapper(boosting_type='dart', learning_rate=0.5, max_depth=10,
                      n_estimators=200, num_leaves=370, reg_alpha=1,
                      reg_lambda=1), LGBMClassifierWrapper(boosting_type='dart', learning_rate=0.5, max_depth=10,
                      n_estimators=200, num_leaves=370, reg_alpha=1,
                      reg_lambda=1)]
```

##### Search strategies

HyperGBM provides following search strategies(implementation class)：
  - Evolution search（hypernets.searchers.evolution_searcher.EvolutionSearcher）
  - Monte Carlo Tree Search（hypernets.searchers.mcts_searcher.MCTSSearcher）
  - Random search（hypernets.searchers.random_searcher.RandomSearcher）

Here is an example that using evolution search strategy:
```pydocstring
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> 
>>> X, y = load_iris(return_X_y=True, as_frame=True)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> 
>>> from hypergbm.search_space import search_space_general
>>> from hypergbm import HyperGBM
>>> from hypernets.searchers.evolution_searcher import EvolutionSearcher
>>> rs = EvolutionSearcher(search_space_general,  200, 100, optimize_direction='max')  # using EvolutionSearcher
>>> hk = HyperGBM(rs, task='multiclass', reward_metric='accuracy', callbacks=[])
>>> hk.search(X_train, y_train, X_eval=X_test, y_eval=y_test)
   Trial No.  Reward   Elapsed                      Space Vector
0          1     1.0  0.187103     [1, 2, 0, 1, 160, 3, 0, 1, 2]
1          2     1.0  0.358584             [2, 3, 1, 3, 2, 0, 0]
2          3     1.0  0.127980  [1, 1, 1, 0, 125, 0, 0, 3, 3, 0]
3          4     1.0  0.084272     [1, 1, 0, 2, 115, 1, 2, 3, 0]
4          7     1.0  0.152720     [1, 0, 0, 1, 215, 3, 3, 1, 2]
>>> estimator = hk.load_estimator(hk.get_best_trial().model_file)
>>> y_pred = estimator.predict(X_test)
>>> 
>>> from sklearn.metrics import accuracy_score
>>> accuracy_score(y_test, y_pred)
1.0
```

##### Early stopping

When the performance of the model can not be improved or meet certain conditions, the training can be terminated in advance to release computing resources, now supported strategies: 
* max_no_improvement_trials
* time_limit
* expected_reward

When multiple conditions are set, it will stop when any condition is reached first;
The early stop strategy is implemented through class `hypernets.core.callbacks.EarlyStoppingCallback`;

Here is an example that training stops searching when the reward reaches above 0.95: 
```pydocstring
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> from hypernets.core import EarlyStoppingCallback
>>> 
>>> X, y = load_iris(return_X_y=True, as_frame=True)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> from hypergbm.search_space import search_space_general
>>> from hypergbm import HyperGBM
>>> from hypernets.searchers.evolution_searcher import EvolutionSearcher
>>> rs = EvolutionSearcher(search_space_general,  200, 100, optimize_direction='max')
>>> es = EarlyStoppingCallback(expected_reward=0.95, mode='max')  # Parameter `mode` is the direction of parameter `expected_reward` optimization, the reward metric is accuracy, so set mode to `max`
>>> hk = HyperGBM(rs, task='multiclass', reward_metric='accuracy', callbacks=[es])
>>> hk.search(X_train, y_train, X_eval=X_test, y_eval=y_test)

Early stopping on trial : 1, best reward: None, best_trial: None
   Trial No.  Reward   Elapsed                       Space Vector
0          1     1.0  0.189758  [0, 1, 1, 3, 2, 1, 1, 2, 3, 0, 0]

```

#### Advanced examples
##### Pseudo label

HyperGBM is allowed to use test set training in a semi-supervised way to improve model performance, usage:
```pydocstring
...
experiment = CompeteExperiment(hk, X_train, y_train, X_test=X_test, callbacks=[], scorer=get_scorer('accuracy'),
                               pseudo_labeling=True,  # Enable pseudo label
                               pseudo_labeling_proba_threshold=0.9)
...
```

Here is an example：
```pydocstring
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> X, y = load_iris(return_X_y=True, as_frame=True)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> 
>>> from hypergbm.search_space import search_space_general
>>> from hypergbm import HyperGBM, CompeteExperiment
>>> from hypernets.searchers.evolution_searcher import EvolutionSearcher
>>> from sklearn.metrics import get_scorer
>>> 
>>> rs = EvolutionSearcher(search_space_general,  200, 100, optimize_direction='max')
>>> hk = HyperGBM(rs, task='multiclass', reward_metric='accuracy', callbacks=[])
>>> experiment = CompeteExperiment(hk, X_train, y_train, X_test=X_test, callbacks=[], scorer=get_scorer('accuracy'),
...                                pseudo_labeling=True,  # enable pseudo
...                                pseudo_labeling_proba_threshold=0.9)
>>> 
>>> pipeline = experiment.run(use_cache=True, max_trials=10)  # first stage train a model to label test dataset, the second stage train using labeled test dataset and train dataset 
   Trial No.    Reward   Elapsed                       Space Vector
0          3  0.972222  0.194367  [0, 3, 1, 2, 3, 1, 3, 0, 0, 1, 0]
1          5  0.972222  0.130711  [0, 2, 1, 0, 2, 0, 3, 0, 1, 4, 3]
2          8  0.972222  0.113038     [0, 1, 0, 0, 1, 0, 2, 0, 2, 3]
3         10  0.972222  0.134826      [1, 2, 0, 0, 500, 3, 2, 3, 4]
4          1  0.944444  0.251970                 [2, 2, 0, 3, 1, 2]

   Trial No.    Reward   Elapsed           Space Vector
0          1  0.972222  0.338019  [2, 0, 1, 0, 2, 4, 1]
1          2  0.972222  0.232059  [2, 3, 1, 1, 0, 4, 1]
2          3  0.972222  0.207254     [2, 3, 0, 3, 0, 2]
3          4  0.972222  0.262670  [2, 1, 1, 2, 1, 1, 0]
4          6  0.972222  0.246977     [2, 3, 0, 3, 1, 1]
>>> pipeline
Pipeline(steps=[('data_clean',
                 DataCleanStep(data_cleaner_args={}, name='data_clean',
                               random_state=9527)),
                ('estimator',
                 GreedyEnsemble(weight=[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.], estimators=[<hypergbm.estimators.CatBoostClassifierWrapper object at 0x1a38139110>, None, None, None, None, None, None, None, None, None]))])
>>> import numpy as np
>>> y_pred = pipeline.predict(X_test).astype(np.float64)
>>> 
>>> from sklearn.metrics import accuracy_score
>>> accuracy_score(y_pred, y_test)
1.0
```

##### Features selection

HyperGBM turn features into noise one by one for training, the more the model performance degradation, the more important the features become noise, so as to evaluate the importance of features.
Accord to features importance select part of the features and retraining model to save computing resources and time, here is an example:

```pydocstring
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> X, y = load_iris(return_X_y=True, as_frame=True)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> 
>>> from hypergbm.search_space import search_space_general
>>> from hypergbm import HyperGBM, CompeteExperiment
>>> from hypernets.searchers.evolution_searcher import EvolutionSearcher
>>> from sklearn.metrics import get_scorer
>>> 
>>> rs = EvolutionSearcher(search_space_general,  200, 100, optimize_direction='max')
>>> hk = HyperGBM(rs, task='multiclass', reward_metric='accuracy', callbacks=[])
>>> 
>>> experiment = CompeteExperiment(hk, X_train, y_train, X_test, y_test, callbacks=[], scorer=get_scorer('accuracy'),
...                                two_stage_importance_selection=True,  # enable feature importance selection
...                                n_est_feature_importance=3,  # use 3 estimators to evaluate feature importance
...                                importance_threshold=0.01)  # importance less than the threshold will not be selected
>>> pipeline = experiment.run(use_cache=True, max_trials=10)
   Trial No.  Reward   Elapsed                      Space Vector
0          2     1.0  0.373262                [2, 3, 0, 2, 2, 1]
1          3     1.0  0.194120  [1, 3, 1, 1, 365, 1, 3, 1, 0, 3]
2          4     1.0  0.109643  [1, 0, 1, 2, 140, 0, 2, 3, 4, 1]
3          6     1.0  0.107316    [0, 3, 0, 2, 2, 0, 1, 2, 2, 2]
4          7     1.0  0.117224   [1, 0, 1, 2, 40, 2, 1, 2, 4, 0]

             feature  importance       std
0  sepal length (cm)    0.000000  0.000000
1   sepal width (cm)    0.011111  0.015713
2  petal length (cm)    0.495556  0.199580
3   petal width (cm)    0.171111  0.112787

   Trial No.  Reward   Elapsed                      Space Vector
0          1     1.0  0.204705    [0, 1, 0, 2, 0, 1, 3, 0, 4, 3]
1          2     1.0  0.109204   [1, 1, 1, 2, 90, 1, 2, 0, 0, 1]
2          3     1.0  0.160209  [1, 2, 1, 0, 305, 3, 0, 0, 1, 1]
3          4     1.0  1.062759             [2, 1, 1, 2, 3, 1, 0]
4          6     1.0  0.218692    [0, 0, 0, 1, 0, 1, 2, 0, 0, 3]
>>> 
>>> import numpy as np
>>> y_pred = pipeline.predict(X_test).astype(np.float64)
>>> 
>>> from sklearn.metrics import accuracy_score
>>> accuracy_score(y_pred, y_test)
1.0
```

##### Concept drift

```pydocstring
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> 
>>> X, y = load_iris(return_X_y=True, as_frame=True)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> 
>>> from hypergbm.search_space import search_space_general
>>> from hypergbm import HyperGBM, CompeteExperiment
>>> from hypernets.searchers.evolution_searcher import EvolutionSearcher
>>> from sklearn.metrics import get_scorer
>>> 
>>> rs = EvolutionSearcher(search_space_general,  200, 100, optimize_direction='max')
>>> hk = HyperGBM(rs, task='multiclass', reward_metric='accuracy', callbacks=[])
>>> 
>>> experiment = CompeteExperiment(hk, X_train, y_train, X_test, y_test, callbacks=[], scorer=get_scorer('accuracy'),
...                                drift_detection=True)  # enable drift detection
>>> pipeline = experiment.run(use_cache=True, max_trials=10)
   Trial No.  Reward   Elapsed                       Space Vector
0          1     1.0  0.236796              [2, 2, 1, 3, 0, 4, 2]
1          3     1.0  0.207033     [0, 0, 0, 4, 1, 1, 2, 2, 1, 3]
2          4     1.0  0.106351      [1, 2, 0, 2, 240, 3, 2, 1, 2]
3          5     1.0  0.110495     [0, 0, 0, 2, 2, 0, 2, 1, 2, 2]
4          6     1.0  0.175838  [0, 3, 1, 3, 2, 1, 3, 1, 1, 4, 1]
>>> import numpy as np
>>> y_pred = pipeline.predict(X_test).astype(np.float64)
>>> from sklearn.metrics import accuracy_score
>>> accuracy_score(y_pred, y_test)
1.0
```

##### Ensemble

HyperGBM supports the combination of better models generated in the search process to a model with better generalization ability, example:
```pydocstring
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> 
>>> X, y = load_iris(return_X_y=True, as_frame=True)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> 
>>> from hypergbm.search_space import search_space_general
>>> from hypergbm import HyperGBM, CompeteExperiment
>>> from hypernets.searchers.evolution_searcher import EvolutionSearcher
>>> from sklearn.metrics import get_scorer
>>> 
>>> rs = EvolutionSearcher(search_space_general,  200, 100, optimize_direction='max')
>>> hk = HyperGBM(rs, task='multiclass', reward_metric='accuracy', callbacks=[])
>>> experiment = CompeteExperiment(hk, X_train, y_train, X_test, y_test, callbacks=[], scorer=get_scorer('accuracy'),
...                                ensemble_size=5)  # set ensemble
>>> pipeline = experiment.run(use_cache=True, max_trials=10)
   Trial No.  Reward   Elapsed                     Space Vector
0          1     1.0  0.856545            [2, 1, 1, 1, 3, 4, 1]
1          2     1.0  0.271147               [2, 0, 0, 1, 0, 2]
2          3     1.0  0.160234  [1, 0, 1, 0, 45, 2, 1, 3, 4, 0]
3          4     1.0  0.279989            [2, 0, 1, 0, 0, 1, 4]
4          5     1.0  0.262032            [2, 3, 1, 1, 0, 3, 2]
>>> import numpy as np
>>> y_pred = pipeline.predict(X_test).astype(np.float64)
>>> from sklearn.metrics import accuracy_score
>>> accuracy_score(y_pred, y_test)
1.0
```

#### Distributed

#### Custom search space
##### Feature generation

More features can be generated based on continuous features, such as the difference between two columns:
```pydocstring
>>> import pandas as pd
>>> df = pd.DataFrame(data={"x1": [1, 2, 4], "x2": [9, 8, 7]})
>>> df
   x1  x2
0   1   9
1   2   8
2   4   7
>>> from hypergbm.feature_generators import FeatureGenerationTransformer
>>> ft = FeatureGenerationTransformer(trans_primitives=['subtract_numeric'])
>>> ft.fit(df)
<hypergbm.feature_generators.FeatureGenerationTransformer object at 0x101839d10>
>>> ft.transform(df)
                      x1  x2  x1 - x2
e_hypernets_ft_index                 
0                      1   9       -8
1                      2   8       -6
2                      4   7       -3
```         

In addition to the `subtract_numeric` operation, it also support:
- add_numeric
- subtract_numeric
- divide_numeric
- multiply_numeric
- negate
- modulo_numeric
- modulo_by_feature
- cum_mean
- cum_sum
- cum_min
- cum_max
- percentile    
- absolute

It can also extract fields such as year, month, day and etc. from the datetime feature:
```pydocstring
>>> import pandas as pd
>>> from datetime import datetime
>>> df = pd.DataFrame(data={"x1":  pd.to_datetime([datetime.now()] * 10)})
>>> df[:3]
                          x1
0 2021-01-25 10:27:54.776580
1 2021-01-25 10:27:54.776580
2 2021-01-25 10:27:54.776580

>>> from hypergbm.feature_generators import FeatureGenerationTransformer
>>> ft = FeatureGenerationTransformer(trans_primitives=["year", "month", "week", "minute", "day", "hour", "minute", "second", "weekday", "is_weekend"])
>>> ft.fit(df)
<hypergbm.feature_generators.FeatureGenerationTransformer object at 0x1a29624dd0>
>>> ft.transform(df)
                                             x1  YEAR(x1)  MONTH(x1)  WEEK(x1)  MINUTE(x1)  DAY(x1)  HOUR(x1)  SECOND(x1)  WEEKDAY(x1)  IS_WEEKEND(x1)
e_hypernets_ft_index                                                                                                                                  
0                    2021-01-25 10:27:54.776580      2021          1         4          27       25        10          54            0           False
1                    2021-01-25 10:27:54.776580      2021          1         4          27       25        10          54            0           False
2                    2021-01-25 10:27:54.776580      2021          1         4          27       25        10          54            0           False
3                    2021-01-25 10:27:54.776580      2021          1         4          27       25        10          54            0           False

```

Using feature generation in search space：

```pydocstring
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> X, y = load_iris(return_X_y=True, as_frame=True)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> from hypergbm.estimators import XGBoostEstimator
>>> from hypergbm.pipeline import Pipeline
>>> from hypergbm.sklearn.transformers import FeatureGenerationTransformer
>>> from hypernets.core.ops import ModuleChoice, HyperInput
>>> from hypernets.core.search_space import HyperSpace
>>> from tabular_toolbox.column_selector import column_exclude_datetime
>>> 
>>> def search_space(task=None):  # Define a search space include feature geeration
...     space = HyperSpace()
...     with space.as_default():
...         input = HyperInput(name='input1')
...         feature_gen = FeatureGenerationTransformer(task=task,  # Add feature generation to search space
...                                                    trans_primitives=["add_numeric", "subtract_numeric", "divide_numeric", "multiply_numeric"]) 
...         full_pipeline = Pipeline([feature_gen], name=f'feature_gen_and_preprocess', columns=column_exclude_datetime)(input)
...         xgb_est = XGBoostEstimator(fit_kwargs={})
...         ModuleChoice([xgb_est], name='estimator_options')(full_pipeline)
...         space.set_inputs(input)
...     return space
>>> 
>>> from hypergbm import HyperGBM
>>> from hypernets.searchers.evolution_searcher import EvolutionSearcher
>>> 
>>> rs = EvolutionSearcher(search_space,  200, 100, optimize_direction='max')
>>> hk = HyperGBM(rs, task='multiclass', reward_metric='accuracy', callbacks=[])
>>> hk.search(X_train, y_train, X_eval=X_test, y_eval=y_test)
   Trial No.  Reward   Elapsed Space Vector
0          1     1.0  0.376869           []
>>> estimator = hk.load_estimator(hk.get_best_trial().model_file)
>>> y_pred = estimator.predict(X_test)
>>> 
>>> from sklearn.metrics import accuracy_score
>>> accuracy_score(y_test, y_pred)
1.0
```

##### Using GBM estimators

The GBM algorithms (wrapper class) supported by HyperGBM are:

- XGBoost (hypergbm.estimators.XGBoostEstimator)
- HistGB (hypergbm.estimators.HistGBEstimator)
- LightGBM (hypergbm.estimators.LightGBMEstimator)
- CatBoost (hypergbm.estimators.CatBoostEstimator)

The hyper-parameters are defined into the search space to use in training, here is an example that using xgboost to train iris:
```pydocstring
# Load dataset
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.datasets import load_iris
>>> X, y = load_iris(return_X_y=True, as_frame=True)
>>> X[:3]
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
>>> y[:3]
0    0
1    0
2    0
Name: target, dtype: int64
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

>>> from hypergbm.estimators import XGBoostEstimator
>>> from hypergbm.estimators import XGBoostEstimator
>>> from hypergbm.pipeline import Pipeline, DataFrameMapper
>>> from hypergbm.sklearn.transformers import MinMaxScaler, StandardScaler
>>> from hypernets.core import OptimizeDirection
>>> from hypernets.core.ops import ModuleChoice, HyperInput
>>> from hypernets.core.search_space import HyperSpace
>>> from tabular_toolbox.column_selector import column_number_exclude_timedelta
# Define search space included XGBoost
>>> def search_space():
...     space = HyperSpace()
...     with space.as_default():
...         input = HyperInput(name='input1')
...         scaler_choice = ModuleChoice(
...             [
...                 StandardScaler(name=f'numeric_standard_scaler'),
...                 MinMaxScaler(name=f'numeric_minmax_scaler')
...             ], name=f'numeric_or_scaler'
...         )
...         num_pipeline = Pipeline([scaler_choice], name='numeric_pipeline', columns=column_number_exclude_timedelta)(input)
...         union_pipeline = DataFrameMapper(default=None, input_df=True, df_out=True)([num_pipeline])
...         xgb_est = XGBoostEstimator(fit_kwargs={})
...         ModuleChoice([xgb_est], name='estimator_options')(union_pipeline)  # Make xgboost as a estimator choice
...         space.set_inputs(input)
...     return space

# Search
>>> from hypergbm import HyperGBM
>>> from hypernets.searchers import MCTSSearcher
>>> rs = MCTSSearcher(search_space, max_node_space=10, optimize_direction=OptimizeDirection.Maximize)
>>> hk = HyperGBM(rs, task='multiclass', reward_metric='accuracy', callbacks=[])
>>> 
>>> hk.search(X_train, y_train, X_eval=X_test, y_eval=y_test)
   Trial No.  Reward   Elapsed Space Vector
0          1     1.0  0.206926          [0]
1          2     1.0  0.069099          [1]
```

##### Class balancing 

HyperGBM supports several strategies for unbalanced data sampling:

**Class weight**
- ClassWeight

**Over sampling**
- RandomOverSampling
- SMOTE
- ADASYN

**Down sampling**
- RandomUnderSampling
- NearMiss
- TomeksLinks

Configure class balancing policies in estimator:
```pydocstring
...
xgb_est = XGBoostEstimator(fit_kwargs={}, class_balancing='ClassWeight')  # Use class balancing
...
```

Here is an example that training with `ClassWeight` sampling strategy:
```pydocstring
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split 
>>> X, y = load_iris(return_X_y=True, as_frame=True)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> from hypergbm.estimators import XGBoostEstimator
>>> from hypergbm.pipeline import Pipeline, DataFrameMapper
>>> from hypergbm.sklearn.transformers import MinMaxScaler, StandardScaler
>>> from hypernets.core.ops import ModuleChoice, HyperInput
>>> from hypernets.core.search_space import HyperSpace
>>> from tabular_toolbox.column_selector import column_number_exclude_timedelta
>>> 
>>> def search_space():
...     space = HyperSpace()
...     with space.as_default():
...         input = HyperInput(name='input1')
...         scaler_choice = ModuleChoice(
...             [
...                 StandardScaler(name=f'numeric_standard_scaler'),
...                 MinMaxScaler(name=f'numeric_minmax_scaler')
...             ], name='numeric_or_scaler'
...         )
...         num_pipeline = Pipeline([scaler_choice], name='numeric_pipeline', columns=column_number_exclude_timedelta)(input)
...         union_pipeline = DataFrameMapper(default=None, input_df=True, df_out=True)([num_pipeline])
...         xgb_est = XGBoostEstimator(fit_kwargs={}, class_balancing='ClassWeight')  # Use class balancing
...         ModuleChoice([xgb_est], name='estimator_options')(union_pipeline)
...         space.set_inputs(input)
...     return space
>>> from hypergbm import HyperGBM
>>> from hypernets.searchers.evolution_searcher import EvolutionSearcher
>>> 
>>> rs = EvolutionSearcher(search_space,  200, 100, optimize_direction='max')
>>> hk = HyperGBM(rs, task='multiclass', reward_metric='accuracy', callbacks=[])
>>> hk.search(X_train, y_train, X_eval=X_test, y_eval=y_test)
   Trial No.  Reward   Elapsed Space Vector
0          1     1.0  0.100520          [0]
1          2     1.0  0.083927          [1]
```


