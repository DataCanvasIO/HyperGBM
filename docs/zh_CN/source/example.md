## 例子

### 使用特征衍生

基于数值特征可以生成更多的特征，比如以两列之差作为新的特征：
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

除了`subtract_numeric`操作外还支持：
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

还能从日期列中提取出年、月、日等信息：
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

在搜索空间中使用特征衍生：

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

### 使用GBM算法

hypergbm支持的的GBM的算法(对应封装类)有：
- XGBoost (hypergbm.estimators.XGBoostEstimator)
- HistGB (hypergbm.estimators.HistGBEstimator)
- LightGBM (hypergbm.estimators.LightGBMEstimator)
- CatBoost (hypergbm.estimators.CatBoostEstimator)

将算法的超参数定义到搜索空间内以在训练时候使用该算法，以使用XGBoost训练iris为例：
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

### 自动任务类型推断

HyperGBM支持自动任务类型推断，如果您不指定任务类型，它会对标签列的分布按照下规则进行推断：

- 如果该列只有两个不同值推断成二分类
- 如果目标的不同值多余2个且类型为float则推断为回归
- 以上两条都不符合，则推断为多分类，超过1000个类别不支持

使用时设置`task=None` 或者不指定`task`即可自动推断：

```pydocstring
...
hk = HyperGBM(rs, task=None, reward_metric='accuracy', callbacks=[])
...
```

例子：
```pydocstring
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> from hypergbm.search_space import search_space_general
>>> 
>>> X, y = load_iris(return_X_y=True, as_frame=True)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> 
>>> from hypergbm import HyperGBM
>>> from hypernets.searchers.evolution_searcher import EvolutionSearcher
>>> 
>>> rs = EvolutionSearcher(search_space_general,  200, 100, optimize_direction='max')
>>> hk = HyperGBM(rs, reward_metric='accuracy', callbacks=[])  # do not set `task` param will using automatic task infer
>>> hk.search(X_train, y_train, X_eval=X_test, y_eval=y_test)

16:34:21 I hypernets.m.hyper_model.py 249 - 3 class detected, inferred as a [multiclass classification] task
```

### 使用交叉验证

HyperGBM支持使用交叉验证对模型进行评估，在使用search方法时指定cv的折数：

```python
...
hk = HyperGBM(rs, task='multiclass', reward_metric='accuracy', callbacks=[])
hk.search(X_train, y_train, X_eval=None, y_eval=None, cv=3)  # 3 fold
...
```

其中评估数据将从`X_train`, `y_train`中产生, 评估数据参数指定为`None`即可，以使用交叉验证评估方式训练数据集iris为例：
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
>>> hk.search(X_train, y_train, X_eval=None, y_eval=None, cv=3)  # using Cross Validation
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

## 切换搜索算法

HyperGBM提供以下搜索算法（对应实现类）：
  - 进化算法（hypernets.searchers.evolution_searcher.EvolutionSearcher）
  - 蒙特卡洛树算法（hypernets.searchers.mcts_searcher.MCTSSearcher）
  - 随机搜索算法（hypernets.searchers.random_searcher.RandomSearcher）

以使用进化算法训练iris数据集为例：

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

### 提前停止策略

当模型性能无法提高或满足一定条件时可以提前结束训练以释放计算资源，支持的提前停止策略有：
* max_no_improvement_trials (n次搜索都不再提升，提前停止)
* time_limit (最大用时提前停止)
* expected_reward (到达预期指标提前停止)

当设置多个条件时，先达到任意一个条件时就停止。
提前停止策略通过`hypernets.core.callbacks.EarlyStoppingCallback`实现，以训练数据集iris当reward达到0.95以上就停止搜索为例：

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

### 处理不平衡数据

hypergbm支持对不平衡数据进行采样，支持的策略有：

**类别权重**
- ClassWeight

**过采样**
- RandomOverSampling
- SMOTE
- ADASYN

**降采样**
- RandomUnderSampling
- NearMiss
- TomeksLinks

可以在Estimator中配置类平衡的策略，例如：
```pydocstring
...
xgb_est = XGBoostEstimator(fit_kwargs={}, class_balancing='ClassWeight')  # Use class balancing
...
```

以训练iris数据集为例，使用`ClassWeight` 采样策略：
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

### 伪标签

伪标签允许以半监督的方式使用测试集进行训练，纳入更多的训练数据来提升模型表现，使用方法：
```pydocstring
...
experiment = CompeteExperiment(hk, X_train, y_train, X_test=X_test, callbacks=[], scorer=get_scorer('accuracy'),
                               pseudo_labeling=True,  # Enable pseudo label
                               pseudo_labeling_proba_threshold=0.9) 
...
```

以使用伪标签训练iris数据集为例：

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


### 特征选择

HyperGBM可以逐个将特征变成噪音进行训练评估，观察模型性能下降的程度；模型性能下降的越多说明变成噪音的特征越重要，以此来评估特征的重要性，根据
特征重要性选取一部分特征重新训练模型来节省训练资源和时间，例子：

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


### 概念漂移处理

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

### 模型融合

HyperGBM支持将在搜索过程中产生的的较好的模型组合在一起形成一个泛化能力更好的模型，例子：

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
