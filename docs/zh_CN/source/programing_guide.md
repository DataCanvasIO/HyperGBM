## 用户指南

### 核心组件

本节简要的介绍HyperGBM中的核心组件，如下图所示：
![](images/hypergbm-main-components.png)

* HyperGBM(HyperModel)

    HyperGBM是HyperModel的一个具体实现（关于HyperModel的详细信息，请参考[Hypernets](https://github.com/DataCanvasIO/Hypernets)项目）。
    HyperGBM类是该项目最主要的接口，可以通过它提供的`search`方法来完成使用特定的`Searcher`(搜索器)在特定`Search Space`(搜索空间)中搜索并返回最佳的模型，简单来说就是一次自动建模的过程。

* Search Space（搜索空间）

    HyperGBM中的搜索空间由ModelSpace（各种Transformer和Estimator）,ConnectionSpace（Pipeline，实现Model间的连接和排列）,ParameterSpace（各种类型的超参数）三种搜索组件组合构成。Transformer使用Pipeline按顺序链接起来，Pipeline可以嵌套，这样可以实现非常复杂的多层次数据处理管道，整个搜索空间的最后一个节点只能是Estimator对象。所有的Transformer和Estimator都可以包含一组超参数搜索空间。
    下图是一个搜索空间的示例：
![](images/hypergbm-search-space.png)
    其中Numeric Pipeline的代码示例入下：
```python
import numpy as np
from hypergbm.pipeline import Pipeline
from hypergbm.sklearn.transformers import SimpleImputer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, LogStandardScaler
from hypernets.core.ops import ModuleChoice, Optional, Choice
from hypernets.tabular.column_selector import  column_number_exclude_timedelta


def numeric_pipeline_complex(impute_strategy=None, seq_no=0):
    if impute_strategy is None:
        impute_strategy = Choice(['mean', 'median', 'constant', 'most_frequent'])
    elif isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)

    imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'numeric_imputer_{seq_no}',
                            force_output_as_float=True)
    scaler_options = ModuleChoice(
        [
            LogStandardScaler(name=f'numeric_log_standard_scaler_{seq_no}'),
            StandardScaler(name=f'numeric_standard_scaler_{seq_no}'),
            MinMaxScaler(name=f'numeric_minmax_scaler_{seq_no}'),
            MaxAbsScaler(name=f'numeric_maxabs_scaler_{seq_no}'),
            RobustScaler(name=f'numeric_robust_scaler_{seq_no}')
        ], name=f'numeric_or_scaler_{seq_no}'
    )
    scaler_optional = Optional(scaler_options, keep_link=True, name=f'numeric_scaler_optional_{seq_no}')
    pipeline = Pipeline([imputer, scaler_optional],
                        name=f'numeric_pipeline_complex_{seq_no}',
                        columns=column_number_exclude_timedelta)
    return pipeline
```


* Searcher
    Searcher是用于在搜索空间中完成搜索过程的算法。一个搜索算法最核心的部分就是如何平衡exploration（勘探）和exploitation（开采）的策略，一方面要能够快速的逼近全局最优解，另一方面要避免陷入局部某个次优的局部空间。
    在HyperGBM中提供了MCTSSearcher（蒙特卡洛树搜索）、EvolutionarySearcher（进化搜索）和RandomSearcher（随机搜索）三种算法。
    
* HyperGBMEstimator

    HyperGBMEstimator是根据搜索空间中的一个样本来构建的评估器对象，其中包括完整的预处理器管道（Preprocessing Pipeline）和 一个特定的GBM算法模型，可以用来在训练集上`fit`，用评估集来`evaluate` 以及在新数据上完成`predict`。

* CompeteExperiment

    `CompeteExperiment`是HyperGBM提供的一个强大的工具，它不但可以完成pipeline搜索，同时还包括了一系列高级特性来进一步提升模型的性能，包括data drift handling（数据漂移处理）、pseudo-labeling（伪标签-半监督学习）、ensemble等等。
    
### 用例

* 使用HyperGBM
```python
# import HyperGBM, Search Space and Searcher
from hypergbm import HyperGBM
from hypergbm.search_space import search_space_general
from hypernets.searchers.random_searcher import RandomSearcher
import pandas as pd
from sklearn.model_selection import train_test_split

# instantiate related objects
searcher = RandomSearcher(search_space_general, optimize_direction='max')
hypergbm = HyperGBM(searcher, task='binary', reward_metric='accuracy')

# load data into Pandas DataFrame
df = pd.read_csv('[train_data_file]')
y = df.pop('target')

# split data into train set and eval set
# The evaluation set is used to evaluate the reward of the model fitted with the training set
X_train, X_eval, y_train, y_eval = train_test_split(df, y, test_size=0.3)

# search
hypergbm.search(X_train, y_train, X_eval, y_eval, max_trials=30)

# load best model
best_trial = hypergbm.get_best_trial()
estimator = hypergbm.load_estimator(best_trial.model_file)

# predict on real data
pred = estimator.predict(X_real)
```

* 使用Experiment
```python
from hypergbm import make_experiment
import pandas as pd

# load data into Pandas DataFrame
df = pd.read_csv('[train_data_file]')
target = 'target'

#create an experiment
experiment = make_experiment(df, target=target)

#run experiment
estimator = experiment.run()

# predict on real data
pred = estimator.predict(X_real)
```


### HyperGBM

**必选参数**

- *searcher*: hypernets.searcher.Searcher, A Searcher instance.
    `hypernets.searchers.RandomSearcher`
    `hypernets.searcher.MCTSSearcher`
    `hypernets.searchers.EvolutionSearcher`

**可选参数**

- *dispatcher*: hypernets.core.Dispatcher, Dispatcher为HyperModel提供不同的运行模式，分配和调度每个Trail（搜索采样和样本评估过程）时间的运行任务。例如：单进程串行搜索模式(`InProcessDispatcher`)，分布式并行搜索模式(`DaskDispatcher`) 等，用户可以根据需求实现自己的Dispatcher，系统默认使用`InProcessDispatcher`。
 - *callbacks*: list of callback functions or None, optional (default=None), 用于响应搜索过程中对应事件的Callback函数列表，更多信息请参考`hypernets.callbacks`。
- *reward_metric*: str or None, optinal(default=accuracy), 根据任务类型（分类、回归）设置的评估指标，用于指导搜索算法的搜索方向。
- *task*: str or None, optinal(default=None), 任务类型（*'binary'*,*'multiclass'* or *'regression'*）。如果为None会自动根据目标列的值推断任务类型。
- *param data_cleaner_params*: dict, (default=None), 用来初始化`DataCleaner`对象的参数字典. 如果为None将使用默认值初始化。
- *param cache_dir*: str or None, (default=None), 数据缓存的存储路径，如果为None默认使用当前进程的working dir下'tmp/cache'目录做为缓存目录。
- *param clear_cache*: bool, (default=True), 是否在开始搜索过程之前清空数据缓存所在目录。

#### search

**必选参数**

- *X*: Pandas or Dask DataFrame, 用于训练的特征数据
- *y*: Pandas or Dask Series, 用于训练的目标列
- *X_eval*: (Pandas or Dask DataFrame) or None, 用于评估的特征数据
- *y_eval*: (Pandas or Dask Series) or None, 用于评估的目标列

**可选参数**

- *cv*: bool, (default=False), 如果为True，会使用全部训练数据的cross-validation评估结果来指导搜索方向，否则使用指定的评估集的评估结果。
- *num_folds*: int, (default=3), cross-validated的折数，仅在cv=True时有效。
- *max_trials*: int, (default=10), 搜索尝试次数的上限。
- **fit_kwargs: dict, 可以通过这个参数来指定模型fit时的kwargs。

### Searchers

#### MCTSSearcher：蒙特卡洛树搜索（Monte-Carlo Tree Search）
    
蒙特卡洛树搜索(MCTS)是强化学习的一个分支，有着非常高效的搜索效率，可以满足高维动态搜索空间的效率需求。 MCTS扩展了著名的Multi-armed Bandit算法到树结构的搜索空间，MCTS通过selection, expansion, playout 和 backpropagation四个阶段不断迭代完成搜索。 

**Code example**
```
from hypernets.searchers import MCTSSearcher

searcher = MCTSSearcher(search_space_fn, use_meta_learner=False, max_node_space=10, candidates_size=10, optimize_direction='max')
```

**必选参数**
- *space_fn*: callable, 可以返回`HyperSpace`的搜索空间函数。

**可选参数**
- *policy*: hypernets.searchers.mcts_core.BasePolicy, (default=None), 设置*Selection* and *Backpropagation* 阶段的实现策略, 默认使用 `UCT` 策略.
- *max_node_space*: int, (default=10), 节点*Expansion*的最大空间。
- *use_meta_learner*: bool, (default=True), Meta-learner使用已评估的样本做为数据训练一个模型用来评估未知样本的表现，它使用模拟的方式在一组候选路径中选择最有前途的一条，而不需要对特定样本完成真正的训练，可以节省大量的时间和资源，有效的提升搜索效率。
- *candidates_size*: int, (default=10), Meta-learner在评估候选路径时采样的路径数量。
- *optimize_direction*: 'min' or 'max', (default='min'), 搜索器的优化方向，是寻找reward的最小值还是最大值。
- *space_sample_validation_fn*: callable or None, (default=None), 用于校验来自搜索空间的样本的有效性，可以通过这个函数来指定特定搜索空间的约束规则以减小搜索空间的大小。


#### EvolutionSearcher：Evolutionary Algorithm

进化算法（EA）是evolutionary computation的子集，这是一种基于种群的通用启发式优化算法。 EA受生物进化启发的机制，例如reproduction, mutation, recombination, and selection。 优化的候选解决方案在总体中扮演个体的角色，而fitness函数决定了解的质量（另请参见损失函数）。 然后，在重复应用上述算子之后，种群就会发生演化。


**Code example**
```
from hypernets.searchers import EvolutionSearcher

searcher = EvolutionSearcher(search_space_fn, population_size=20, sample_size=5, optimize_direction='min')
```

**必选参数**
- *space_fn*: callable, 可以返回`HyperSpace`的搜索空间函数。
- *population_size*: int, 种群的数量
- *sample_size*: int, 在每一次进化循环中选择的父代候选样本的数量。

**可选参数**
- *regularized*: bool, (default=False), 是否启用正则进化。
- *use_meta_learner*: bool, (default=True), Meta-learner使用已评估的样本做为数据训练一个模型用来评估未知样本的表现，它使用模拟的方式在一组候选路径中选择最有前途的一条，而不需要对特定样本完成真正的训练，可以节省大量的时间和资源，有效的提升搜索效率。
- *candidates_size*: int, (default=10), Meta-learner在评估候选路径时采样的路径数量。
- *optimize_direction*: 'min' or 'max', (default='min'), 搜索器的优化方向，是寻找reward的最小值还是最大值。
- *space_sample_validation_fn*: callable or None, (default=None), 用于校验来自搜索空间的样本的有效性，可以通过这个函数来指定特定搜索空间的约束规则以减小搜索空间的大小。


#### Random Search

每一次在搜索空间中随机选取样本进行搜索的算法

**Code example**
```
from hypernets.searchers import RandomSearcher

searcher = RandomSearcher(search_space_fn, optimize_direction='min')
```

**必选参数**
- *space_fn*: callable, 可以返回`HyperSpace`的搜索空间函数。

**可选参数**
- *optimize_direction*: 'min' or 'max', (default='min'), 搜索器的优化方向，是寻找reward的最小值还是最大值。
- *space_sample_validation_fn*: callable or None, (default=None), 用于校验来自搜索空间的样本的有效性，可以通过这个函数来指定特定搜索空间的约束规则以减小搜索空间的大小。


### Search Space

#### 内置Search Space

* search_space_general：

**Code example**
```
from hypergbm.search_space import search_space_general

searcher = RandomSearcher(search_space_general, optimize_direction='min')
# or 
searcher = RandomSearcher(lambda: search_space_general(n_estimators=300, early_stopping_rounds=10, verbose=0), optimize_direction='min')
```

#### Custom Search Space
**Code example**
```
```

### CompeteExperiment

在结构化数据的机器学习建模过程中依然存在这诸多挑战，例如：样本不均衡、数据漂移、泛化能力不足等问题。这些问题并不同通过模型搜索来完全解决，因此我们引入了一个更高级的工具`CompeteExperiment`。

`CompeteExperiment` 由一系列的step组成，Pipeline搜索只是其中的一步。它很多高级的特性，如数据清洗、数据漂移检测和处理、二阶段搜索、自动模型融合等。

如下图所示：

![](images/hypergbm-competeexperiment.png)



**必选参数**
- *hyper_model*: hypergbm.HyperGBM, 一个`HyperGBM` 实例。
- *X_train*: Pandas or Dask DataFrame, 用于训练的特征数据
- *y_train*: Pandas or Dask Series, 用于训练的目标列


**可选参数**
- *X_eval*: (Pandas or Dask DataFrame) or None, (default=None), 用于评估的特征数据
- *y_eval*: (Pandas or Dask Series) or None, (default=None), 用于评估的目标列
- *X_test*: (Pandas or Dask Series) or None, (default=None), 用于semi-supervised learning的无法观测到目标列的特征数据
- *eval_size*: float or int, (default=None), 仅在``X_eval`` 或 ``y_eval`` 为 None时有效。 如果是float值，应该是在0.0到1.0之间的数值，代表分割到验证集中的样本比例。如果是int值，代表分割到验证集中的样本数量。
- *train_test_split_strategy*: *'adversarial_validation'* or None, (default=None), 仅在``X_eval`` 或 ``y_eval`` 为 None时有效。 如果为None使用eval_size来分割数据，如果为'adversarial_validation' 使用对抗验证的方法来分割数据集。
- *cv*: bool, (default=False), 如果为True，会使用全部训练数据的cross-validation评估结果来指导搜索方向，否则使用指定的评估集的评估结果。
- *num_folds*: int, (default=3), cross-validated的折数，仅在cv=True时有效。
- *task*: str or None, optinal(default=None), 任务类型（*'binary'*,*'multiclass'* or *'regression'*）。如果为None会自动根据目标列的值推断任务类型。
- *callbacks*: list of callback functions or None, (default=None), 用于获取实验step事件的callback函数列表， See `hypernets.experiment.ExperimentCallback` for more information.
- *random_state*: int or RandomState instance, (default=9527), 数据分割时使用的随机状态。
- *scorer*: str, callable or None, (default=None), 用于特征重要性评估和ensemble的评分器，可以是scorer的名称(see [get_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html))，也可以是一个可以调用的函数(see [make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html))。如果为None会发生异常。
- *data_cleaner_args*: dict, (default=None), 用来初始化`DataCleaner`对象的参数字典. 如果为None将使用默认值初始化。
- *collinearity_detection*: bool, (default=False), 是否删除发生共线性的特征。
- *drift_detection*: bool,(default=True), 是否开启自动数据漂移检测和处理，只有提供了*X_test*时才生效。 数据漂移是建模过程中的一个主要挑战。当数据的分布随着时间在不断的发现变化时，模型的表现会越来越差，我们在HyperGBM中引入了对抗验证的方法专门处理数据漂移问题。这个方法会自动的检测是否发生漂移，并且找出发生漂移的特征并删除他们，以保证模型在真实数据上保持良好的状态。
- *feature_reselection*: bool, (default=True), 是否开始二阶段特征筛选和模型搜索。
- *feature_reselection_estimator_size*: int, (default=10), 用于评估特征重要性的estimator数量（在一阶段搜索中表现最好的n个模型。 仅在*feature_reselection*为True是有效。
- *feature_reselection_threshold*: float, (default=1e-5), 二阶搜索是特征选择重要性阈值，重要性低于该阈值的特征会被删除。仅在*feature_reselection*为True是有效。
- *ensemble_size*: int or None, (default=20), 用于ensemble的模型数量。ensemble_size为None或者小于1时会跳过ensemble step。 在模型搜索的过程中会产生很多模型，它们使用不同的预处理管道、不同的算法模型、不同的超参数，通常选择其中一些模型做ensemble会比只选择表现最好的单一模型获得更好的模型表现。
- *pseudo_labeling*: bool, (default=False), 是否开启伪标签学习。伪标签是一种半监督学习技术，将测试集中未观测标签列的特征数据通过一阶段训练的模型预测标签后，将置信度高于一定阈值的样本添加到训练数据中重新训练模型，有时候可以进一步提升模型在新数据上的拟合效果。
- *pseudo_labeling_proba_threshold*: float, (default=0.8), 伪标签的置信度阈值。 仅在 *pseudo_labeling* 为 True时有效。
- *pseudo_labeling_resplit*: bool, (default=False), 添加新的伪标签数据后是否重新分割训练集和评估集. 如果为False, 直接把所有伪标签数据添加到训练集中重新训练模型，否则把训练集、评估集及伪标签数据合并后重新分割. 仅在 *pseudo_labeling* 为 True时有效。
- *retrain_on_wholedata*: bool, (default=False), 在搜索完成后是否把训练集和评估集数据合并后用全量数据重新训练模型。
- *log_level*: int or None, (default=None), 搜索过程中的日志输出级别, possible values:[logging.CRITICAL, logging.FATAL, logging.ERROR, logging.WARNING, logging.WARN, logging.INFO, logging.DEBUG, logging.NOTSET]


**Code example**
```python
from hypergbm import make_experiment
from hypergbm.search_space import search_space_general
import pandas as pd
import logging

# load data into Pandas DataFrame
df = pd.read_csv('[train_data_file]')
target = 'target'

#create an experiment
experiment = make_experiment(df, target=target, 
                 search_space=lambda: search_space_general(class_balancing='SMOTE',n_estimators=300, early_stopping_rounds=10, verbose=0),
                 collinearity_detection=False,
                 drift_detection=True,
                 feature_reselection=False,
                 feature_reselection_estimator_size=10,
                 feature_reselection_threshold=1e-5,
                 ensemble_size=20,
                 pseudo_labeling=False,
                 pseudo_labeling_proba_threshold=0.8,
                 pseudo_labeling_resplit=False,
                 retrain_on_wholedata=False,
                 log_level=logging.ERROR,)

#run experiment
estimator = experiment.run()

# predict on real data
pred = estimator.predict(X_real)
```


#### Imbalance data handling

不均衡数据通常出现在分类任务中，出现不同类的样本分布极度不均匀的情况，我们提供了几种处理类别不均衡问题的方法，包括：
**Class Weight**
- ClassWeight

**Oversampling**
- RandomOverSampling
- SMOTE
- ADASYN

**Undersampling**
- RandomUnderSampling
- NearMiss
- TomeksLinks
- EditedNearestNeighbours

**Code example**
```python
from hypernets.tabular.datasets import dsutils
from sklearn.model_selection import train_test_split
from hypergbm.search_space import search_space_general
from hypergbm import make_experiment
# load data into Pandas DataFrame
df = dsutils.load_bank().head(1000)
target = 'y'
train, test = train_test_split(df, test_size=0.3)
#create an experiment
#possible values of class_balancing: None, 'ClassWeight','RandomOverSampling','SMOTE','ADASYN','RandomUnderSampling','NearMiss','TomeksLinks'
experiment = make_experiment(df, target=target, search_space=lambda: search_space_general(class_balancing='SMOTE'))
#run experiment
estimator = experiment.run()
# predict on test data without target values
test.pop(target)
pred = estimator.predict(test)
```


#### Pseudo labeling 
伪标签是一种半监督学习技术，将测试集中未观测标签列的特征数据通过一阶段训练的模型预测标签后，将置信度高于一定阈值的样本添加到训练数据中重新训练模型，有时候可以进一步提升模型在新数据上的拟合效果。
![](images/pseudo-labeling.png)

**Code example**
```
experiment = make_experiment(df, target=target, pseudo_labeling=True)
#run experiment
estimator = experiment.run()
```

#### Concept drift handling
数据漂移是建模过程中的一个主要挑战。当数据的分布随着时间在不断的发现变化时，模型的表现会越来越差，我们在HyperGBM中引入了对抗验证的方法专门处理数据漂移问题。这个方法会自动的检测是否发生漂移，并且找出发生漂移的特征并删除他们，以保证模型在真实数据上保持良好的状态。

**Code example**
```
experiment = make_experiment(df, target=target, drift_detection=True)
#run experiment
estimator = experiment.run()
```

#### Ensemble
在模型搜索的过程中会产生很多模型，它们使用不同的预处理管道、不同的算法模型、不同的超参数，通常选择其中一些模型做ensemble会比只选择表现最好的单一模型获得更好的模型表现。


**Code example**
```
experiment = make_experiment(df, target=target, ensemble_size=20)
#run experiment
estimator = experiment.run()
```

#### Early Stopping
在自动建模的Pipeline搜索过程中由于搜索空间巨大通常是个天文数字，我们无法遍历整个搜索空间，因此必须要设置一个最大搜索次数（max_trails）。max_trails通常会设置一个较大的数值以确保搜索算法能够完成对全局最优空间的发现。但如果只按照最大搜索次数完成所有的搜索会耗费大量的时间和算力，而且经常会出现在搜索次数未达到max_trails之前，已经基本逼近全局最优空间，很难继续提升模型效果的情况。
我们可以设置合理的提前停止策略来避免大量时间成本和算力的浪费。另外，提前停止策略也可以有效的防止模型过拟合问题。HyperGBM可以支持多种提前停止策略，并且几种策略可以组合应用，其中包括：
* max_no_improvement_trials (n次搜索都不再提升，提前停止)
* time_limit (最大用时提前停止)
* expected_reward (到达预期指标提前停止)

**Use experiment**
```python
from hypernets.core import EarlyStoppingCallback
from hypergbm.experiment import make_experiment

es = EarlyStoppingCallback(max_no_improvement_trials=0, mode='max', min_delta=0, time_limit=3600, expected_reward=0.95)

experiment = make_experiment(df, target=target, ensemble_size=20, search_callbacks=[es])
#run experiment
estimator = experiment.run()
```

**Use HyperGBM**
```python
from hypergbm import HyperGBM
from hypergbm.search_space import search_space_general
from hypernets.searchers import EvolutionSearcher
from hypernets.core import EarlyStoppingCallback,SummaryCallback

# instantiate related objects
searcher = EvolutionSearcher(search_space_general,optimize_direction='max', population_size=30, sample_size=10)
hypergbm = HyperGBM(searcher, task='binary', reward_metric='accuracy')

es = EarlyStoppingCallback(max_no_improvement_trials=0, mode='max', min_delta=0, time_limit=3600, expected_reward=0.95)
hk = HyperGBM(searcher, reward_metric='AUC', callbacks=[es, SummaryCallback()])
hk.search(...)

```


### 大规模数据支持

HyperGBM支持使用 [Dask](https://docs.dask.org/en/latest/) 集群进行数据处理和模型训练, 突破单服务器计算能力的限制。

Dask通过scheduler在进行任务调度， 有两种模式

* Single machine scheduler：单机模式，基于本地线程池或进程池实现。
* Distributed scheduler: 分布式模式， 利用多台服务器进行任务调度，支持公有云、Kubernets集群、Hadoop集群、HPC环境等多种部署方式。

关于部署Dask集群的详细信息请参考 [Dask 官方网站](https://docs.dask.org/en/latest/setup.html) .

#### 在实验中启用Dask支持

为了在实验中启用Dask支持，您需要：

* 配置 Dask scheduler 并初始化 Dask Client 对象
* 通过 Dask 加载数据训练数据和测试数据 (Dask DataFrame)
* 用 Dask DataFrame 创建 HyperGBM 实验

**Code example**

```python
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client
from hypergbm import make_experiment

cluster = LocalCluster(processes=True)
client = Client(cluster)
ddf_train = dd.read_parquet(...)
target = 'TARGET'
experiment = make_experiment(ddf_train, target=target)
estimator = experiment.run()

```

#### 自定义 Search Space

启用Dask时，要求Search Space中所使用的Transformer、Estimator都能够处理Dask数据集，HyperGBM内置的支持Dask的Trnsformer包括：

* SimpleImputer
* StandardScaler
* MinMaxScaler
* MaxAbsScaler
* RobustScaler
* SafeOneHotEncoder
* MultiLabelEncoder
* OrdinalEncoder
* SafeOrdinalEncoder
* TruncatedSVD

支持Dask的Estimator包括：

* XGBoostDaskEstimator
* LightGBMEstimator（LocalCluster only）
* CatBoostEstimator（LocalCluster only）

HyperGBM实验中默认的支持Dask的Search Space是`hypergbm.dask.search_space.search_space_general`, 用到了上述Transformer和Estimator。
HynperGBM允许您定义自己的Search Space，并在`make_experiment`时使用，如：

```python
from foo_package import bar_search_space

...

experiment = make_experiment(..., search_space=bar_search_space)

```

#### 在HyperGBM(HyperModel)中启用Dask支持


如果您希望直接使用HyperGBM(HyperModel)进行模型训练并中启用Dask支持，您需要：

* 配置 Dask scheduler 并初始化 Dask Client 对象
* 使用支持Dask的Search Space的创建HyperGBM(HyperModel)实例
* 通过 Dask 加载数据训练数据和测试数据 (Dask DataFrame)
* 用 Dask DataFrame 进行`search`

**Code example**

```python
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client
from dask_ml.model_selection import train_test_split
from hypergbm import HyperGBM
from hypergbm.dask.search_space import search_space_general

cluster = LocalCluster(processes=True)
client = Client(cluster)

ddf = dd.read_parquet(...)
target = 'TARGET'

X_train, X_eval = train_test_split(ddf, test_size=0.3, random_state=9527, shuffle=True)
y_train = X_train.pop(target)
y_eval = X_eval.pop(target)
X_train, X_test, y_train, y_test = client.persist([X_train, X_eval, y_train, y_eval])

hm = HyperGBM(search_space_general, task='binary', reward_metric='accuracy',callbacks=[])
hm.search(X_train, y_train, X_eval, y_eval, max_trials=200, use_cache=False, verbose=0)

```
