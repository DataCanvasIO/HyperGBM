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
from tabular_toolbox.column_selector import  column_number_exclude_timedelta


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

**Required Parameters**

- *searcher*: hypernets.searcher.Searcher, A Searcher instance.
    `hypernets.searchers.RandomSearcher`
    `hypernets.searcher.MCTSSearcher`
    `hypernets.searchers.EvolutionSearcher`

**Optinal Parameters**

- *dispatcher*: hypernets.core.Dispatcher, Dispatcher为HyperModel提供不同的运行模式，分配和调度每个Trail（搜索采样和样本评估过程）时间的运行任务。例如：单进程串行搜索模式(`InProcessDispatcher`)，分布式并行搜索模式(`DaskDispatcher`) 等，用户可以根据需求实现自己的Dispatcher，系统默认使用`InProcessDispatcher`。
 - *callbacks*: list of callback functions or None, optional (default=None), 用于响应搜索过程中对应事件的Callback函数列表，更多信息请参考`hypernets.callbacks`。
- *reward_metric*: str or None, optinal(default=accuracy), 根据任务类型（分类、回归）设置的评估指标，用于指导搜索算法的搜索方向。
- *task*: str or None, optinal(default=None), 任务类型（*'binary'*,*'multiclass'* or *'regression'*）。如果为None会自动根据目标列的值推断任务类型。
- *param data_cleaner_params*: dict, (default=None), 用来初始化`DataCleaner`对象的参数字典. 如果为None将使用默认值初始化。
- *param cache_dir*: str or None, (default=None), 数据缓存的存储路径，如果为None默认使用当前进程的working dir下'tmp/cache'目录做为缓存目录。
- *param clear_cache*: bool, (default=True), 是否在开始搜索过程之前清空数据缓存所在目录。

#### search

**Required Parameters**

- *X*: Pandas or Dask DataFrame, 用于训练的特征数据
- *y*: Pandas or Dask Series, 用于训练的目标列
- *X_eval*: (Pandas or Dask DataFrame) or None, 用于评估的特征数据
- *y_eval*: (Pandas or Dask Series) or None, 用于评估的目标列

**Optinal Parameters**

- *cv*: bool, (default=False), 如果为True，会使用全部训练数据的cross-validation评估结果来指导搜索方向，否则使用指定的评估集的评估结果。
- *num_folds*: int, (default=3), cross-validated的折数，仅在cv=True时有效。
- *max_trials*: int, (default=10), 搜索尝试次数的上限。
- **fit_kwargs: dict, 可以通过这个参数来指定模型fit时的kwargs。

### Searchers

#### Monte-Carlo Tree Search
    
蒙特卡洛树搜索(MCTS) 扩展了著名的Multi-armed Bandit算法到树结构的搜索空间，MCTS通过selection, expansion, playout 和 backpropagation四个阶段不断迭代完成搜索。 MCTS是强化学习的一个分支，有着非常高效的搜索效率，可以满足高维动态搜索空间的效率需求。

**Code example**
```
from hypernets.searchers import MCTSSearcher

searcher = MCTSSearcher(search_space_fn, use_meta_learner=False, max_node_space=10, candidates_size=10, optimize_direction='max')
```

**Required Parameters**
- *space_fn*: callable, 可以返回`HyperSpace`的搜索空间函数。

**Optinal Parameters**
- *policy*: hypernets.searchers.mcts_core.BasePolicy, (default=None), 设置*Selection* and *Backpropagation* 阶段的实现策略, 默认使用 `UCT` 策略.
- *max_node_space*: int, (default=10), 节点*Expansion*的最大空间。
- *use_meta_learner*: bool, (default=True), Meta-learner使用已评估的样本做为数据训练一个模型用来评估未知样本的表现，它使用模拟的方式在一组候选路径中选择最有前途的一条，而不需要对特定样本完成真正的训练，可以节省大量的时间和资源，有效的提升搜索效率。
- *candidates_size*: int, (default=10), Meta-learner在评估候选路径时采样的路径数量。
- *optimize_direction*: 'min' or 'max', (default='min'), 搜索器的优化方向，是寻找reward的最小值还是最大值。
- *space_sample_validation_fn*: callable or None, (default=None), 用于校验来自搜索空间的样本的有效性，可以通过这个函数来指定特定搜索空间的约束规则以减小搜索空间的大小。


#### Evolutionary Algorithm

Evolutionary algorithm (EA) is a subset of evolutionary computation, a generic population-based metaheuristic optimization algorithm. An EA uses mechanisms inspired by biological evolution, such as reproduction, mutation, recombination, and selection. Candidate solutions to the optimization problem play the role of individuals in a population, and the fitness function determines the quality of the solutions (see also loss function). Evolution of the population then takes place after the repeated application of the above operators.


**Code example**
```
from hypernets.searchers import EvolutionSearcher

searcher = EvolutionSearcher(search_space_fn, population_size=20, sample_size=5, optimize_direction='min')
```

**Required Parameters**
- *space_fn*: callable, A search space function which when called returns a `HyperSpace` instance
- *population_size*: int, Size of population
- *sample_size*: int, The number of parent candidates selected in each cycle of evolution

**Optinal Parameters**
- *regularized*: bool, (default=False), Whether to enable regularized
- *use_meta_learner*: bool, (default=True), Meta-learner aims to evaluate the performance of unseen samples based on previously evaluated samples. It provides a practical solution to accurately estimate a search branch with many simulations without involving the actual training.
- *candidates_size*: int, (default=10), The number of samples for the meta-learner to evaluate candidate paths when roll out
- *optimize_direction*: 'min' or 'max', (default='min'), Whether the search process is approaching the maximum or minimum reward value.
- *space_sample_validation_fn*: callable or None, (default=None), Used to verify the validity of samples from the search space, and can be used to add specific constraint rules to the search space to reduce the size of the space.


#### Random Search

As its name suggests, Random Search uses random combinations of hyperparameters.
**Code example**
```
from hypernets.searchers import RandomSearcher

searcher = RandomSearcher(search_space_fn, optimize_direction='min')
```

**Required Parameters**
- *space_fn*: callable, A search space function which when called returns a `HyperSpace` instance

**Optinal Parameters**
- *optimize_direction*: 'min' or 'max', (default='min'), Whether the search process is approaching the maximum or minimum reward value.
- *space_sample_validation_fn*: callable or None, (default=None), Used to verify the validity of samples from the search space, and can be used to add specific constraint rules to the search space to reduce the size of the space.


### Search Space
#### Build-in Search Space
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
There are still many challenges in the machine learning modeling process for tabular data, such as imbalanced data, data drift, poor generalization ability, etc.  This challenges cannot be completely solved by pipeline search, so we introduced in HyperGBM a more powerful tool is `CompeteExperiment`.

`CompteExperiment` is composed of a series of steps and *Pipeline Search* is just one step. It also includes advanced steps such as data cleaning, data drift handling, two-stage search, ensemble etc., as shown in the figure below:
![](images/hypergbm-competeexperiment.png)


**Code example**
```
```


**Required Parameters**
- *hyper_model*: hypergbm.HyperGBM, A `HyperGBM` instance
- *X_train*: Pandas or Dask DataFrame, Feature data for training
- *y_train*: Pandas or Dask Series, Target values for training


**Optinal Parameters**
- *X_eval*: (Pandas or Dask DataFrame) or None, (default=None), Feature data for evaluation
- *y_eval*: (Pandas or Dask Series) or None, (default=None), Target values for evaluation
- *X_test*: (Pandas or Dask Series) or None, (default=None), Unseen data without target values for semi-supervised learning
- *eval_size*: float or int, (default=None), Only valid when ``X_eval`` or ``y_eval`` is None. If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the eval split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. 
- *train_test_split_strategy*: *'adversarial_validation'* or None, (default=None), Only valid when ``X_eval`` or ``y_eval`` is None. If None, use eval_size to split the dataset, otherwise use adversarial validation approach.
- *cv*: bool, (default=False), If True, use cross-validation instead of evaluation set reward to guide the search process
- *num_folds*: int, (default=3), Number of cross-validated folds, only valid when cv is true
- *task*: str or None, optinal(default=None), Task type(*binary*, *multiclass* or *regression*). If None, inference the type of task automatically
- *callbacks*: list of callback functions or None, (default=None), List of callback functions that are applied at each experiment step. See `hypernets.experiment.ExperimentCallback` for more information.
- *random_state*: int or RandomState instance, (default=9527), Controls the shuffling applied to the data before applying the split.
- *scorer*: str, callable or None, (default=None), Scorer to used for feature importance evaluation and ensemble. It can be a single string (see [get_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html)) or a callable (see [make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)). If None, exception will occur.
- *data_cleaner_args*: dict, (default=None),  dictionary of parameters to initialize the `DataCleaner` instance. If None, `DataCleaner` will initialized with default values.
- *drop_feature_with_collinearity*: bool, (default=False), Whether to clear multicollinearity features
- *drift_detection*: bool,(default=True), Whether to enable data drift detection and processing. Only valid when *X_test* is provided. Concept drift in the input data is one of the main challenges. Over time, it will worsen the performance of model on new data. We introduce an adversarial validation approach to concept drift problems in HyperGBM. This approach will detect concept drift and identify the drifted features and process them automatically.
- *two_stage_importance_selection*: bool, (default=True), Whether to enable two stage feature selection and searching
- *n_est_feature_importance*: int, (default=10), The number of estimator to evaluate feature importance. Only valid when *two_stage_importance_selection* is True.
- *importance_threshold*: float, (default=1e-5), The threshold for feature selection. Features with importance below the threshold will be dropped.  Only valid when *two_stage_importance_selection* is True.
- *ensemble_size*: int, (default=20), The number of estimator to ensemble. During the AutoML process, a lot of models will be generated with different preprocessing pipelines, different models, and different hyperparameters. Usually selecting some of the models that perform well to ensemble can obtain better generalization ability than just selecting the single best model.
- *pseudo_labeling*: bool, (default=False), Whether to enable pseudo labeling. Pseudo labeling is a semi-supervised learning technique, instead of manually labeling the unlabelled data, we give approximate labels on the basis of the labelled data. Pseudo-labeling can sometimes improve the generalization capabilities of the model.
- *pseudo_labeling_proba_threshold*: float, (default=0.8), Confidence threshold of pseudo-label samples. Only valid when *two_stage_importance_selection* is True.
- *pseudo_labeling_resplit*: bool, (default=False), Whether to re-split the training set and evaluation set after adding pseudo-labeled data. If False, the pseudo-labeled data is only appended to the training set. Only valid when *two_stage_importance_selection* is True.
- *retrain_on_wholedata*: bool, (default=False), Whether to retrain the model with whole data after the search is completed.
- *log_level*: int or None, (default=None), Level of logging, possible values:[logging.CRITICAL, logging.FATAL, logging.ERROR, logging.WARNING, logging.WARN, logging.INFO, logging.DEBUG, logging.NOTSET]

#### Imbalance data handling
Imbalanced data typically refers to a classification problem where the number of samples per class is not equally distributed; often you'll have a large amount of samples for one class (referred to as the majority class), and much fewer samples for one or more other classes (referred to as the minority classes). 
We have provided several approaches to deal with imbalanced data: *Class Weight*, *Oversampling* and *Undersampling*.

**Class Weight**
- ClassWeight

**Oversampling**
- RandomOverSampling
- SMOTE
- ADASYN

**Undersampling**
- RandomUnderSampling
- Near miss
- Tomeks links

#### Pseudo labeling 
Pseudo labeling is a semi-supervised learning technique, instead of manually labeling the unlabelled data, we give approximate labels on the basis of the labelled data. Pseudo-labeling can sometimes improve the generalization capabilities of the model. Let’s make it simpler by breaking into steps as shown in the figure below.

![](images/pseudo-labeling.png)


#### Concept drift handling
Concept drift in the input data is one of the main challenges. Over time, it will worsen the performance of model on new data. We introduce an adversarial validation approach to concept drift problems in HyperGBM. This approach will detect concept drift and identify the drifted features and process them automatically.


#### Ensemble
During the AutoML process, a lot of models will be generated with different preprocessing pipelines, different models, and different hyperparameters. Usually selecting some of the models that perform well to ensemble can obtain better generalization ability than just selecting the single best model.