## Programing Guide

### Main components

In this section, we briefly cover the **main components** in HyperGBM.
As shown below:
![](images/hypergbm-main-components.png)

* HyperGBM(HyperModel)

    HyperGBM is a specific implementation of HyperModel (for HyperModel, please refer to the [Hypernets](https://github.com/DataCanvasIO/Hypernets) project).
It is the core interface of the HyperGBM project. By calling the `search` method to explore and return the best model in the specified `Search Space` with the specified `Searcher`.


* Search Space

    Search spaces are constructed by arranging ModelSpace(transformer and estimator), ConnectionSpace(pipeline) and ParameterSpace(hyperparameter). The transformers are chained together by pipelines while the pipelines can be nested. The last node of a search space must be an estimator. Each transformer and estimator can define a set of hyperparameterss.

![](images/hypergbm-search-space.png)


* Searcher

    Searcher is an algorithm used to explore a search space.It encompasses the classical exploration-exploitation trade-off since, on the one hand, it is desirable to find well-performing model quickly, while on the other hand, premature convergence to a region of suboptimal solutions should be avoided.

* HyperGBMEstimator

    HyperGBMEstimator is an object built from a sample in the search space, including the full preprocessing pipeline and a GBM model. It can be used to `fit` on training data, `evaluate` on evaluation data, and `predict` on new data.

* CompeteExperiment

### Quick Start

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
df = pd.read_csv('/data_file_path/')
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

### HyperGBM

**Required Parameters**

- searcher: hypernets.searcher.Searcher, A Searcher instance.
    `hypernets.searchers.RandomSearcher`
    `hypernets.searcher.MCTSSearcher`
    `hypernets.searchers.EvolutionSearcher`

**Optinal Parameters**

- dispatcher: hypernets.core.Dispatcher, Dispatcher is used to provide different execution modes for search trials, such as stand-alone mode (`InProcessDispatcher`), distributed parallel mode (`DaskDispatcher`), etc. `InProcessDispatcher` is used by default.
- callbacks: list of callback functions or None, optional (default=None), List of callback functions that are applied at each trial. See `hypernets.callbacks` for more information.
- reward_metric: str or None, optinal(default=accuracy), Set corresponding metric  according to task type to guide search direction of searcher.
- task: str or None, optinal(default=None), Task type(*binary*,*multiclass* or *regression*). If None, inference the type of task automatically


#### search

**Required Parameters**

- X: Pandas or Dask DataFrame, feature data for training
- y: Pandas or Dask Series, target values for training
- X_eval: (Pandas or Dask DataFrame) or None, feature data for evaluation
- y_eval: (Pandas or Dask Series) or None, target values for evaluation

**Optinal Parameters**

- cv: bool, (default=False), If True, use cross-validation instead of evaluation set reward to guide the search process
- num_folds: int, (default=3), Number of cross-validated folds, only valid when cv is true
- max_trials: int, (default=10), The upper limit of the number of search trials, the search process stops when the number is exceeded
- **fit_kwargs: dict, parameters for fit method of model

### Searchers

#### Monte-Carlo Tree Search
    
Monte-Carlo Tree Search (MCTS) extends the celebrated Multi-armed Bandit algorithm to tree-structured search spaces. The MCTS algorithm iterates over four phases: selection, expansion, playout and backpropagation.
    
* Selection: In each node of the tree, the child node is selected after a Multi-armed Bandit strategy, e.g. the UCT (Upper Confidence bound applied to Trees) algorithm.

* Expansion: The algorithm adds one or more nodes to the tree. This node corresponds to the first encountered position that was not added in the tree.

* Playout: When reaching the limits of the visited tree, a roll-out strategy is used to select the options until reaching a terminal node and computing the associated
reward.

* Backpropagation: The reward value is propagated back, i.e. it is used to update the value associated to all nodes along the visited path up to the root node.

**Code example**
```
from hypernets.searchers import MCTSSearcher

searcher = MCTSSearcher(search_space_fn, use_meta_learner=False, max_node_space=10, candidates_size=10, optimize_direction='max')
```

**Required Parameters**
- space_fn: callable, A search space function which when called returns a `HyperSpace` instance.

**Optinal Parameters**
- policy: hypernets.searchers.mcts_core.BasePolicy, (default=None), The policy for *Selection* and *Backpropagation* phases, `UCT` by default.
- max_node_space: int, (default=10), Maximum space for node expansion
- use_meta_learner: bool, (default=True), Meta-learner aims to evaluate the performance of unseen samples based on previously evaluated samples. It provides a practical solution to accurately estimate a search branch with many simulations without involving the actual training.
- candidates_size: int, (default=10), The number of samples for the meta-learner to evaluate candidate paths when roll out
- optimize_direction: 'min' or 'max', (default='min'), Whether the search process is approaching the maximum or minimum reward value.
- space_sample_validation_fn: callable or None, (default=None), Used to verify the validity of samples from the search space, and can be used to add specific constraint rules to the search space to reduce the size of the space.

#### Evolutionary Algorithm

Evolutionary algorithm (EA) is a subset of evolutionary computation, a generic population-based metaheuristic optimization algorithm. An EA uses mechanisms inspired by biological evolution, such as reproduction, mutation, recombination, and selection. Candidate solutions to the optimization problem play the role of individuals in a population, and the fitness function determines the quality of the solutions (see also loss function). Evolution of the population then takes place after the repeated application of the above operators.


**Code example**
```
from hypernets.searchers import EvolutionSearcher

searcher = EvolutionSearcher(search_space_fn, population_size=20, sample_size=5, optimize_direction='min')
```

**Required Parameters**
- space_fn: callable, A search space function which when called returns a `HyperSpace` instance
- population_size: int, Size of population
- sample_size: int, The number of parent candidates selected in each cycle of evolution

**Optinal Parameters**
- regularized: bool, (default=False), Whether to enable regularized
- use_meta_learner: bool, (default=True), Meta-learner aims to evaluate the performance of unseen samples based on previously evaluated samples. It provides a practical solution to accurately estimate a search branch with many simulations without involving the actual training.
- candidates_size: int, (default=10), The number of samples for the meta-learner to evaluate candidate paths when roll out
- optimize_direction: 'min' or 'max', (default='min'), Whether the search process is approaching the maximum or minimum reward value.
- space_sample_validation_fn: callable or None, (default=None), Used to verify the validity of samples from the search space, and can be used to add specific constraint rules to the search space to reduce the size of the space.


#### Random Search

As its name suggests, Random Search uses random combinations of hyperparameters.
**Code example**
```
from hypernets.searchers import RandomSearcher

searcher = RandomSearcher(search_space_fn, optimize_direction='min')
```

**Required Parameters**
- space_fn: callable, A search space function which when called returns a `HyperSpace` instance

**Optinal Parameters**
- optimize_direction: 'min' or 'max', (default='min'), Whether the search process is approaching the maximum or minimum reward value.
- space_sample_validation_fn: callable or None, (default=None), Used to verify the validity of samples from the search space, and can be used to add specific constraint rules to the search space to reduce the size of the space.


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
- hyper_model: hypergbm.HyperGBM, A `HyperGBM` instance
- X_train: Pandas or Dask DataFrame, Feature data for training
- y_train: Pandas or Dask Series, Target values for training


**Optinal Parameters**
- X_eval: (Pandas or Dask DataFrame) or None, (default=None), Feature data for evaluation
- y_eval: (Pandas or Dask Series) or None, (default=None), Target values for evaluation
- X_test: (Pandas or Dask Series) or None, (default=None), Unseen data without target values for semi-supervised learning
- eval_size: float or int, (default=None), Only valid when ``X_eval`` or ``y_eval`` is None. If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the eval split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. 
- train_test_split_strategy: *'adversarial_validation'* or None, (default=None), Only valid when ``X_eval`` or ``y_eval`` is None. If None, use eval_size to split the dataset, otherwise use adversarial validation approach.
- cv: bool, (default=False), If True, use cross-validation instead of evaluation set reward to guide the search process
- num_folds: int, (default=3), Number of cross-validated folds, only valid when cv is true
- task: str or None, optinal(default=None), Task type(*binary*, *multiclass* or *regression*). If None, inference the type of task automatically
- callbacks: list of callback functions or None, (default=None), List of callback functions that are applied at each experiment step. See `hypernets.experiment.ExperimentCallback` for more information.
- random_state: int or RandomState instance, (default=9527), Controls the shuffling applied to the data before applying the split.
- scorer: str, callable or None, (default=None), Scorer to used for feature importance evaluation and ensemble. It can be a single string (see [get_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html)) or a callable (see [make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)). If None, exception will occur.
- data_cleaner_args: dict, (default=None),  dictionary of parameters to initialize the `DataCleaner` instance. If None, `DataCleaner` will initialized with default values.
- drop_feature_with_collinearity: bool, (default=False), Whether to clear multicollinearity features
- drift_detection: bool,(default=True), Whether to enable data drift detection and processing. Only valid when *X_test* is provided. Concept drift in the input data is one of the main challenges. Over time, it will worsen the performance of model on new data. We introduce an adversarial validation approach to concept drift problems in HyperGBM. This approach will detect concept drift and identify the drifted features and process them automatically.
- two_stage_importance_selection: bool, (default=True), Whether to enable two stage feature selection and searching
- n_est_feature_importance: int, (default=10), The number of estimator to evaluate feature importance. Only valid when *two_stage_importance_selection* is True.
- importance_threshold: float, (default=1e-5), The threshold for feature selection. Features with importance below the threshold will be dropped.  Only valid when *two_stage_importance_selection* is True.
- ensemble_size: int, (default=20), The number of estimator to ensemble. During the AutoML process, a lot of models will be generated with different preprocessing pipelines, different models, and different hyperparameters. Usually selecting some of the models that perform well to ensemble can obtain better generalization ability than just selecting the single best model.
- pseudo_labeling: bool, (default=False), Whether to enable pseudo labeling. Pseudo labeling is a semi-supervised learning technique, instead of manually labeling the unlabelled data, we give approximate labels on the basis of the labelled data. Pseudo-labeling can sometimes improve the generalization capabilities of the model.
- pseudo_labeling_proba_threshold: float, (default=0.8), Confidence threshold of pseudo-label samples. Only valid when *two_stage_importance_selection* is True.
- pseudo_labeling_resplit: bool, (default=False), Whether to re-split the training set and evaluation set after adding pseudo-labeled data. If False, the pseudo-labeled data is only appended to the training set. Only valid when *two_stage_importance_selection* is True.
- retrain_on_wholedata: bool, (default=False), Whether to retrain the model with whole data after the search is completed.
- log_level: int or None, (default=None), Level of logging, possible values:[logging.CRITICAL, logging.FATAL, logging.ERROR, logging.WARNING, logging.WARN, logging.INFO, logging.DEBUG, logging.NOTSET]

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