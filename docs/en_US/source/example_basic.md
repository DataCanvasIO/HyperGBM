## Basic Applications

In this section, we are going to provide an example to show how to train a model using the experiment. In this example, we use the `blood` dataset, which is loaded from `hypernets.tabular`. The columns of this dataset can be shown as follows:
```text
Recency,Frequency,Monetary,Time,Class
2,50,12500,98,1
0,13,3250,28,1
1,16,4000,35,1
2,20,5000,45,1
1,24,6000,77,0
4,4,1000,4,0

...

```


### Create and Run an Experiment
Using the tool `make_experiment` can create an executable experiment object. The only necessary parameter when using this tool is `train_data`. Then simply calling the method `run` of the created experiment object will start training and return a model. Note that if the target column of the data is not `y`, one needs to manually set it through the parameter `target`.

An example code:
```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class')
estimator = experiment.run()
print(estimator)

```

output:
```
Pipeline(steps=[('data_clean',
                 DataCleanStep(...),
                ('estimator',
                 GreedyEnsemble(...)])

```
Training will return a Pipeline while the final returned model is a collection of multiple models.



For training data with file extension .csv or .parquet, the experiment can be created through using the data file path directly and `make_experiment` will load data as DataFrame automatically. For an example:

```python
from hypergbm import make_experiment

train_data = '/path/to/mydata.csv'
experiment = make_experiment(train_data, target='my_target')
estimator = experiment.run()
print(estimator)

```  



### Use Cross Validation

Users can apply cross validation in teh experiment by manually setting parameter `cv`. Setting `cv` as 'False' will lead the experiment to avoid using cross validation and apply train_test_split instead. On the other hand, when `cv` is `True`, the experiment will use cross validation where the number of folds can be adjusted through the parameter `num_folds`. The default value of `num_folds` is 3.


Example code when `cv=True`:
```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', cv=True, num_folds=5)
estimator = experiment.run()
print(estimator)

```

### Evaluation dataset

When `cv=False`, training model will require evaluating its perfomance additionally on evaluation dataset. This can be done by setting `eval_data` when creating `make_experiment`. For example:

```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils
from sklearn.model_selection import train_test_split

train_data = dsutils.load_blood()
train_data,eval_data=train_test_split(train_data,test_size=0.3)
experiment = make_experiment(train_data, target='Class', eval_data=eval_data, cv=False)
estimator = experiment.run()
print(estimator)

```

If the `eval_data` is not given, the experiment object will split the `train_data` to get an evaluation dataset, whose size can be adjusted by setting `eval_size`:

```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', cv=False, eval_size=0.2)
estimator = experiment.run()
print(estimator)

```



### Set the Evaluation Criterion

The default evaluation criterion of the model when creating an experiment with `make_experiment` for classification task is `accuracy`, while the criterion for regression task is `rmse`. Other criterions can be used by setting `reward_metric`. For example:

```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', reward_metric='auc')
estimator = experiment.run()
print(estimator)

```



### Set the Numbers of Search and Early Stopping

One can set the max search numbers by adjusting `max_trials`. Early stopping strategy can be enabled and adjusted by setting `early_stopping_round`, `early_stopping_time_limit` and `early_stopping_reward`.

The following code sets the max searching time as 3 hours:
```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', max_trials=300, early_stopping_time_limit=3600 * 3)
estimator = experiment.run()
print(estimator)

```


### Choose a Searcher

HyperGBM performs parameter tuning with the search algorithms provided by Hypernets, which includes EvolutionSearch, MCTSSearcher, RandomSearcher. One can choose a specific searcher when using `make_experiment` by setting the parameter `searcher`.

```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', searcher='random')
estimator = experiment.run()
print(estimator)

```


Furthermore, you can make a new searcher object for experiment, for an example:

```python
from hypergbm import make_experiment
from hypergbm.search_space import search_space_general
from hypernets.searchers import MCTSSearcher
from hypernets.tabular.datasets import dsutils

my_searcher = MCTSSearcher(lambda: search_space_general(n_estimators=100),
                           max_node_space=20,
                           optimize_direction='max')

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', searcher=my_searcher)
estimator = experiment.run()
print(estimator)

```


### Ensemble Models

`make_experiment` automatically turns on the model ensembling function to get a better model when created. It will ensemble the best 20 models while the number for ensembling can be changed by setting `ensemble_size` as the following code, where `ensemble_size=0` means no ensembling wii be made.

```python
train_data = ...
experiment = make_experiment(train_data, ensemble_size=10, ...)

```



### Change the log level

The progress message during training can be shown by setting `log_level` (`str` or `int`) to change the log level. Please refer the `logging` package of python for further details. Besides, more thorough messages will show  when `verobs` is set as `1`.

The following codes sets the log level to 'INFO':
```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', log_level='INFO', verbose=1)
estimator = experiment.run()
print(estimator)

```

Output:
```console
14:24:33 I hypernets.tabular.u._common.py 30 - 2 class detected, {0, 1}, so inferred as a [binary classification] task
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
```

