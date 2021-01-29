
## HyperGBM

    HyperGBM is a specific implementation of HyperModel (for HyperModel, please refer to the [Hypernets](https://github.com/DataCanvasIO/Hypernets) project).
It is the core interface of the HyperGBM project. By calling the `search` method to explore and return the best model in the specified `Search Space` with the specified `Searcher`.


**Required Parameters**

- *searcher*: hypernets.searcher.Searcher, A Searcher instance.
    `hypernets.searchers.RandomSearcher`
    `hypernets.searcher.MCTSSearcher`
    `hypernets.searchers.EvolutionSearcher`

**Optinal Parameters**

- *dispatcher*: hypernets.core.Dispatcher, Dispatcher is used to provide different execution modes for search trials, such as in-process mode (`InProcessDispatcher`), distributed parallel mode (`DaskDispatcher`), etc. `InProcessDispatcher` is used by default.
- *callbacks*: list of callback functions or None, optional (default=None), List of callback functions that are applied at each trial. See `hypernets.callbacks` for more information.
- *reward_metric*: str or None, optinal(default=accuracy), Set corresponding metric  according to task type to guide search direction of searcher.
- *task*: str or None, optinal(default=None), Task type(*binary*,*multiclass* or *regression*). If None, inference the type of task automatically
- *param data_cleaner_params*: dict, (default=None), Dictionary of parameters to initialize the `DataCleaner` instance. If None, `DataCleaner` will initialized with default values.
- *param cache_dir*: str or None, (default=None), Path of data cache. If None, uses 'working directory/tmp/cache' as cache dir
- *param clear_cache*: bool, (default=True), Whether clear the cache dir before searching

### search

**Required Parameters**

- *X*: Pandas or Dask DataFrame, feature data for training
- *y*: Pandas or Dask Series, target values for training
- *X_eval*: (Pandas or Dask DataFrame) or None, feature data for evaluation
- *y_eval*: (Pandas or Dask Series) or None, target values for evaluation

**Optinal Parameters**

- *cv*: bool, (default=False), If True, use cross-validation instead of evaluation set reward to guide the search process
- *num_folds*: int, (default=3), Number of cross-validated folds, only valid when cv is true
- *max_trials*: int, (default=10), The upper limit of the number of search trials, the search process stops when the number is exceeded
- **fit_kwargs: dict, parameters for fit method of model


### Use case
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