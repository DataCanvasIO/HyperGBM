
## HyperGBM

HyperGBM是HyperModel的一个具体实现（关于HyperModel的详细信息，请参考[Hypernets](https://github.com/DataCanvasIO/Hypernets)项目）。
HyperGBM类是该项目最主要的接口，可以通过它提供的`search`方法来完成使用特定的`Searcher`(搜索器)在特定`Search Space`(搜索空间)中搜索并返回最佳的模型，简单来说就是一次自动建模的过程。


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

### search

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


### 示例代码
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