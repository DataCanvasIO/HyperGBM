## 基础应用

本节通过示例讲解如何使用实验进行模型训练，示例中使用数据集`blood`，该数据集中的目标列`Class`，数据如下：
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
本节示例从`hypernets.tabular`中加载该数据集。


### 以缺省配置创建并运行实验

利用工具`make_experiment`可快速创建一个可运行的实验对象，执行该实验对象的`run`方法即可开始训练并得到模型。使用该工具时只有实验数据`train_data`是必须的，其它都是可选项。数据的目标列如果不是`y`的话，需要通过参数`target`设置。

示例代码：
```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class')
estimator = experiment.run()
print(estimator)

```

输出：
```
Pipeline(steps=[('data_clean',
                 DataCleanStep(...),
                ('estimator',
                 GreedyEnsemble(...)])

```

可以看出，训练得到的是一个Pipeline，最终模型是由多个模型构成的融合模型。



如果您的训练数据是csv或parquet格式，而且数据文件的扩展名是“.csv”或“.parquet”的话，可以直接使用文件路径创建实验，make_experiment会自动将数据加载为DataFrame，如：

```python
from hypergbm import make_experiment

train_data = '/path/to/mydata.csv'
experiment = make_experiment(train_data, target='my_target')
estimator = experiment.run()
print(estimator)

```

### 设置最大搜索次数(max_trials)
缺省情况下，`make_experiment`所创建的实验最多搜索10种参数便会停止搜索。实际使用中，建议将最大搜索次数设置为30以上。

```python
from hypergbm import make_experiment

train_data = ...
experiment = make_experiment(train_data, max_trials=50)
...
```

### 交叉验证

可通过参数`cv`指定是否启用交叉验证。当`cv`设置为`False`时表示禁用交叉验证并使用经典的train_test_split方式进行模型训练；当`cv`设置为`True`（缺省）时表示开启交叉验证，折数可通过参数`num_folds`设置（默认：3）。


启用交叉验证的示例代码：
```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', cv=True, num_folds=5)
estimator = experiment.run()
print(estimator)

```

### 指定验证数据集(eval_data)

在禁用交叉验证时，模型训练除了需要训练数据集，还需要评估数据集，您可在`make_experiment`时通过eval_data指定评估集，如：

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

在禁用交叉验证时，如果您未指定`eval_data`，实验对象将从`train_data`中拆分部分数据作为评估集，拆分大小可通过`eval_size`设置，如：

```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', cv=False, eval_size=0.2)
estimator = experiment.run()
print(estimator)

```



### 指定模型的评价指标

使用`make_experiment`创建实验时，分类任务默认的模型评价指标是`accuracy`，回归任务默认的模型评价指标是`rmse`，可通过参数`reward_metric`指定，如：

```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', reward_metric='auc')
estimator = experiment.run()
print(estimator)

```



### 设置搜索次数和早停（Early Stopping）策略

使用`make_experiment`时，可通过参数`early_stopping_round`、`early_stopping_time_limit`、`early_stopping_reward`设置实验的早停策略。

将搜索时间设置为最多3小时的示例代码：
```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', max_trials=300, early_stopping_time_limit=3600 * 3)
estimator = experiment.run()
print(estimator)

```


### 指定搜索算法（Searcher）

HyperGBM通过Hypernets中的搜索算法进行参数搜索，包括：EvolutionSearcher（缺省）、MCTSSearcher、RandomSearch，在`make_experiment`时可通过参数`searcher`指定，可以指定搜索算法的类名(class)、搜索算法的名称（str）。


示例代码：
```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()

experiment = make_experiment(train_data, target='Class', searcher='random')
estimator = experiment.run()
print(estimator)

```


您也可以自己创建searcher对象，然后用所创建的对象创建实验，如：

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


### 模型融合

为了获取较好的模型效果，`make_experiment`创建实验时默认开启了模型融合的特性，并使用效果最好的20个模型进行融合，可通过参数`ensemble_size`指定参与融合的模型的数量。当`ensemble_size`设置为`0`时则表示禁用模型融合。

调整参与融合的模型数量的示例代码：
```python
train_data = ...
experiment = make_experiment(train_data, ensemble_size=10, ...)

```



### 调整日志级别

如果希望在训练过程中看到使用进度信息的话，可通过log_level指定日志级别，可以是`str`或`int`。关于日志级别的详细定义可参考python的`logging`包。 另外，如果将`verbose`设置为`1`的话，可以得到更详细的信息。


将日志级别设置为`INFO`的示例代码如下：
```python
from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

train_data = dsutils.load_blood()
experiment = make_experiment(train_data, target='Class', log_level='INFO', verbose=1)
estimator = experiment.run()
print(estimator)

```

输出：
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


### 实验可视化

如果希望在训练过程中看到实验可视化信息，可通过`webui`打开实验可视化服务，可以是`True`或`False`。
还可以通过`webui_options`参数设置web服务的端口、实验可视化数据持久化目录、是否退出web进程当训练完毕后。

*注意：使用该功能时请确保您是通过`pip install hypergbm[board]` 安装的hypergbm。*


开启基于web的实验可视化示例代码如下：
```python
from sklearn.model_selection import train_test_split

from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils

df = dsutils.load_bank()
df_train, df_test = train_test_split(df, test_size=0.8, random_state=42)

experiment = make_experiment(df_train, target='y', webui=True)
estimator = experiment.run(max_trials=10)

print(estimator)
```

输出：
```console
02-17 19:08:48 I hypernets.t.estimator_detector.py 85 - EstimatorDetector error: GPU Tree Learner was not enabled in this build.
Please recompile with CMake option -DUSE_GPU=1
...
server is running at: 0.0.0.0:8888 
...

02-17 19:08:55 I hypernets.t.metrics.py 153 - calc_score ['auc', 'accuracy'], task=binary, pos_label=yes, classes=['no' 'yes'], average=None
final result:{'auc': 0.8913467492260062, 'accuracy': 0.8910699474702792}
```

这时候您可以打开浏览器访问`http://localhost:8888` 查看实验运行情况：

![web-experiment-visualization](images/web-experiment-visualization.png)


将开启训练可视化，并把端口设置为`8888`，持久化目录设置为`./events`，实验结束后退出web进程设置为`False` 的代码示例如下：
```python
...
webui_options = {
    'event_file_dir': "./events",  # persist experiment running events log to './events'
    'server_port': 8888, # http server port
    'exit_web_server_on_finish': False  # exit http server after experiment finished
}
experiment = make_experiment(df_train, target='y', webui=True, webui_options=webui_options)
...
```
