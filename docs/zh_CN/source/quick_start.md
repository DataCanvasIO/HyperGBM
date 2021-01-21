## 快速开始

本章介绍hypergbm主要功能，假设您已经知道机器学习的基本知识（加载数据、模型训练、预测、评估等），如果您还没安装请参照[安装文档](installation.md)来安装HyperGBM。
您可以使用python api和命令行工具来使用HyperGBM。

### 通过api训练模型
本节将使用数据[Bank Marketing](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing) 演示如何使用HyperGBM训练一个二分类模型。

可以使用`tabular_toolbox`提供的工具类来读取Bank Marketing 数据集： 
```pydocstring
>>> from tabular_toolbox.datasets import dsutils
>>> df = dsutils.load_bank()
>>> df[:3]
   id  age         job  marital  education default  balance housing loan   contact  day month  duration  campaign  pdays  previous poutcome   y
0   0   30  unemployed  married    primary      no     1787      no   no  cellular   19   oct        79         1     -1         0  unknown  no
1   1   33    services  married  secondary      no     4789     yes  yes  cellular   11   may       220         1    339         4  failure  no
2   2   35  management   single   tertiary      no     1350     yes   no  cellular   16   apr       185         1    330         1  failure  no
```

接着我们将数据拆分为训练集和测试集，分别用来训练模型和验证最终模型的效果：
```pydocstring
>>> from sklearn.model_selection import train_test_split
>>> y = df.pop('y')  # target col is "y"
>>> X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=9527)
```

HyperGBM提供了多种搜索策略，这里定义使用随机搜索方法在内置的搜索空间里训练：
```pydocstring
>>> from hypernets.searchers import RandomSearcher
>>> from hypernets.core import OptimizeDirection
>>> from hypergbm.search_space import search_space_general
>>> rs = RandomSearcher(space_fn=search_space_general,
...                     optimize_direction=OptimizeDirection.Maximize)
>>> rs
<hypernets.searchers.random_searcher.RandomSearcher object at 0x10e5b9850>
```
参数`space_fn`用来指定搜索空间，`search_space_general` 是hypergbm提供的默认搜索空间；
参数`optimize_direction` 用来指定优化的方向，训练模型时，对于二分类任务使用`auc`指标，这里设置为`OptimizeDirection.Maximize`表示该指标的值越大越好。

接着使用Experiment接口来训练模型：
```
>>> from hypergbm import HyperGBM, CompeteExperiment
>>> hk = HyperGBM(rs, reward_metric='auc', cache_dir=f'hypergbm_cache', callbacks=[])
>>> experiment = CompeteExperiment(hk, X_train, y_train, X_test=X_test)
19:19:31 I hypergbm.experiment.py 714 - create experiment with ['data_clean', 'drift_detected', 'base_search_and_train']
>>> pipeline = experiment.run(use_cache=True, max_trials=2)
...
   Trial No.    Reward   Elapsed                      Space Vector
0          1  0.994731  8.490173             [2, 0, 1, 2, 0, 0, 0]
1          2  0.983054  4.980630  [1, 2, 1, 2, 215, 3, 0, 0, 4, 3]
>>> pipeline
Pipeline(steps=[('data_clean',
                 DataCleanStep(data_cleaner_args={}, name='data_clean',
                               random_state=9527)),
                ('drift_detected', DriftDetectStep(name='drift_detected')),
                ('base_search_and_train',
                 BaseSearchAndTrainStep(name='base_search_and_train',
                                        scorer=make_scorer(log_loss, greater_is_better=False, needs_proba=True))),
                ('estimator',
                 <tabular_toolbox.ensemble.voting.GreedyEnsemble object at 0x1a24ca00d0>)])

```

训练实验结束后我们来用测试集评估一下效果：
```pydocstring
>>> y_proba = pipeline.predict_proba(X_test)
>>> metrics.roc_auc_score(y_test, y_proba[:, 1])
0.9956872713648863
```

### 通过命令行训练模型

HyperGBM 也提供了命令行工具来训练模型和预测数据，查看命令行帮助：
```
hypergm -h

usage: hypergbm [-h] --train_file TRAIN_FILE [--eval_file EVAL_FILE]
                [--eval_size EVAL_SIZE] [--test_file TEST_FILE] --target
                TARGET [--pos_label POS_LABEL] [--max_trials MAX_TRIALS]
                [--model_output MODEL_OUTPUT]
                [--prediction_output PREDICTION_OUTPUT] [--searcher SEARCHER]
...
```

同样以训练数据Bank Marketing为例子，我们先将数据集拆分成训练集和测试集并生成csv文件：
```pydocstring
>>> from tabular_toolbox.datasets import dsutils
>>> from sklearn.model_selection import train_test_split
>>> df = dsutils.load_bank()
>>> df_train, df_test = train_test_split(df, test_size=0.3, random_state=9527)
>>> df_train.to_csv('bank_train.csv', index=None)
>>> df_test.to_csv('bank_test.csv', index=None)
```

将生成的csv文件作为训练命令参数，执行命令：
```shell script
hypergbm --train_file=bank_train.csv --test_file=bank_test.csv --target=y --pos_label=yes --model_output=model.pkl prediction_output=bank_predict.csv

...
   Trial No.    Reward    Elapsed                       Space Vector
0         10  1.000000  64.206514  [0, 0, 1, 3, 2, 1, 2, 1, 2, 2, 3]
1          7  0.999990   2.433192   [1, 1, 1, 2, 215, 0, 2, 3, 0, 4]
2          4  0.999950  37.057761  [0, 3, 1, 0, 2, 1, 3, 1, 3, 4, 3]
3          9  0.967292   9.977973   [1, 0, 1, 1, 485, 2, 2, 5, 3, 0]
4          1  0.965844   4.304114    [1, 2, 1, 1, 60, 2, 2, 5, 0, 1]
```

训练结束后模型会保持到`model.pkl`文件，对测试集的预测结果会保存到`bank_predict.csv`中。
