## 如何在HyperGBM中自定义评价指标?



您可通过如下方式其定义一个新的模型评价指标：

1. 定义一个参数 为 `y_true` and `y_preds` 的函数 (not lambda) 
2. 使用您定义的函数创建一个 sklearn scorer
3. 使用您定义的评价函数和scorer创建实验 



参考示例：

```python
from sklearn.metrics import make_scorer, accuracy_score

from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils


def foo(y_true, y_preds):
    return accuracy_score(y_true, y_preds)  # replace this line with yours

my_scorer = make_scorer(foo, greater_is_better=True, needs_proba=False)

train_data = dsutils.load_adult()
train_data.columns = [f'c{i}' for i in range(14)] + ['target']

exp = make_experiment(train_data.copy(), target='target',
                      reward_metric=foo,
                      scorer=my_scorer,
                      max_trials=3,
                      log_level='info')
estimator = exp.run()
print(estimator)

```