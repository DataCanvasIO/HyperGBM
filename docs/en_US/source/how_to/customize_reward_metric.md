## How to customize reward_metric in HyperGBM?

To customize a new reward_metric, do the followings:
1. Define a function (not lambda) with argument `y_true` and `y_preds`
2. Make a sklearn scorer with your function
3. Call make_experiment with your reward_metric and scorer

Example code:

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