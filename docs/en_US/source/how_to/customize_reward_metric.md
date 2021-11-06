## How to customize reward_metric in HyperGBM?

To customize a new reward_metric, do the followings:
1. Create a reward_metric  object with following features:
    * `__name__` attribute
    * be callable with argument `y_true` and `y_preds`
    * can be checked in dict or not (override `__hash__` and `__eq__`)
2. Make a sklearn scorer with sklearn.metrics.make_scorer
3. Call make_experiment with your reward_metric and scorer

Example code:

```python
from sklearn.metrics import make_scorer, accuracy_score

from hypergbm import make_experiment
from hypernets.tabular.datasets import dsutils


class MyRewardMetric:
    __name__ = 'foo'

    def __hash__(self):
        return hash(self.__name__)

    def __eq__(self, other):
        return self.__name__ == str(other)

    def __call__(self, y_true, y_preds):
        return accuracy_score(y_true, y_preds)  # replace this line with yours


my_reward_metric = MyRewardMetric()
my_scorer = make_scorer(my_reward_metric, greater_is_better=True, needs_proba=False)

train_data = dsutils.load_adult()
train_data.columns = [f'c{i}' for i in range(14)] + ['target']

exp = make_experiment(train_data.copy(), target='target',
                      reward_metric=my_reward_metric,
                      scorer=my_scorer,
                      max_trials=3,
                      log_level='info')
estimator = exp.run()
print(estimator)

```