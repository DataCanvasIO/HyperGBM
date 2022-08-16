import shap
from sklearn.model_selection import train_test_split

from hypergbm import HyperGBM
from hypergbm.hyper_gbm import HyperGBMShapExplainer
from hypergbm.search_space import GeneralSearchSpaceGenerator
from hypergbm.tests import test_output_dir
from hypernets.core.callbacks import SummaryCallback, FileLoggingCallback
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.tabular.datasets import dsutils

df = dsutils.load_bank()
df.drop(['id'], axis=1, inplace=True)
search_space_general_c = GeneralSearchSpaceGenerator(enable_lightgbm=False, enable_xgb=False, enable_catboost=True, enable_histgb=False,)

rs = RandomSearcher(search_space_general_c, optimize_direction=OptimizeDirection.Maximize)
hk = HyperGBM(rs, task='binary', reward_metric='accuracy',
              callbacks=[SummaryCallback(), FileLoggingCallback(rs, output_dir=f'{test_output_dir}/hyn_logs')])

df = dsutils.load_bank()
df.drop(['id'], axis=1, inplace=True)
X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
y_train = X_train.pop('y')
y_test = X_test.pop('y')


hk.search(X_train, y_train, X_test, y_test, cv=True, num_folds=3, max_trials=2)
best_trial = hk.get_best_trial()

best_estimator = best_trial.get_model()

# estimator = hk.final_train(best_trial.space_sample, X_train, y_train)

explainer = HyperGBMShapExplainer(best_estimator)

shap_values = explainer(X_test)


shap.plots.waterfall(shap_values[0][0])




