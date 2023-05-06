# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypergbm.cfg import HyperGBMCfg as cfg
from hypergbm.estimators import LightGBMEstimator, XGBoostEstimator, CatBoostEstimator, HistGBEstimator
from hypergbm.estimators import detect_lgbm_gpu
from hypergbm.sklearn.sklearn_ops import numeric_pipeline_simple, numeric_pipeline_complex, \
    categorical_pipeline_simple, categorical_pipeline_complex, \
    datetime_pipeline_simple, text_pipeline_simple
from hypernets.core import randint
from hypernets.core.ops import ModuleChoice, HyperInput
from hypernets.core.search_space import HyperSpace, Choice, Int, Real
from hypernets.pipeline.base import DataFrameMapper
from hypernets.tabular.column_selector import column_object
from hypernets.pipeline.transformers import FeatureImportanceSelection
from hypernets.tabular import column_selector

from hypernets.utils import logging, get_params

logger = logging.get_logger(__name__)


def _merge_dict(*args):
    d = {}
    for a in args:
        if isinstance(a, dict):
            d.update(a)
    return d


class _HyperEstimatorCreator(object):
    def __init__(self, cls, init_kwargs, fit_kwargs):
        super(_HyperEstimatorCreator, self).__init__()

        self.estimator_cls = cls
        self.estimator_fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self.estimator_init_kwargs = init_kwargs if init_kwargs is not None else {}

    def __call__(self, *args, **kwargs):
        return self.estimator_cls(self.estimator_fit_kwargs, **self.estimator_init_kwargs)


class SearchSpaceGenerator(object):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.options = kwargs

    def create_preprocessor(self, hyper_input, options):
        raise NotImplementedError()

    def create_feature_selection(self, hyper_input, options):
        return None

    def create_estimators(self, hyper_input, options):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        space = HyperSpace()
        with space.as_default():
            options = _merge_dict(self.options, kwargs)
            hyper_input = HyperInput(name='input1')

            if "importances" in options and options["importances"] is not None:
                importances = options.pop("importances")
                ss = self.create_feature_selection(hyper_input, importances)
                self.create_estimators(self.create_preprocessor(ss, options), options)
            else:
                self.create_estimators(self.create_preprocessor(hyper_input, options), options)
            space.set_inputs(hyper_input)

        return space

    def __repr__(self):
        params = get_params(self)
        params.update(self.options)
        repr_ = ', '.join(['%s=%r' % (k, v) for k, v in params.items()])
        return f'{type(self).__name__}({repr_})'


class BaseSearchSpaceGenerator(SearchSpaceGenerator):
    @property
    def estimators(self):
        # return dict:  key-->(hyper_estimator_cls, default_init_kwargs, default_fit_kwargs)
        raise NotImplementedError()

    def create_preprocessor(self, hyper_input, options):
        cat_pipeline_mode = options.pop('cat_pipeline_mode', cfg.category_pipeline_mode)
        num_pipeline_mode = options.pop('num_pipeline_mode', cfg.numeric_pipeline_mode)
        dataframe_mapper_default = options.pop('dataframe_mapper_default', False)

        pipelines = []
        # text
        if cfg.text_pipeline_enabled:
            pipelines.append(text_pipeline_simple()(hyper_input))

        # category
        if cfg.category_pipeline_enabled:
            if cat_pipeline_mode == 'simple':
                pipelines.append(categorical_pipeline_simple()(hyper_input))
            else:
                pipelines.append(categorical_pipeline_complex()(hyper_input))

        # datetime
        if cfg.datetime_pipeline_enabled:
            pipelines.append(datetime_pipeline_simple()(hyper_input))

        # numeric
        if num_pipeline_mode == 'simple':
            pipelines.append(numeric_pipeline_simple()(hyper_input))
        else:
            pipelines.append(numeric_pipeline_complex()(hyper_input))

        preprocessor = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                       df_out_dtype_transforms=[(column_object, 'int')])(pipelines)

        return preprocessor

    def create_estimators(self, hyper_input, options):
        estimators = self.estimators
        assert len(estimators.keys()) > 0

        creators = [_HyperEstimatorCreator(pairs[0],
                                           init_kwargs=_merge_dict(pairs[1], options.pop(f'{k}_init_kwargs', None)),
                                           fit_kwargs=_merge_dict(pairs[2], options.pop(f'{k}_fit_kwargs', None)))
                    for k, pairs in estimators.items()]

        unused = {}
        for k, v in options.items():
            used = False
            for c in creators:
                if k in c.estimator_init_kwargs.keys():
                    if isinstance(v, (list, tuple)):
                        c.estimator_init_kwargs[k] = Choice(list(v))
                    else:
                        c.estimator_init_kwargs[k] = v
                    used = True
                if k in c.estimator_fit_kwargs.keys():
                    if isinstance(v, (list, tuple)):
                        c.estimator_fit_kwargs[k] = Choice(list(v))
                    else:
                        c.estimator_fit_kwargs[k] = v
                    used = True
            if not used:
                # logger.warn(f'Unused parameter: {k} = {v}')
                unused[k] = v

        # set unused options as fit_kwargs of all estimators
        if unused:
            for c in creators:
                c.estimator_fit_kwargs.update(unused)

        estimators = [c() for c in creators]
        return ModuleChoice(estimators, name='estimator_options')(hyper_input)

    def create_feature_selection(self, hyper_input, importances, seq_no=0):
        from hypernets.pipeline.base import Pipeline
        selection = FeatureImportanceSelection(name=f'feature_importance_selection_{seq_no}',
                                               importances=importances,
                                               quantile=Real(0, 1, step=0.1))
        pipeline = Pipeline([selection],
                            name=f'feature_selection_{seq_no}',
                            columns=column_selector.column_all)(hyper_input)

        preprocessor = DataFrameMapper(default=False, input_df=True, df_out=True,
                                       df_out_dtype_transforms=None)([pipeline])
        return preprocessor


class GeneralSearchSpaceGenerator(BaseSearchSpaceGenerator):
    def __init__(self, enable_lightgbm=True, enable_xgb=True, enable_catboost=True, enable_histgb=False, **kwargs):
        super(GeneralSearchSpaceGenerator, self).__init__(**kwargs)

        self.enable_lightgbm = enable_lightgbm
        self.enable_xgb = enable_xgb
        self.enable_catboost = enable_catboost
        self.enable_histgb = enable_histgb

    @property
    def default_xgb_init_kwargs(self):
        return {
            # 'booster': Choice(['gbtree', 'dart']),
            'max_depth': Choice([3, 5, 7, 10]),
            'n_estimators': Choice([10, 30, 50, 100, 200, 300]),
            'learning_rate': Choice([0.001, 0.01, 0.5, 0.1]),
            'min_child_weight': Choice([1, 5, 10]),
            'gamma': Choice([0.5, 1, 1.5, 2, 5]),
            'reg_alpha': Choice([0.001, 0.01, 0.1, 1, 10, 100]),
            'reg_lambda': Choice([0.001, 0.01, 0.1, 0.5, 1]),
            'random_state': randint(),
            'class_balancing': None,
        }

    @property
    def default_xgb_fit_kwargs(self):
        return {}

    @property
    def default_lightgbm_init_kwargs(self):
        return {
            'n_estimators': Choice([10, 30, 50, 100, 200, 300]),
            'boosting_type': Choice(['gbdt', 'dart', 'goss']),
            'num_leaves': Int(15, 513, 5),
            'learning_rate': Choice([0.001, 0.01, 0.5, 0.1]),
            'max_depth': Choice([3, 5, 7, 10]),
            'reg_alpha': Choice([0.001, 0.01, 0.1, 1, 10, 100]),
            'reg_lambda': Choice([0.001, 0.01, 0.1, 0.5, 1]),
            'random_state': randint(),
            'class_balancing': None,
        }

    @property
    def default_lightgbm_fit_kwargs(self):
        return {}

    @property
    def default_catboost_init_kwargs(self):
        return {
            'silent': True,
            'n_estimators': Choice([10, 30, 50, 100, 200, 300]),
            'depth': Choice([3, 5, 7, 10]),
            'learning_rate': Choice([0.001, 0.01, 0.5, 0.1]),
            'l2_leaf_reg': Choice([None, 2, 10, 20, 30]),
            'random_state': randint(),
            'class_balancing': None,
        }

    @property
    def default_catboost_fit_kwargs(self):
        return {}

    @property
    def default_histgb_init_kwargs(self):
        return {
            'learning_rate': Choice([0.01, 0.1, 0.2, 0.5, 0.8, 1]),
            'min_samples_leaf': Choice([10, 20, 50, 80, 100, 150, 180, 200]),
            'max_leaf_nodes': Int(15, 513, 5),
            'l2_regularization': Choice([1e-10, 1e-8, 1e-6, 1e-5, 1e-3, 0.01, 0.1, 1]),
            'random_state': randint(),
            'class_balancing': None,
        }

    @property
    def default_histgb_fit_kwargs(self):
        return {}

    lightgbm_estimator_cls = LightGBMEstimator
    xgboost_estimator_cls = XGBoostEstimator
    catboost_estimator_cls = CatBoostEstimator
    histgb_estimator_cls = HistGBEstimator

    @property
    def estimators(self):
        r = {}

        if self.enable_lightgbm:
            r['lightgbm'] = (self.lightgbm_estimator_cls,
                             self.default_lightgbm_init_kwargs,
                             self.default_lightgbm_fit_kwargs)
        if self.enable_xgb:
            r['xgb'] = (self.xgboost_estimator_cls,
                        self.default_xgb_init_kwargs,
                        self.default_xgb_fit_kwargs)
        if self.enable_catboost:
            r['catboost'] = (self.catboost_estimator_cls,
                             self.default_catboost_init_kwargs,
                             self.default_catboost_fit_kwargs)
        if self.enable_histgb:
            r['histgb'] = (self.histgb_estimator_cls,
                           self.default_histgb_init_kwargs,
                           self.default_histgb_fit_kwargs)

        return r


_default_options = dict(enable_lightgbm=cfg.estimator_lightgbm_enabled,
                        enable_xgb=cfg.estimator_xgboost_enabled,
                        enable_catboost=cfg.estimator_catboost_enabled,
                        enable_histgb=cfg.estimator_histgb_enabled,
                        n_estimators=200)

_default_options_with_gpu = _default_options.copy()
if _default_options_with_gpu['enable_lightgbm'] and detect_lgbm_gpu():
    _default_options_with_gpu['lightgbm_init_kwargs'] = {'device': 'GPU'}
if _default_options_with_gpu['enable_xgb']:
    _default_options_with_gpu['xgb_init_kwargs'] = {'tree_method': 'gpu_hist'}
if _default_options_with_gpu['enable_catboost']:
    _default_options_with_gpu['catboost_init_kwargs'] = {'task_type': 'GPU'}

_default_options_with_class_balancing = _default_options.copy()
_default_options_with_class_balancing['class_balancing'] = True

search_space_general = GeneralSearchSpaceGenerator(**_default_options)
search_space_general_gpu = GeneralSearchSpaceGenerator(**_default_options_with_gpu)
search_space_general_with_class_balancing = GeneralSearchSpaceGenerator(**_default_options_with_class_balancing)


def search_space_one_trial(dataframe_mapper_default=False,
                           eval_set=None,
                           early_stopping_rounds=None,
                           lightgbm_fit_kwargs=None):
    if lightgbm_fit_kwargs is None:
        lightgbm_fit_kwargs = {}

    if eval_set is not None:
        lightgbm_fit_kwargs['eval_set'] = eval_set

    if early_stopping_rounds is not None:
        lightgbm_fit_kwargs['early_stopping_rounds'] = early_stopping_rounds

    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        num_pipeline = numeric_pipeline_simple()(input)
        cat_pipeline = categorical_pipeline_simple()(input)
        union_pipeline = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                         df_out_dtype_transforms=[(column_object, 'int')])([num_pipeline, cat_pipeline])

        lightgbm_init_kwargs = {
            'boosting_type': 'gbdt',
            'num_leaves': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 5,
            'class_weight': 'balanced',
        }
        lightgbm_est = LightGBMEstimator(task='binary', fit_kwargs=lightgbm_fit_kwargs, **lightgbm_init_kwargs)
        lightgbm_est(union_pipeline)
        space.set_inputs(input)
    return space
