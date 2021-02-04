# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypergbm.estimators import LightGBMEstimator, XGBoostEstimator, CatBoostEstimator, HistGBEstimator
from hypergbm.feature_generators import CrossCategorical
from hypergbm.pipeline import DataFrameMapper, Pipeline
from hypergbm.sklearn.sklearn_ops import numeric_pipeline_simple, numeric_pipeline_complex, \
    categorical_pipeline_simple, categorical_pipeline_complex
from hypergbm.sklearn.transformers import FeatureGenerationTransformer
from hypernets.core.ops import ModuleChoice, HyperInput
from hypernets.core.search_space import Choice, Real, Int
from hypernets.core.search_space import HyperSpace
from hypernets.utils import logging
from tabular_toolbox.column_selector import column_object, column_exclude_datetime

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

        self.kwargs = kwargs

    def create_preprocessor(self, hyper_input, **kwargs):
        raise NotImplementedError()

    def create_estimators(self, hyper_input, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        kwargs = _merge_dict(self.kwargs, kwargs)

        space = HyperSpace()
        with space.as_default():
            hyper_input = HyperInput(name='input1')
            self.create_estimators(self.create_preprocessor(hyper_input, **kwargs), **kwargs)
            space.set_inputs(hyper_input)

        return space


class BaseSearchSpaceGenerator(SearchSpaceGenerator):
    @property
    def estimators(self):
        # return dict:  key-->(hyper_estimator_cls, default_init_kwargs, default_fit_kwargs)
        raise NotImplementedError()

    def create_preprocessor(self, hyper_input, **kwargs):
        cat_pipeline_mode = kwargs.pop('cat_pipeline_mode', 'simple')
        dataframe_mapper_default = kwargs.pop('dataframe_mapper_default', False)

        num_pipeline = numeric_pipeline_complex()(hyper_input)
        if cat_pipeline_mode == 'simple':
            cat_pipeline = categorical_pipeline_simple()(hyper_input)
        else:
            cat_pipeline = categorical_pipeline_complex()(hyper_input)

        preprocessor = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                       df_out_dtype_transforms=[(column_object, 'int')])([num_pipeline, cat_pipeline])

        return preprocessor

    def create_estimators(self, hyper_input, **kwargs):
        assert len(self.estimators.keys()) > 0

        creators = [_HyperEstimatorCreator(pairs[0],
                                           init_kwargs=_merge_dict(pairs[1], kwargs.pop(f'{k}_init_kwargs', None)),
                                           fit_kwargs=_merge_dict(pairs[2], kwargs.pop(f'{k}_fit_kwargs', None)))
                    for k, pairs in self.estimators.items()]

        unused = {}
        for k, v in kwargs.items():
            used = False
            for c in creators:
                if k in c.estimator_init_kwargs.keys():
                    c.estimator_init_kwargs[k] = v
                    used = True
                if k in c.estimator_fit_kwargs.keys():
                    c.estimator_fit_kwargs[k] = v
                    used = True
            if not used:
                # logger.warn(f'Unused parameter: {k} = {v}')
                unused[k] = v

        # set unused kwargs as fit_kwargs of all estimators
        if unused:
            for c in creators:
                c.estimator_fit_kwargs.update(unused)

        estimators = [c() for c in creators]
        return ModuleChoice(estimators, name='estimator_options')(hyper_input)


class GeneralSearchSpaceGenerator(BaseSearchSpaceGenerator):
    def __init__(self, enable_lightgbm=True, enable_xgb=True, enable_catboost=True, enable_histgb=True, **kwargs):
        super(GeneralSearchSpaceGenerator, self).__init__(**kwargs)

        self.enable_lightgbm = enable_lightgbm
        self.enable_xgb = enable_xgb
        self.enable_catboost = enable_catboost
        self.enable_histgb = enable_histgb

    @property
    def default_xgb_init_kwargs(self):
        return {
            'booster': Choice(['gbtree', 'dart']),
            'max_depth': Choice([3, 5, 7, 10]),
            'n_estimators': Choice([10, 30, 50, 100, 200, 300]),
            'learning_rate': Choice([0.001, 0.01, 0.5, 0.1]),
            'min_child_weight': Choice([1, 5, 10]),
            'gamma': Choice([0.5, 1, 1.5, 2, 5]),
            'reg_alpha': Choice([0.001, 0.01, 0.1, 1, 10, 100]),
            'reg_lambda': Choice([0.001, 0.01, 0.1, 0.5, 1]),
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
            'l2_regularization': Choice([1e-10, 1e-8, 1e-6, 1e-5, 1e-3, 0.01, 0.1, 1])
        }

    @property
    def default_histgb_fit_kwargs(self):
        return {}

    @property
    def estimators(self):
        r = {}

        if self.enable_lightgbm:
            r['lightgbm'] = (LightGBMEstimator, self.default_lightgbm_init_kwargs, self.default_lightgbm_fit_kwargs)
        if self.enable_xgb:
            r['xgb'] = (XGBoostEstimator, self.default_xgb_init_kwargs, self.default_xgb_fit_kwargs)
        if self.enable_catboost:
            r['catboost'] = (CatBoostEstimator, self.default_catboost_init_kwargs, self.default_catboost_fit_kwargs)
        if self.enable_histgb:
            r['histgb'] = (HistGBEstimator, self.default_histgb_init_kwargs, self.default_histgb_fit_kwargs)

        return r


search_space_general = GeneralSearchSpaceGenerator(enable_histgb=False, n_estimators=200)


def search_space_general_removed(dataframe_mapper_default=False,
                                 lightgbm_fit_kwargs=None,
                                 xgb_fit_kwargs=None,
                                 catboost_fit_kwargs=None,
                                 histgb_fit_kwargs=None,
                                 cat_pipeline_mode='simple',
                                 class_balancing=None,
                                 n_esitimators=200,
                                 **kwargs):
    """
    A general search space function

    :param dataframe_mapper_default: bool, default=False
        Param for DataFrameMapper, default transformer to apply to the columns not explicitly selected in the mapper.
        If False (default), discard them.  If None, pass them through untouched. Any other transformer will be applied
        to all the unselected columns as a whole,  taken as a 2d-array.
    :param lightgbm_fit_kwargs: dict, default=None
        kwargs for calling `fit` method of lightgbm
    :param xgb_fit_kwargs: dict, default=None
        kwargs for calling `fit` method of xgb
    :param catboost_fit_kwargs: dict, default=None
        kwargs for calling `fit` method of catboost
    :param histgb_fit_kwargs: dict, default=None
        kwargs for calling `fit` method of histgb
    :param cat_pipeline_mode: 'simple' or None, default='simple'
        Mode of categorical pipeline
    :param class_balancing: str or None, default=None
        Strategy for class balancing.
            - 'ClassWeight'
            - 'RandomOverSampling'
            - 'SMOTE'
            - 'ADASYN'
            - 'RandomUnderSampling'
            - 'NearMiss'
            - 'TomeksLinks'
            - 'EditedNearestNeighbours'
    :param n_esitimators: int, default=200
        Number of estimators
    :param kwargs:
    :return:
    """
    if lightgbm_fit_kwargs is None:
        lightgbm_fit_kwargs = {}
    if xgb_fit_kwargs is None:
        xgb_fit_kwargs = {}
    if catboost_fit_kwargs is None:
        catboost_fit_kwargs = {}
    if histgb_fit_kwargs is None:
        histgb_fit_kwargs = {}

    lightgbm_fit_kwargs.update(kwargs)
    xgb_fit_kwargs.update(kwargs)
    catboost_fit_kwargs.update(kwargs)
    histgb_fit_kwargs.update(kwargs)

    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        num_pipeline = numeric_pipeline_complex()(input)
        if cat_pipeline_mode == 'simple':
            cat_pipeline = categorical_pipeline_simple()(input)
        else:
            cat_pipeline = categorical_pipeline_complex()(input)

        union_pipeline = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                         df_out_dtype_transforms=[(column_object, 'int')])([num_pipeline, cat_pipeline])

        # polynomial = Pipeline([PolynomialFeatures(
        #     degree=Choice([2, 3], name='poly_fea_degree'),
        #     interaction_only=Bool(name='poly_fea_interaction_only'),
        #     include_bias=Bool(name='poly_fea_include_bias'))], columns=column_number)
        # poly_dm = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
        #                           df_out_dtype_transforms=[(column_object, 'int')])([polynomial])
        # poly_pipeline = Pipeline(module_list=[poly_dm])(union_pipeline)

        lightgbm_init_kwargs = {
            'n_estimators': n_esitimators,  # Choice([10, 30, 50, 100, 200, 300, 500]),
            'boosting_type': Choice(['gbdt', 'dart', 'goss']),
            'num_leaves': Int(15, 513, 5),
            'learning_rate': Choice([0.001, 0.01, 0.5, 0.1]),
            'max_depth': Choice([3, 5, 7, 10]),
            'reg_alpha': Choice([0.001, 0.01, 0.1, 1, 10, 100]),
            'reg_lambda': Choice([0.001, 0.01, 0.1, 0.5, 1]),
            'class_balancing': class_balancing,
        }

        xgb_init_kwargs = {
            'booster': Choice(['gbtree', 'dart']),
            'max_depth': Choice([3, 5, 7, 10]),
            'n_estimators': n_esitimators,  # Choice([10, 30, 50, 100, 200, 300]),
            'learning_rate': Choice([0.001, 0.01, 0.5, 0.1]),
            'min_child_weight': Choice([1, 5, 10]),
            'gamma': Choice([0.5, 1, 1.5, 2, 5]),
            'reg_alpha': Choice([0.001, 0.01, 0.1, 1, 10, 100]),
            'reg_lambda': Choice([0.001, 0.01, 0.1, 0.5, 1]),
            'class_balancing': class_balancing,
        }

        catboost_init_kwargs = {
            'silent': True,
            'n_estimators': n_esitimators,
            'depth': Choice([3, 5, 7, 10]),
            'learning_rate': Choice([0.001, 0.01, 0.5, 0.1]),
            'l2_leaf_reg': Choice([None, 2, 10, 20, 30]),
            'class_balancing': class_balancing,
        }

        histgb_init_kwargs = {
            'learning_rate': Choice([0.01, 0.1, 0.2, 0.5, 0.8, 1]),
            'min_samples_leaf': Choice([10, 20, 50, 80, 100, 150, 180, 200]),
            'max_leaf_nodes': Int(15, 513, 5),
            'l2_regularization': Choice([1e-10, 1e-8, 1e-6, 1e-5, 1e-3, 0.01, 0.1, 1])
        }

        lightgbm_est = LightGBMEstimator(fit_kwargs=lightgbm_fit_kwargs, **lightgbm_init_kwargs)
        xgb_est = XGBoostEstimator(fit_kwargs=xgb_fit_kwargs, **xgb_init_kwargs)
        catboost_est = CatBoostEstimator(fit_kwargs=catboost_fit_kwargs, **catboost_init_kwargs)
        # histgb_est = HistGBEstimator(fit_kwargs=histgb_fit_kwargs, **histgb_init_kwargs)
        # ModuleChoice([xgb_est, lightgbm_est, catboost_est], name='estimator_options')(union_pipeline)
        ModuleChoice([xgb_est, lightgbm_est, catboost_est], name='estimator_options')(union_pipeline)
        space.set_inputs(input)
    return space


def search_space_feature_gen(dataframe_mapper_default=False,
                             early_stopping_rounds=None,
                             lightgbm_fit_kwargs=None,
                             xgb_fit_kwargs=None,
                             catboost_fit_kwargs=None,
                             task=None,
                             **kwargs):
    if lightgbm_fit_kwargs is None:
        lightgbm_fit_kwargs = {}
    if xgb_fit_kwargs is None:
        xgb_fit_kwargs = {}
    if catboost_fit_kwargs is None:
        catboost_fit_kwargs = {}

    lightgbm_fit_kwargs.update(kwargs)
    xgb_fit_kwargs.update(kwargs)
    catboost_fit_kwargs.update((kwargs))

    if early_stopping_rounds is not None:
        lightgbm_fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
        xgb_fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
        catboost_fit_kwargs['early_stopping_rounds'] = early_stopping_rounds

    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        num_pipeline = numeric_pipeline_complex()
        cat_pipeline = categorical_pipeline_simple()
        union_pipeline = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                                         df_out_dtype_transforms=[(column_object, 'int')])([num_pipeline, cat_pipeline])

        lightgbm_init_kwargs = {
            'boosting_type': Choice(['gbdt', 'dart', 'goss']),
            'num_leaves': Choice([3, 5]),
            'learning_rate': 0.1,
            'n_estimators': Choice([10, 30, 50]),
            'max_depth': Choice([3, 5]),
            'reg_alpha': Choice([1e-2, 0.1, 1, 100]),
            'reg_lambda': Choice([1e-2, 0.1, 1, 100]),
            # 'class_weight': 'balanced',
            # subsample_for_bin = 200000, objective = None, class_weight = None,
            #  min_split_gain = 0., min_child_weight = 1e-3, min_child_samples = 20,
        }
        lightgbm_est = LightGBMEstimator(fit_kwargs=lightgbm_fit_kwargs, **lightgbm_init_kwargs)
        xgb_init_kwargs = {
            'max_depth': Choice([3, 5]),
            'n_estimators': Choice([10, 30, 50]),
            'learning_rate': 0.1,
            'min_child_weight': Choice([1, 5, 10]),
            'gamma': Choice([0.5, 1, 1.5, 2, 5]),
            'reg_alpha': Choice([1e-2, 0.1, 1, 100]),
            'reg_lambda': Choice([1e-2, 0.1, 1, 100]),
            'subsample': Choice([0.6, 0.8, 1.0]),
            'colsample_bytree': Choice([0.6, 0.8, 1.0]),
            # 'scale_pos_weight': Int(1,5,1),
        }
        xgb_est = XGBoostEstimator(fit_kwargs=xgb_fit_kwargs, **xgb_init_kwargs)

        catboost_init_kwargs = {
            'silent': True,
            'depth': Choice([3, 5]),
            'learning_rate': Real(0.001, 0.1, step=0.005),
            'iterations': Choice([30, 50, 100]),
            'l2_leaf_reg': Choice([1, 3, 5, 7, 9]),
            # 'class_weights': [0.59,3.07],
            # 'border_count': Choice([5, 10, 20, 32, 50, 100, 200]),
        }
        catboost_est = CatBoostEstimator(fit_kwargs=catboost_fit_kwargs, **catboost_init_kwargs)
        cross_cat = CrossCategorical()
        feature_gen = FeatureGenerationTransformer(task=task, trans_primitives=[cross_cat, 'divide_numeric'])
        full_pipeline = Pipeline([feature_gen, union_pipeline],
                                 name=f'feature_gen_and_preprocess',
                                 columns=column_exclude_datetime)(input)
        # full_dfm = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True)(full_pipeline)
        ModuleChoice([lightgbm_est, xgb_est, catboost_est], name='estimator_options')(full_pipeline)
        space.set_inputs(input)
    return space


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


def search_space_compact():
    raise NotImplementedError


def search_space_universal():
    raise NotImplementedError
