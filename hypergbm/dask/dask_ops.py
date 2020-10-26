# -*- coding:utf-8 -*-
"""

"""

from hypernets.core.ops import ModuleChoice, Optional, HyperInput, Real
from hypernets.core.search_space import Choice
from hypernets.core.search_space import HyperSpace
from hypergbm.estimators import LightGBMDaskEstimator, XGBoostDaskEstimator
from tabular_toolbox.column_selector import column_object_category_bool, column_number_exclude_timedelta, column_object
from .dask_transformers import *
from hypergbm.pipeline import Pipeline


def categorical_pipeline_simple(impute_strategy='constant', seq_no=0):
    pipeline = Pipeline([
        SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'categorical_imputer_{seq_no}',
                      fill_value=''),
        MultiLabelEncoder(name=f'categorical_label_encoder_{seq_no}')
    ],
        columns=column_object_category_bool,
        name=f'categorical_pipeline_simple_{seq_no}',
    )
    return pipeline


def categorical_pipeline_complex(impute_strategy=None, svd_components=5, seq_no=0):
    if impute_strategy is None:
        impute_strategy = Choice(['constant', 'most_frequent'])
    elif isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)
    if isinstance(svd_components, list):
        svd_components = Choice(svd_components)

    def onehot_svd():
        onehot = OneHotEncoder(name=f'categorical_onehot_{seq_no}', sparse=False)
        optional_svd = Optional(TruncatedSVD(n_components=svd_components, name=f'categorical_svd_{seq_no}'),
                                name=f'categorical_optional_svd_{seq_no}',
                                keep_link=True)(onehot)
        return optional_svd

    imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'categorical_imputer_{seq_no}',
                            fill_value='')
    label_encoder = MultiLabelEncoder(name=f'categorical_label_encoder_{seq_no}')
    onehot = onehot_svd()
    # onehot = OneHotEncoder(name=f'categorical_onehot_{seq_no}', sparse=False)
    le_or_onehot_pca = ModuleChoice([label_encoder, onehot], name=f'categorical_le_or_onehot_pca_{seq_no}')
    pipeline = Pipeline([imputer, le_or_onehot_pca],
                        name=f'categorical_pipeline_complex_{seq_no}',
                        columns=column_object_category_bool)
    return pipeline


def numeric_pipeline(impute_strategy='mean', seq_no=0):
    pipeline = Pipeline([
        SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'numeric_imputer_{seq_no}',
                      fill_value=0),
        StandardScaler(name=f'numeric_standard_scaler_{seq_no}')
    ],
        columns=column_number_exclude_timedelta,
        name=f'numeric_pipeline_simple_{seq_no}',
    )
    return pipeline


def numeric_pipeline_complex(impute_strategy=None, seq_no=0):
    if impute_strategy is None:
        impute_strategy = Choice(['mean', 'median', 'constant', 'most_frequent'])
    elif isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)
    # reduce_skewness_kurtosis = SkewnessKurtosisTransformer(transform_fn=Choice([np.log, np.log10, np.log1p]))
    # reduce_skewness_kurtosis_optional = Optional(reduce_skewness_kurtosis, keep_link=True,
    #                                             name=f'numeric_reduce_skewness_kurtosis_optional_{seq_no}')

    imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'numeric_imputer_{seq_no}',
                            fill_value=0)
    scaler_options = ModuleChoice(
        [
            StandardScaler(name=f'numeric_standard_scaler_{seq_no}'),
            MinMaxScaler(name=f'numeric_minmax_scaler_{seq_no}'),
            MaxAbsScaler(name=f'numeric_maxabs_scaler_{seq_no}'),
            RobustScaler(name=f'numeric_robust_scaler_{seq_no}')
        ], name=f'numeric_or_scaler_{seq_no}'
    )
    scaler_optional = Optional(scaler_options, keep_link=True, name=f'numeric_scaler_optional_{seq_no}')

    pipeline = Pipeline([imputer, scaler_optional],
                        name=f'numeric_pipeline_complex_{seq_no}',
                        columns=column_number_exclude_timedelta)
    return pipeline


def get_space_num_cat_pipeline_complex(dataframe_mapper_default=False,
                                       lightgbm_fit_kwargs={},
                                       xgb_fit_kwargs={},
                                       catboost_fit_kwargs={}):
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = numeric_pipeline_complex()(input)
        p2 = categorical_pipeline_complex()(input)
        # p2 = categorical_pipeline_simple()(input)
        p3 = DataFrameMapper(default=dataframe_mapper_default, input_df=True, df_out=True,
                             df_out_dtype_transforms=[(column_object, 'int')])([p1, p2])

        lightgbm_init_kwargs = {
            'boosting_type': Choice(['gbdt', 'dart', 'goss']),
            'num_leaves': Choice([11, 31, 101, 301, 501]),
            'learning_rate': Real(0.001, 0.1, step=0.005),
            'n_estimators': 100,
            'max_depth': -1,
            'tree_learner': 'data'  # add for dask
            # subsample_for_bin = 200000, objective = None, class_weight = None,
            #  min_split_gain = 0., min_child_weight = 1e-3, min_child_samples = 20,
        }
        lightgbm_est = LightGBMDaskEstimator(task='binary', fit_kwargs=lightgbm_fit_kwargs, **lightgbm_init_kwargs)

        xgb_init_kwargs = {'tree_method': 'approx'  # add for dask
                           }
        xgb_est = XGBoostDaskEstimator(task='binary', fit_kwargs=xgb_fit_kwargs, **xgb_init_kwargs)

        # catboost_init_kwargs = {
        #     'silent': True
        # }
        # catboost_est = CatBoostEstimator(task='binary', fit_kwargs=catboost_fit_kwargs, **catboost_init_kwargs)
        # or_est = ModuleChoice([lightgbm_est, xgb_est, catboost_est], name='estimator_options')(p3)

        or_est = ModuleChoice([lightgbm_est, xgb_est], name='estimator_options')(p3)

        space.set_inputs(input)
    return space
