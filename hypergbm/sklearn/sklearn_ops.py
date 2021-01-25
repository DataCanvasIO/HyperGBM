# -*- coding:utf-8 -*-
"""

"""
import numpy as np

from hypergbm.pipeline import Pipeline
from hypergbm.sklearn.transformers import SimpleImputer, SafeOneHotEncoder, TruncatedSVD, \
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, SafeOrdinalEncoder, \
    LogStandardScaler
from hypernets.core.ops import ModuleChoice, Optional, Choice
from tabular_toolbox.column_selector import column_object_category_bool, column_number_exclude_timedelta


def categorical_pipeline_simple(impute_strategy='constant', seq_no=0):
    pipeline = Pipeline([
        SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'categorical_imputer_{seq_no}'),
        SafeOrdinalEncoder(name=f'categorical_label_encoder_{seq_no}', dtype='int32')
        # MultiLabelEncoder(name=f'categorical_label_encoder_{seq_no}')
    ],
        columns=column_object_category_bool,
        name=f'categorical_pipeline_simple_{seq_no}',
    )
    return pipeline


def categorical_pipeline_complex(impute_strategy=None, svd_components=3, seq_no=0):
    if impute_strategy is None:
        impute_strategy = Choice(['constant', 'most_frequent'])
    elif isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)
    if isinstance(svd_components, list):
        svd_components = Choice(svd_components)

    def onehot_svd():
        onehot = SafeOneHotEncoder(name=f'categorical_onehot_{seq_no}')
        optional_svd = Optional(TruncatedSVD(n_components=svd_components, name=f'categorical_svd_{seq_no}'),
                                name=f'categorical_optional_svd_{seq_no}',
                                keep_link=True)(onehot)
        return optional_svd

    imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'categorical_imputer_{seq_no}')
    label_encoder = SafeOrdinalEncoder(name=f'categorical_label_encoder_{seq_no}')
    onehot = onehot_svd()
    le_or_onehot_pca = ModuleChoice([label_encoder, onehot], name=f'categorical_le_or_onehot_pca_{seq_no}')
    pipeline = Pipeline([imputer, le_or_onehot_pca],
                        name=f'categorical_pipeline_complex_{seq_no}',
                        columns=column_object_category_bool)
    return pipeline


def numeric_pipeline_simple(impute_strategy='mean', seq_no=0):
    pipeline = Pipeline([
        SimpleImputer(missing_values=np.nan, strategy=impute_strategy,
                      name=f'numeric_imputer_{seq_no}', force_output_as_float=True),
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

    imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'numeric_imputer_{seq_no}',
                            force_output_as_float=True)
    scaler_options = ModuleChoice(
        [
            LogStandardScaler(name=f'numeric_log_standard_scaler_{seq_no}'),
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
