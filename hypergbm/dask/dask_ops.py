# -*- coding:utf-8 -*-
"""

"""

import numpy as np

from hypergbm.pipeline import Pipeline
from hypernets.core.ops import ModuleChoice, Optional
from hypernets.core.search_space import Choice
from hypernets.utils import logging
from tabular_toolbox.column_selector import column_object_category_bool, column_number_exclude_timedelta
from . import dask_transformers as tf

logger = logging.get_logger(__name__)


def default_transformer_decorator(transformer):
    return transformer


def categorical_pipeline_simple(impute_strategy='constant', seq_no=0,
                                decorate=default_transformer_decorator):
    pipeline = Pipeline([
        decorate(tf.SimpleImputer(missing_values=np.nan, strategy=impute_strategy,
                                  name=f'categorical_imputer_{seq_no}', fill_value='')),
        decorate(tf.SafeOrdinalEncoder(name=f'categorical_label_encoder_{seq_no}'))
    ],
        columns=column_object_category_bool,
        name=f'categorical_pipeline_simple_{seq_no}',
    )
    return pipeline


def categorical_pipeline_complex(impute_strategy=None, svd_components=5, seq_no=0,
                                 decorate=default_transformer_decorator):
    if impute_strategy is None:
        impute_strategy = Choice(['constant', 'most_frequent'])
    elif isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)
    if isinstance(svd_components, list):
        svd_components = Choice(svd_components)

    def onehot_svd():
        onehot = decorate(tf.SafeOneHotEncoder(name=f'categorical_onehot_{seq_no}', sparse=False))
        svd = decorate(tf.TruncatedSVD(n_components=svd_components, name=f'categorical_svd_{seq_no}'))
        optional_svd = Optional(svd, name=f'categorical_optional_svd_{seq_no}', keep_link=True)(onehot)
        return optional_svd

    imputer = decorate(tf.SimpleImputer(missing_values=np.nan, strategy=impute_strategy,
                                        name=f'categorical_imputer_{seq_no}', fill_value=''))

    # label_encoder = decorate(tf.MultiLabelEncoder(name=f'categorical_label_encoder_{seq_no}'))
    label_encoder = decorate(tf.SafeOrdinalEncoder(name=f'categorical_label_encoder_{seq_no}'))

    onehot = onehot_svd()

    # onehot = SafeOneHotEncoder(name=f'categorical_onehot_{seq_no}', sparse=False)
    le_or_onehot_pca = ModuleChoice([label_encoder, onehot], name=f'categorical_le_or_onehot_svd_{seq_no}')

    pipeline = Pipeline([imputer, le_or_onehot_pca],
                        name=f'categorical_pipeline_complex_{seq_no}',
                        columns=column_object_category_bool)
    return pipeline


def numeric_pipeline(impute_strategy='mean', seq_no=0, decorate=default_transformer_decorator):
    pipeline = Pipeline([
        decorate(tf.SimpleImputer(missing_values=np.nan, strategy=impute_strategy,
                                  name=f'numeric_imputer_{seq_no}', fill_value=0)),
        decorate(tf.StandardScaler(name=f'numeric_standard_scaler_{seq_no}'))
    ],
        columns=column_number_exclude_timedelta,
        name=f'numeric_pipeline_simple_{seq_no}',
    )
    return pipeline


def numeric_pipeline_complex(impute_strategy=None, seq_no=0, decorate=default_transformer_decorator):
    if impute_strategy is None:
        impute_strategy = Choice(['mean', 'median', 'constant', 'most_frequent'])
    elif isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)
    # reduce_skewness_kurtosis = SkewnessKurtosisTransformer(transform_fn=Choice([np.log, np.log10, np.log1p]))
    # reduce_skewness_kurtosis_optional = Optional(reduce_skewness_kurtosis, keep_link=True,
    #                                             name=f'numeric_reduce_skewness_kurtosis_optional_{seq_no}')

    imputer = decorate(tf.SimpleImputer(missing_values=np.nan, strategy=impute_strategy,
                                        name=f'numeric_imputer_{seq_no}', fill_value=0))
    scaler_options = ModuleChoice(
        [
            decorate(tf.StandardScaler(name=f'numeric_standard_scaler_{seq_no}')),
            decorate(tf.MinMaxScaler(name=f'numeric_minmax_scaler_{seq_no}')),
            decorate(tf.MaxAbsScaler(name=f'numeric_maxabs_scaler_{seq_no}')),
            decorate(tf.RobustScaler(name=f'numeric_robust_scaler_{seq_no}'))
        ], name=f'numeric_or_scaler_{seq_no}'
    )
    scaler_optional = Optional(scaler_options, keep_link=True, name=f'numeric_scaler_optional_{seq_no}')

    pipeline = Pipeline([imputer, scaler_optional],
                        name=f'numeric_pipeline_complex_{seq_no}',
                        columns=column_number_exclude_timedelta)
    return pipeline
