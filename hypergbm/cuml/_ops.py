# -*- coding:utf-8 -*-
"""

"""

import cudf
import numpy as np

from hypernets.core import ModuleChoice, Choice
from hypernets.pipeline.base import Pipeline, PipelineOutput, HyperTransformer
from hypernets.tabular.cuml_ex import CumlToolBox

_cs = CumlToolBox.column_selector
_tfs = CumlToolBox.transformers

_support_median = hasattr(cudf.DataFrame(), 'median')


class CumlPipelineOutput(PipelineOutput):
    @staticmethod
    def create_pipeline(steps):
        return _tfs['Pipeline'](steps)


class CumlHyperPipeline(Pipeline):
    output_space_cls = CumlPipelineOutput


def categorical_pipeline_simple(seq_no=0):
    steps = [
        HyperTransformer(_tfs['ConstantImputer'], missing_values=[np.nan, None], fill_value='',
                         name=f'categorical_imputer_{seq_no}'),
        # ModuleChoice([
        #   HyperTransformer(_tfs['MultiLabelEncoder'], name=f'categorical_label_encoder_{seq_no}', dtype='float32'),
        #   HyperTransformer(_tfs['MultiTargetEncoder'], name=f'categorical_target_encoder_{seq_no}', dtype='float32'),
        # ], name=f'categorical_encoder_{seq_no}'),
        HyperTransformer(_tfs['MultiLabelEncoder'], name=f'categorical_label_encoder_{seq_no}', dtype='float32'),
    ]

    pipeline = CumlHyperPipeline(steps, columns=_cs.column_object_category_bool,
                                 name=f'categorical_pipeline_simple_{seq_no}')
    return pipeline


def categorical_pipeline_complex(seq_no=0):
    steps = [
        HyperTransformer(_tfs['ConstantImputer'], missing_values=[np.nan, None], fill_value='',
                         name=f'categorical_imputer_{seq_no}'),
        ModuleChoice([
            HyperTransformer(_tfs['MultiLabelEncoder'], name=f'categorical_label_encoder_{seq_no}',
                             dtype='float32'),
            HyperTransformer(_tfs['MultiTargetEncoder'], name=f'categorical_target_encoder_{seq_no}',
                             dtype='float32',
                             split_method=Choice(['random', 'continuous', 'interleaved']),
                             smooth=Choice([None, 0, 10]),
                             ),
            CumlHyperPipeline([
                HyperTransformer(_tfs['OneHotEncoder'], name=f'categorical_onehot_{seq_no}',
                                 sparse=False, handle_unknown='ignore'),
                HyperTransformer(_tfs['TruncatedSVD'], name=f'categorical_svd_{seq_no}',
                                 n_components=3),
            ], name=f'categorical_onehot_svd_{seq_no}')
        ], name=f'categorical_encoder_{seq_no}')
    ]

    pipeline = CumlHyperPipeline(steps, columns=_cs.column_object_category_bool,
                                 name=f'categorical_pipeline_complex_{seq_no}')
    return pipeline


def numeric_pipeline_simple(impute_strategy='mean', seq_no=0):
    steps = [
        HyperTransformer(_tfs['SimpleImputer'], missing_values=np.nan, strategy=impute_strategy, fill_value=0,
                         name=f'numeric_imputer_{seq_no}'),
        HyperTransformer(_tfs['StandardScaler'], name=f'numeric_standard_scaler_{seq_no}')
    ]
    pipeline = CumlHyperPipeline(steps, columns=_cs.column_number_exclude_timedelta,
                                 name=f'numeric_pipeline_simple_{seq_no}')
    return pipeline


def numeric_pipeline_complex(impute_strategy=None, seq_no=0):
    if impute_strategy is None:
        if _support_median:
            impute_strategy = Choice(['mean', 'median', 'constant', ])  # 'most_frequent'
        else:
            impute_strategy = Choice(['mean', 'constant', ])  # 'median',  'most_frequent'
    elif isinstance(impute_strategy, list):
        impute_strategy = Choice(impute_strategy)

    imputer = HyperTransformer(_tfs['SimpleImputer'], missing_values=np.nan, strategy=impute_strategy,
                               name=f'numeric_imputer_{seq_no}')
    scaler_options = ModuleChoice(
        [
            # HyperTransformer(pre.LogStandardScaler,name=f'numeric_log_standard_scaler_{seq_no}'),
            HyperTransformer(_tfs['PassThroughEstimator'], name=f'numeric_no_scaler_{seq_no}'),
            HyperTransformer(_tfs['StandardScaler'], name=f'numeric_standard_scaler_{seq_no}'),
            HyperTransformer(_tfs['MinMaxScaler'], name=f'numeric_minmax_scaler_{seq_no}'),
            HyperTransformer(_tfs['MaxAbsScaler'], name=f'numeric_maxabs_scaler_{seq_no}'),
            HyperTransformer(_tfs['RobustScaler'], name=f'numeric_robust_scaler_{seq_no}')
        ], name=f'numeric_scaler_{seq_no}'
    )

    pipeline = CumlHyperPipeline([imputer, scaler_options], columns=_cs.column_number_exclude_timedelta,
                                 name=f'numeric_pipeline_complex_{seq_no}')
    return pipeline
