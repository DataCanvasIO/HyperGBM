from hypernets.conf import configure, Configurable, Bool, Float, Int, Enum, List
from hypernets.tabular.sklearn_ex import DatetimeEncoder


@configure()
class HyperGBMCfg(Configurable):
    # numeric
    numeric_pipeline_mode = \
        Enum(['simple', 'complex'], default_value='complex',
             config=True,
             help='Feature scaling mode, simple (standard only) or '
                  'complex (search in standard, logstandard, minmax, maxabs and robust).'
             )

    # category
    category_pipeline_enabled = \
        Bool(True,
             config=True,
             help='detect and encode category feature from training data or not.'
             )
    category_pipeline_mode = \
        Enum(['simple', 'complex'], default_value='simple',
             config=True,
             help='Feature encoding mode, simple (SafeOrdinalEncoder) or '
                  'complex (search in SafeOrdinalEncoder and SafeOneHot+Optional(SVD)).'
             )
    category_pipeline_auto_detect = \
        Bool(False,
             config=True,
             help='detect category feature from numeric and datetime columns or not.'
             )
    category_pipeline_auto_detect_exponent = \
        Float(0.5,
              config=True,
              help=''
              )

    # datetime
    datetime_pipeline_enabled = \
        Bool(False,
             config=True,
             help='detect and encode datetime feature from training data or not.'
             )
    datetime_pipeline_encoder_include = \
        List(DatetimeEncoder.default_include, allow_none=True, config=True,
             help='include items when encoding datetime feature.')
    datetime_pipeline_encoder_exclude = \
        List(allow_none=True,
             config=True,
             help='exclude items when encoding datetime feature.')

    # text
    text_pipeline_enabled = \
        Bool(False,
             config=True,
             help='detect and encode text feature from training data or not.'
             )
    text_pipeline_word_count_threshold = \
        Int(3,
            config=True,
            help='')

    # estimators
    estimator_lightgbm_enabled = \
        Bool(True,
             config=True,
             help='enable lightgbm or not.'
             )
    estimator_xgboost_enabled = \
        Bool(True,
             config=True,
             help='enable xgboost or not.'
             )
    estimator_catboost_enabled = \
        Bool(True,
             config=True,
             help='enable catboost or not.'
             )
    estimator_histgb_enabled = \
        Bool(False,
             config=True,
             help='enable HistGradientBoosting or not.'
             )

    straightforward_excluded = \
        List(['TruncatedSVD', 'OneHot'],
             allow_none=True,
             config=True,
             help='no-straightforward transformer name list.')
