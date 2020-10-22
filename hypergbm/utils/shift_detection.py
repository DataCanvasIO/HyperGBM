# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMClassifier
from copy import deepcopy
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, matthews_corrcoef, make_scorer
from hypergbm.utils.column_selector import column_object_category_bool, column_number_exclude_timedelta

roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)


def covariate_shift_score(X_train, X_test, copy=True, scorer=roc_auc_scorer, cv=None):
    assert isinstance(X_train, pd.DataFrame) and isinstance(X_test,
                                                            pd.DataFrame), 'X_train and X_test must be a pandas DataFrame.'
    assert len(set(X_train.columns.to_list()) - set(
        X_test.columns.to_list())) == 0, 'The columns in X_train and X_test must be the same.'
    target_col = '__hypernets_csd__target__'
    if copy:
        train = deepcopy(X_train)
        test = deepcopy(X_test)
    else:
        train = X_train
        test = X_test

    # Set target value
    train[target_col] = 0
    test[target_col] = 1
    mixed = pd.concat([train, test], axis=0)
    y = mixed.pop(target_col)

    print('Preprocessing...')
    # Preprocess data: imputing and scaling
    cat_cols = column_object_category_bool(mixed)
    num_cols = column_number_exclude_timedelta(mixed)
    cat_transformer = Pipeline(
        steps=[('imputer_cat', SimpleImputer(strategy='constant')), ('encoder', OrdinalEncoder())])
    num_transformer = Pipeline(steps=[('imputer_num', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('cat', cat_transformer, cat_cols),
                                                   ('num', num_transformer, num_cols)],
                                     remainder='passthrough')
    mixed[cat_cols + num_cols] = preprocessor.fit_transform(mixed)

    # Calculate the shift score for each column separately.
    scores = {}
    print('Scoring...')
    for c in mixed.columns:
        x = mixed[[c]]
        model = LGBMClassifier()
        if cv is None:
            mixed_x_train, mixed_x_test, mixed_y_train, mixed_y_test = train_test_split(x, y, test_size=0.3,
                                                                                        random_state=9527, stratify=y)

            model.fit(mixed_x_train, mixed_y_train, eval_set=(mixed_x_test, mixed_y_test), early_stopping_rounds=20,
                      verbose=False)
            score = scorer(model, mixed_x_test, mixed_y_test)
        else:
            score_ = cross_val_score(model, X=x, y=y, verbose=0, scoring=scorer, cv=cv)
            score = np.mean(score_)
        print(f'column:{c}, score:{score}')
        scores[c] = score

    return scores
