# -*- coding:utf-8 -*-
"""

"""
import dask_ml.metrics as dm_metrics
import numpy as np
from dask import dataframe as dd, array as da
from sklearn import metrics as sk_metrics

_DASK_TYPES = (dd.DataFrame, da.Array, dd.Series)
_DASK_METRICS = ('accuracy', 'logloss')


def calc_score(y_true, y_preds, y_proba=None, metrics=['accuracy'], task='binary', pos_label=1):
    if any(isinstance(y, _DASK_TYPES) for y in (y_true, y_proba, y_preds)):
        if len(set(metrics).difference(set(_DASK_METRICS))) == 0:
            fn = _calc_score_dask
        else:
            if isinstance(y_true, _DASK_TYPES):
                y_true = y_true.compute()
            if isinstance(y_preds, _DASK_TYPES):
                y_preds = y_preds.compute()
            if isinstance(y_proba, _DASK_TYPES):
                y_proba = y_proba.compute()
            fn = _calc_score_sklean
    else:
        fn = _calc_score_sklean

    return fn(y_true, y_preds, y_proba, metrics, task, pos_label)


def _calc_score_sklean(y_true, y_preds, y_proba=None, metrics=['accuracy'], task='binary', pos_label=1):
    score = {}
    if y_proba is None:
        y_proba = y_preds
    if len(y_proba.shape) == 2 and y_proba.shape[-1] == 1:
        y_proba = y_proba.reshape(-1)
    if len(y_preds.shape) == 2 and y_preds.shape[-1] == 1:
        y_preds = y_preds.reshape(-1)
    for metric in metrics:
        if callable(metric):
            score[metric.__name__] = metric(y_true, y_preds)
        else:
            metric = metric.lower()
            if task == 'multiclass':
                average = 'micro'
            else:
                average = 'binary'

            if metric == 'auc':
                if len(y_proba.shape) == 2:

                    score['auc'] = sk_metrics.roc_auc_score(y_true, y_proba[:, 1], multi_class='ovo')
                else:
                    score['auc'] = sk_metrics.roc_auc_score(y_true, y_proba)

            elif metric == 'accuracy':
                if y_proba is None:
                    score['accuracy'] = 0
                else:
                    score['accuracy'] = sk_metrics.accuracy_score(y_true, y_preds)
            elif metric == 'recall':
                score['recall'] = sk_metrics.recall_score(y_true, y_preds, average=average, pos_label=pos_label)
            elif metric == 'precision':
                score['precision'] = sk_metrics.precision_score(y_true, y_preds, average=average, pos_label=pos_label)
            elif metric == 'f1':
                score['f1'] = sk_metrics.f1_score(y_true, y_preds, average=average, pos_label=pos_label)

            elif metric == 'mse':
                score['mse'] = sk_metrics.mean_squared_error(y_true, y_preds)
            elif metric == 'mae':
                score['mae'] = sk_metrics.mean_absolute_error(y_true, y_preds)
            elif metric == 'msle':
                score['msle'] = sk_metrics.mean_squared_log_error(y_true, y_preds)
            elif metric == 'rmse':
                score['rmse'] = np.sqrt(sk_metrics.mean_squared_error(y_true, y_preds))
            elif metric == 'rootmeansquarederror':
                score['rootmeansquarederror'] = np.sqrt(sk_metrics.mean_squared_error(y_true, y_preds))
            elif metric == 'r2':
                score['r2'] = sk_metrics.r2_score(y_true, y_preds)
            elif metric == 'logloss':
                score['logloss'] = sk_metrics.log_loss(y_true, y_proba)

    return score


def _calc_score_dask(y_true, y_preds, y_proba=None, metrics=('accuracy',), task='binary', pos_label=1):
    score = {}
    if y_proba is None:
        y_proba = y_preds
    if len(y_proba.shape) == 2 and y_proba.shape[-1] == 1:
        y_proba = y_proba.reshape(-1)
    if len(y_preds.shape) == 2 and y_preds.shape[-1] == 1:
        y_preds = y_preds.reshape(-1)

    for metric in metrics:
        if callable(metric):
            score[metric.__name__] = metric(y_true, y_preds)
        else:
            metric = metric.lower()
            if metric == 'accuracy':
                if y_proba is None:
                    score['accuracy'] = 0
                else:
                    score['accuracy'] = dm_metrics.accuracy_score(y_true, y_preds)
            elif metric == 'logloss':
                if isinstance(y_true, dd.Series):
                    ll = dm_metrics.log_loss(y_true.compute(), y_proba)  # fixme
                else:
                    ll = dm_metrics.log_loss(y_true, y_proba)
                if hasattr(ll, 'compute'):
                    ll = ll.compute()
                score['logloss'] = ll
            else:
                import sys
                print('unknown metric:', metric, file=sys.stderr)
    return score
