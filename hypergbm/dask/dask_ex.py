# -*- coding:utf-8 -*-
"""

"""
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml import decomposition as dm_dec
from dask_ml import preprocessing as dm_pre
from sklearn import preprocessing as sk_pre
from sklearn.base import BaseEstimator, TransformerMixin


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        assert len(X.shape) == 2

        if isinstance(X, (pd.DataFrame, dd.DataFrame)):
            return self._fit_df(X, y)
        elif isinstance(X, (np.ndarray, da.Array)):
            return self._fit_array(X, y)
        else:
            raise Exception(f'Unsupported type "{type(X)}"')

    def _fit_df(self, X, y=None):
        if y is not None:
            return self._fit_array(X.values, y.values)
        else:
            return self._fit_array(X.values)

    def _fit_array(self, X, y=None):
        n_features = X.shape[1]
        for n in range(n_features):
            le = dm_pre.LabelEncoder()
            le.fit(X[:, n])
            self.encoders[n] = le
        return self

    def transform(self, X):
        assert len(X.shape) == 2

        if isinstance(X, dd.DataFrame):
            return self._transform_dask_df(X)
        elif isinstance(X, da.Array):
            return self._transform_dask_array(X)
        else:
            raise Exception(f'Unsupported type "{type(X)}"')

    def _transform_dask_df(self, X):
        data = self._transform_dask_array(X.values)

        return dd.from_dask_array(data, columns=X.columns)

    def _transform_dask_array(self, X):
        n_features = X.shape[1]
        assert n_features == len(self.encoders.items())

        data = []
        for n in range(n_features):
            data.append(self.encoders[n].transform(X[:, n]))

        return da.stack(data, axis=-1, allow_unknown_chunksizes=True)

    # def fit_transform(self, X, y=None):
    #     return self.fit(X, y).transform(X)


class OneHotEncoder(dm_pre.OneHotEncoder):
    def fit(self, X, y=None):
        if isinstance(X, dd.DataFrame) and self.categories == "auto" \
                and any(d.name == 'object' for d in X.dtypes):
            a = []
            for i in range(len(X.columns)):
                Xi = X.iloc[:, i]
                if Xi.dtype == 'object':
                    Xi = Xi.astype('category').cat.as_known()
                a.append(Xi)
            X = dd.concat(a, axis=1)

        return super(OneHotEncoder, self).fit(X, y)

    def get_feature_names(self, input_features=None):
        if not hasattr(self, 'drop_idx_'):
            setattr(self, 'drop_idx_', None)
        return super(OneHotEncoder, self).get_feature_names(input_features)


class TruncatedSVD(dm_dec.TruncatedSVD):
    def fit_transform(self, X, y=None):
        if isinstance(X, dd.DataFrame):
            r = super(TruncatedSVD, self).fit_transform(X.values, y)
            return r  # fixme, restore to DataFrame ??

        return super(TruncatedSVD, self).fit_transform(X, y)

    def transform(self, X, y=None):
        if isinstance(X, dd.DataFrame):
            return super(TruncatedSVD, self).transform(X.values, y)

        return super(TruncatedSVD, self).transform(X, y)

    def inverse_transform(self, X):
        if isinstance(X, dd.DataFrame):
            return super(TruncatedSVD, self).inverse_transform(X.values)

        return super(TruncatedSVD, self).inverse_transform(X)


class MaxAbsScaler(sk_pre.MaxAbsScaler):
    __doc__ = sk_pre.MaxAbsScaler.__doc__

    def fit(self, X, y=None, ):
        from dask_ml.utils import handle_zeros_in_scale

        self._reset()

        max_abs = X.reduction(lambda x: x.abs().max(),
                              aggregate=lambda x: x.max(),
                              token=self.__class__.__name__
                              ).compute()
        scale = handle_zeros_in_scale(max_abs)

        setattr(self, 'max_abs_', max_abs)
        setattr(self, 'scale_', scale)
        setattr(self, 'n_samples_seen_', 0)

        self.n_features_in_ = X.shape[1]
        return self

    def partial_fit(self, X, y=None, ):
        raise NotImplementedError()

    def transform(self, X, y=None, copy=None, ):
        # Workaround for https://github.com/dask/dask/issues/2840
        if isinstance(X, dd.DataFrame):
            X = X.div(self.scale_)
        else:
            X = X / self.scale_
        return X

    def inverse_transform(self, X, y=None, copy=None, ):
        if not hasattr(self, "scale_"):
            raise Exception(
                "This %(name)s instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before "
                "using this method."
            )
        if copy:
            X = X.copy()
        if isinstance(X, dd.DataFrame):
            X = X.mul(self.scale_)
        else:
            X = X * self.scale_

        return X

