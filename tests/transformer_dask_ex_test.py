# -*- coding:utf-8 -*-
"""

"""
import dask.dataframe as dd
import numpy as np


def prepare_dataframe():
    from hypergbm.datasets import dsutils
    pdf = dsutils.load_bank()
    ddf = dd.from_pandas(pdf, npartitions=2)

    return pdf, ddf


def test_max_abs_scale():
    from sklearn import preprocessing as sk_pre
    import hypergbm.dask_ex as de

    TOL = 0.00001

    pdf, ddf = prepare_dataframe()

    num_columns = [k for k, t in pdf.dtypes.items()
                   if t in (np.int32, np.int64, np.float32, np.float64)]
    pdf = pdf[num_columns]
    ddf = ddf[num_columns]

    sk_s = sk_pre.MaxAbsScaler()
    sk_r = sk_s.fit_transform(pdf)

    de_s = de.MaxAbsScaler()
    de_r = de_s.fit_transform(ddf)

    delta = (sk_s.scale_ - de_s.scale_).abs().max()
    assert delta < TOL

    delta = (sk_r - de_r.compute()).abs().max().max()
    assert delta < TOL

    delta = (sk_s.inverse_transform(sk_r) - de_s.inverse_transform(de_r).compute()) \
        .abs().max().max()
    assert delta < TOL
