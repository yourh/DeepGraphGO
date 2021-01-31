#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/25
@author yrh

"""

import warnings
import numpy as np
import scipy.sparse as ssp
from sklearn.metrics import average_precision_score as aupr

__all__ = ['fmax', 'aupr', 'pair_aupr', 'ROOT_GO_TERMS']
ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}


def fmax(targets: ssp.csr_matrix, scores: np.ndarray):
    fmax_ = 0.0, 0.0
    for cut in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, r = correct / cut_sc.sum(axis=1), correct / targets.sum(axis=1)
            p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)
        if np.isnan(p):
            continue
        try:
            fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, cut))
        except ZeroDivisionError:
            pass
    return fmax_


def pair_aupr(targets: ssp.csr_matrix, scores: np.ndarray, top=200):
    scores[np.arange(scores.shape[0])[:, None],
           scores.argpartition(scores.shape[1] - top)[:, :-top]] = -1e100
    return aupr(targets.toarray().flatten(), scores.flatten())
