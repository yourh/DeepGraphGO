#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/25
@author yrh

"""

import click
from itertools import chain

from deepgraphgo.data_utils import get_pid_go, get_pid_go_sc, get_pid_go_mat, get_pid_go_sc_mat
from deepgraphgo.evaluation import fmax, pair_aupr, ROOT_GO_TERMS


def evaluate_metrics(pid_go, pid_go_sc):
    pid_go_sc, pid_go = get_pid_go_sc(pid_go_sc), get_pid_go(pid_go)
    pid_list = list(pid_go.keys())
    go_list = sorted(set(list(chain(*([pid_go[p_] for p_ in pid_list] +
                                      [pid_go_sc[p_] for p_ in pid_list if p_ in pid_go_sc])))) - ROOT_GO_TERMS)
    go_mat, score_mat = get_pid_go_mat(pid_go, pid_list, go_list), get_pid_go_sc_mat(pid_go_sc, pid_list, go_list)
    return fmax(go_mat, score_mat), pair_aupr(go_mat, score_mat)


@click.command()
@click.argument('pid_go', type=click.Path(exists=True))
@click.argument('pid_go_sc', type=click.Path(exists=True))
def main(pid_go, pid_go_sc):
    (fmax_, t_), aupr_ = evaluate_metrics(pid_go, pid_go_sc)
    print(F'Fmax: {fmax_:.3f} {t_:.2f}', F'AUPR: {aupr_:.3f}')


if __name__ == '__main__':
    main()
