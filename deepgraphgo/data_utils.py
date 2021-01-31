#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/25
@author yrh

"""

import joblib
import numpy as np
import scipy.sparse as ssp
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO
from sklearn.preprocessing import MultiLabelBinarizer

from deepgraphgo.psiblast_utils import blast

__all__ = ['get_pid_list', 'get_go_list', 'get_pid_go', 'get_pid_go_sc', 'get_data', 'output_res', 'get_mlb',
           'get_pid_go_mat', 'get_pid_go_sc_mat', 'get_ppi_idx', 'get_homo_ppi_idx']


def get_pid_list(pid_list_file):
    try:
        with open(pid_list_file) as fp:
            return [line.split()[0] for line in fp]
    except TypeError:
        return pid_list_file


def get_go_list(pid_go_file, pid_list):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                pid_go[(line_list:=line.split())[0]].append(line_list[1])
        return [pid_go[pid_] for pid_ in pid_list]
    else:
        return None


def get_pid_go(pid_go_file):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                pid_go[(line_list:=line.split('\t'))[0]].append(line_list[1])
        return dict(pid_go)
    else:
        return None


def get_pid_go_sc(pid_go_sc_file):
    pid_go_sc = defaultdict(dict)
    with open(pid_go_sc_file) as fp:
        for line in fp:
            pid_go_sc[line_list[0]][line_list[1]] = float((line_list:=line.split('\t'))[2])
    return dict(pid_go_sc)


def get_data(fasta_file, pid_go_file=None, feature_type=None, **kwargs):
    pid_list, data_x = [], []
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        pid_list.append(seq.id)
        data_x.append(str(seq.seq))
    if feature_type is not None:
        feature_path = Path(kwargs[feature_type])
        if feature_path.suffix == '.npy':
            data_x = np.load(feature_path)
        elif feature_path.suffix == '.npz':
            data_x = ssp.load_npz(feature_path)
        else:
            raise ValueError(F'Only support suffix of .npy for np.ndarray or .npz for scipy.csr_matrix as feature.')
    return pid_list, data_x, get_go_list(pid_go_file, pid_list)


def get_mlb(mlb_path: Path, labels=None, **kwargs) -> MultiLabelBinarizer:
    if mlb_path.exists():
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True, **kwargs)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb


def output_res(res_path: Path, pid_list, go_list, sc_mat):
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, 'w') as fp:
        for pid_, sc_ in zip(pid_list, sc_mat):
            for go_, s_ in zip(go_list, sc_):
                if s_ > 0.0:
                    print(pid_, go_, s_, sep='\t', file=fp)


def get_pid_go_mat(pid_go, pid_list, go_list):
    go_mapping = {go_: i for i, go_ in enumerate(go_list)}
    r_, c_, d_ = [], [], []
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go:
            for go_ in pid_go[pid_]:
                if go_ in go_mapping:
                    r_.append(i)
                    c_.append(go_mapping[go_])
                    d_.append(1)
    return ssp.csr_matrix((d_, (r_, c_)), shape=(len(pid_list), len(go_list)))


def get_pid_go_sc_mat(pid_go_sc, pid_list, go_list):
    sc_mat = np.zeros((len(pid_list), len(go_list)))
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go_sc:
            for j, go_ in enumerate(go_list):
                sc_mat[i, j] = pid_go_sc[pid_].get(go_, -1e100)
    return sc_mat


def get_ppi_idx(pid_list, data_y, net_pid_map):
    pid_list_ = tuple(zip(*[(i, pid, net_pid_map[pid])
                            for i, pid in enumerate(pid_list) if pid in net_pid_map]))
    assert pid_list_
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    return pid_list_[0], pid_list_[1], pid_list_[2], data_y[pid_list_[0]] if data_y is not None else data_y


def get_homo_ppi_idx(pid_list, fasta_file, data_y, net_pid_map, net_blastdb, blast_output_path):
    blast_sim = blast(net_blastdb, pid_list, fasta_file, blast_output_path)
    pid_list_ = []
    for i, pid in enumerate(pid_list):
        blast_sim[pid][None] = float('-inf')
        pid_ = pid if pid in net_pid_map else max(blast_sim[pid].items(), key=lambda x: x[1])[0]
        if pid_ is not None:
            pid_list_.append((i, pid, net_pid_map[pid_]))
    pid_list_ = tuple(zip(*pid_list_))
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    return pid_list_[0], pid_list_[1], pid_list_[2], data_y[pid_list_[0]] if data_y is not None else data_y
