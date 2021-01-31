#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2021/1/29
@author yrh

"""

import click
import numpy as np
import scipy.sparse as ssp
import networkx as nx
import dgl
import dgl.data
from tqdm import tqdm, trange
from logzero import logger

# from deepgraphgo.data_utils import get_pid_list


def get_norm_net_mat(net_mat):
    degree_0 = np.asarray(net_mat.sum(0)).squeeze()
    mat_d_0 = ssp.diags(degree_0 ** -0.5, format='csr')
    degree_1 = np.asarray(net_mat.sum(1)).squeeze()
    mat_d_1 = ssp.diags(degree_1 ** -0.5, format='csr')
    return mat_d_0 @ net_mat @ mat_d_1


@click.command()
@click.argument('ppi_net_mat_path', type=click.Path(exists=True))
@click.argument('dgl_graph_path', type=click.Path())
@click.argument('top', type=click.INT, default=100, required=False)
def main(ppi_net_mat_path, dgl_graph_path, top):
    ppi_net_mat = (mat_:=ssp.load_npz(ppi_net_mat_path)) + ssp.eye(mat_.shape[0], format='csr')
    logger.info(F'{ppi_net_mat.shape} {ppi_net_mat.nnz}')
    r, c, v = [], [], []
    for i in trange(ppi_net_mat.shape[0]):
        for v_, c_ in sorted(zip(ppi_net_mat[i].data, ppi_net_mat[i].indices), reverse=True)[:top]:
            r.append(i)
            c.append(c_)
            v.append(v_)
    ppi_net_mat = get_norm_net_mat(ssp.csc_matrix((v, (r, c)), shape=ppi_net_mat.shape).T)
    logger.info(F'{ppi_net_mat.shape} {ppi_net_mat.nnz}')
    ppi_net_mat_coo = ssp.coo_matrix(ppi_net_mat)
    nx_graph = nx.DiGraph()
    for u, v, d in tqdm(zip(ppi_net_mat_coo.row, ppi_net_mat_coo.col, ppi_net_mat_coo.data),
                        total=ppi_net_mat_coo.nnz, desc='PPI'):
        nx_graph.add_edge(u, v, ppi=d)
    dgl_graph = dgl.DGLGraph()
    dgl_graph.from_networkx(nx_graph, edge_attrs=('ppi',))
    assert dgl_graph.in_degrees().max() <= top
    dgl.data.utils.save_graphs(dgl_graph_path, dgl_graph)


if __name__ == '__main__':
    main()
