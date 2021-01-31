#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/25
@author yrh

"""

import warnings
import click
import numpy as np
import scipy.sparse as ssp
import torch
import dgl.data
from pathlib import Path
from ruamel.yaml import YAML
from logzero import logger

from deepgraphgo.data_utils import get_pid_list, get_data, get_mlb, output_res, get_ppi_idx, get_homo_ppi_idx
from deepgraphgo.models import Model


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
@click.option('--mode', type=click.Choice(['train', 'eval']), default=None)
@click.option('--model-id', type=click.INT, default=None)
def main(data_cnf, model_cnf, mode, model_id):
    model_id = F'-Model-{model_id}' if model_id is not None else ''
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    data_name, model_name = data_cnf['name'], model_cnf['name']
    run_name = F'{model_name}{model_id}-{data_name}'
    model, model_cnf['model']['model_path'] = None, Path(data_cnf['model_path'])/F'{run_name}'
    data_cnf['mlb'] = Path(data_cnf['mlb'])
    data_cnf['results'] = Path(data_cnf['results'])
    logger.info(F'Model: {model_name}, Path: {model_cnf["model"]["model_path"]}, Dataset: {data_name}')

    net_pid_list = get_pid_list(data_cnf['network']['pid_list'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    net_blastdb = data_cnf['network']['blastdb']
    dgl_graph = dgl.data.utils.load_graphs(data_cnf['network']['dgl'])[0][0]
    self_loop = torch.zeros_like(dgl_graph.edata['ppi'])
    self_loop[dgl_graph.edge_ids(nr_:=np.arange(dgl_graph.number_of_nodes()), nr_)] = 1.0
    dgl_graph.edata['self'] = self_loop
    dgl_graph.edata['ppi'] = dgl_graph.edata['ppi'].float().cuda()
    dgl_graph.edata['self'] = dgl_graph.edata['self'].float().cuda()
    logger.info(F'{dgl_graph}')
    network_x = ssp.load_npz(data_cnf['network']['feature'])

    if mode is None or mode == 'train':
        train_pid_list, _, train_go = get_data(**data_cnf['train'])
        valid_pid_list, _, valid_go = get_data(**data_cnf['valid'])
        mlb = get_mlb(data_cnf['mlb'], train_go)
        labels_num = len(mlb.classes_)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            train_y, valid_y = mlb.transform(train_go).astype(np.float32), mlb.transform(valid_go).astype(np.float32)
        *_, train_ppi, train_y = get_ppi_idx(train_pid_list, train_y, net_pid_map)
        *_, valid_ppi, valid_y = get_homo_ppi_idx(valid_pid_list, data_cnf['valid']['fasta_file'],
                                                  valid_y, net_pid_map, net_blastdb,
                                                  data_cnf['results']/F'{data_name}-valid-ppi-blast-out')
        logger.info(F'Number of Labels: {labels_num}')
        logger.info(F'Size of Training Set: {len(train_ppi)}')
        logger.info(F'Size of Validation Set: {len(valid_ppi)}')

        model = Model(labels_num=labels_num, dgl_graph=dgl_graph, network_x=network_x,
                      input_size=network_x.shape[1], **model_cnf['model'])
        model.train((train_ppi, train_y), (valid_ppi, valid_y), **model_cnf['train'])

    if mode is None or mode == 'eval':
        mlb = get_mlb(data_cnf['mlb'])
        labels_num = len(mlb.classes_)
        if model is None:
            model = Model(labels_num=labels_num, dgl_graph=dgl_graph, network_x=network_x,
                          input_size=network_x.shape[1], **model_cnf['model'])
        test_cnf = data_cnf['test']
        test_name = test_cnf.pop('name')
        test_pid_list, _, test_go = get_data(**test_cnf)
        test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, test_cnf['fasta_file'],
                                                                      None, net_pid_map, net_blastdb,
                                                                      data_cnf['results']/F'{data_name}-{test_name}'
                                                                                          F'-ppi-blast-out')
        scores = np.zeros((len(test_pid_list), len(mlb.classes_)))
        scores[test_res_idx_] = model.predict(test_ppi, **model_cnf['test'])
        res_path = data_cnf['results']/F'{run_name}-{test_name}'
        output_res(res_path.with_suffix('.txt'), test_pid_list, mlb.classes_, scores)
        np.save(res_path, scores)


if __name__ == '__main__':
    main()
