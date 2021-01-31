#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/25
@author yrh

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from logzero import logger

from deepgraphgo.modules import *

__all__ = ['GcnNet']


class GcnNet(nn.Module):
    """
    """

    def __init__(self, *, labels_num, input_size, hidden_size, num_gcn=0, dropout=0.5, residual=True,
                 **kwargs):
        super(GcnNet, self).__init__()
        logger.info(F'GCN: labels_num={labels_num}, input size={input_size}, hidden_size={hidden_size}, '
                    F'num_gcn={num_gcn}, dropout={dropout}, residual={residual}')
        self.labels_num = labels_num
        self.input = nn.EmbeddingBag(input_size, hidden_size, mode='sum', include_last_offset=True)
        self.input_bias = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.update = nn.ModuleList(NodeUpdate(hidden_size, hidden_size, dropout) for _ in range(num_gcn))
        self.output = nn.Linear(hidden_size, self.labels_num)
        self.residual = residual
        self.num_gcn = num_gcn
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input.weight)
        for update in self.update:
            update.reset_parameters()
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, nf: dgl.NodeFlow, inputs):
        nf.copy_from_parent()
        outputs = self.dropout(F.relu(self.input(*inputs) + self.input_bias))
        nf.layers[0].data['h'] = outputs
        for i, update in enumerate(self.update):
            if self.residual:
                nf.block_compute(i,
                                 dgl.function.u_mul_e('h', 'self', out='m_res'),
                                 dgl.function.sum(msg='m_res', out='res'))
            nf.block_compute(i,
                             dgl.function.u_mul_e('h', 'ppi', out='ppi_m_out'),
                             dgl.function.sum(msg='ppi_m_out', out='ppi_out'), update)
        return self.output(nf.layers[-1].data['h'])
