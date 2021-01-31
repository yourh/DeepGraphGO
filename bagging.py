#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/25
@author yrh

"""

import click
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML

from deepgraphgo.data_utils import get_mlb, get_pid_list, output_res


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
@click.option('-n', 'num_models', type=click.INT, default=None)
def main(data_cnf, model_cnf, num_models):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    data_name, model_name = data_cnf['name'], model_cnf['name']
    res_path = Path(data_cnf['results'])
    mlb = get_mlb(Path(data_cnf['mlb']))
    test_cnf = data_cnf['test']
    test_name, test_pid_list = test_cnf.pop('name'), get_pid_list(test_cnf['pid_list_file'])
    sc_mat = np.zeros((len(test_pid_list), len(mlb.classes_)))
    for i in range(num_models):
        sc_mat += np.load(res_path/F'{model_name}-Model-{i}-{data_name}-{test_name}.npy') / num_models
    res_path_ = res_path/F'{model_name}-Ensemble-{data_name}-{test_name}'
    np.save(res_path_, sc_mat)
    output_res(res_path_.with_suffix('.txt'), test_pid_list, mlb.classes_, sc_mat)


if __name__ == '__main__':
    main()
