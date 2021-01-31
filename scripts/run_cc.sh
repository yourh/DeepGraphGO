#!/usr/bin/env bash

python main.py -m configure/dgg.yaml -d configure/cc.yaml --model-id 0
python main.py -m configure/dgg.yaml -d configure/cc.yaml --model-id 1
python main.py -m configure/dgg.yaml -d configure/cc.yaml --model-id 2
python bagging.py -m configure/dgg.yaml -d configure/cc.yaml -n 3
python evaluation.py data/cc_test_go.txt results/DeepGraphGO-Ensemble-cc-test.txt