# DeepGraphGO
DeepGraphGO: graph neural net for large-scale, multispecies protein function prediction

## Requirements

* python==3.8.3
* numpy==1.19.2
* scipy==1.5.0
* scikit-learn==0.22.1
* networkx==2.4
* torch==1.6.0
* dgl==0.4.3post2
* click==7.1.2
* ruamel.yaml==0.16.6
* biopython==1.78
* tqdm==4.47.0
* logzero==1.5.0
* joblib==0.16.0

## Experiments
```bash
./scripts/preprocessing.sh
./scripts/run_mf.sh
./scripts/run_bp.sh
./scripts/run_cc.sh
```
