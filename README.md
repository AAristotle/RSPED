# [Exploring Relevant Snapshots and Neighboring Entities for Temporal Knowledge Graph Reasoning](https://github.com/AAristotle/RSPED)

This repository contains the implementation of **RSPED**, a Relation-Selective Potential Event Discovery framework for temporal knowledge graph reasoning.

RSPED enhances extrapolative reasoning by selecting relevant relational and temporal contexts from historical snapshots. It consists of a Relevant Relation Selector for filtering noisy temporal facts and a Potential Event Discovery module for capturing semantic dependencies among candidate entities.

## Installation

Create a conda environment with PyTorch and the required packages:

```bash
conda create --name rsped_env python=3.8
conda activate rsped_env
conda install pytorch cudatoolkit -c pytorch
pip install dgl numpy scipy pandas tqdm rdflib
```

Please install the PyTorch and DGL versions that match your CUDA environment.

## Datasets

Download the datasets and place the processed files under the `data/` folder:

```text
data/
  ICEWS14/
  ICEWS18/
  GDELT/
  YAGO/
  WIKI/
```

Each dataset folder should contain `stat.txt`, `train.txt`, `test.txt`, and the required frequency and label files used by `main.py`. For datasets with validation splits, also include `valid.txt` and the corresponding validation files.

The datasets used in the paper are:

| Dataset | Entities | Relations | Training | Validation | Test | Time gap | Timestamps |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ICEWS14 | 12,498 | 260 | 323,895 | - | 341,409 | 1 day | 365 |
| ICEWS18 | 23,033 | 256 | 373,018 | 45,995 | 49,545 | 1 day | 304 |
| GDELT | 7,691 | 240 | 1,734,399 | 238,765 | 305,241 | 15 mins | 2,976 |
| YAGO | 10,623 | 10 | 161,540 | 19,523 | 20,026 | 1 year | 189 |
| WIKI | 12,554 | 24 | 539,286 | 67,538 | 63,110 | 1 year | 232 |

## Reproducing Results of RSPED

To reproduce the results of RSPED, run the following commands:

```bash
python main.py --dataset ICEWS14 --description rsped_icews14 --max-epochs 15 --timestamps 365 --use-valid False --lr 0.001 --batch-size 1024 --embedding-dim 200 --dropout 0.2 --graph-layer 2 --history-len 1 --alpha 0.1 --lambdax 2 --gamma 0.1

python main.py --dataset ICEWS18 --description rsped_icews18 --max-epochs 25 --timestamps 304 --use-valid True --lr 0.001 --batch-size 1024 --embedding-dim 200 --dropout 0.2 --graph-layer 2 --history-len 1 --alpha 0.1 --lambdax 2 --gamma 0.1

python main.py --dataset GDELT --description rsped_gdelt --max-epochs 25 --timestamps 2976 --use-valid True --lr 0.001 --batch-size 1024 --embedding-dim 200 --dropout 0.2 --graph-layer 2 --history-len 1 --alpha 0.1 --lambdax 2 --gamma 0.1

python main.py --dataset YAGO --description rsped_yago --max-epochs 25 --timestamps 189 --use-valid True --lr 0.001 --batch-size 1024 --embedding-dim 200 --dropout 0.2 --graph-layer 2 --history-len 1 --alpha 0.1 --lambdax 2 --gamma 0.1

python main.py --dataset WIKI --description rsped_wiki --max-epochs 25 --timestamps 232 --use-valid True --lr 0.001 --batch-size 1024 --embedding-dim 200 --dropout 0.2 --graph-layer 2 --history-len 1 --alpha 0.1 --lambdax 2 --gamma 0.1
```

The trained models and evaluation logs will be saved under the `SAVE/` folder.

## Evaluation

By default, training is followed by evaluation on the test set. To evaluate a saved model only, specify `--only-eva True` and provide the saved model directory:

```bash
python main.py --dataset ICEWS18 --only-eva True --model-dir <saved_model_directory> --timestamps 304
```

## Citation

If you use this code, please cite the following paper:

```bibtex
@misc{rsped2025exploring,
  title={Exploring Relevant Snapshots and Neighboring Entities for Temporal Knowledge Graph Reasoning},
  year={2025},
  note={Preprint submitted to Elsevier}
}
```

## Acknowledgement

This implementation builds on temporal knowledge graph reasoning methods based on graph neural networks and temporal decoders. Thanks to the authors of previous TKGR models and datasets for their valuable contributions.
