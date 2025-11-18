# gLSTM (GNN-xLSTM)
Code for the paper "gLSTM: Mitigating Over-Squashing by Increasing Storage Capacity".

This repository builds upon the [Reassessed LRGB codebase](https://github.com/toenshoff/LRGB); I am grateful to the authors for making their work publicly available. I have attempted to preserve the original commit history to clearly distinguish newly introduced code. Note, however, that automated formatting has resulted in widespread changes across many files.

Note there are some nomenclature changes between the code and the paper. In particular, we refer to the model as GNN-xLSTM rather than gLSTM, NAR is key-recall and NARR is key-recall-regression.

## Installation

Create a `.env` file and put your WandB username in as `WANDB_ENTITY=...`.

You can create an install a matching conda environment to the one I used with
```bash
conda env create -f environment.yml
```
And activate with
```bash
conda activate gnn_xlstm
```

## Codebase Structure

This codebase uses [GraphGym](https://pytorch-geometric.readthedocs.io/en/latest/advanced/graphgym.html). This means it follows the GraphGym structure - our top-level folder for code is `gnn_xlstm`, and the subfolders inside specify the GraphGym components.

Lots of this is simply inherited code, mostly from the LRGB codebase above. For our work, the particularly relevant code files are as follows:
```
.
├── config
│   ├── custom_gnn_config.py
│   ├── key_recall_config.py
│   └── xlstm_config.py
├── encoder
│   └── key_recall_encoder.py
├── head
│   └── ring_transfer.py
├── loader
│   ├── dataset
│   │   ├── key_recall_regression.py
│   │   └── key_recall.py
│   └── master_loader.py
├── network
│   └── gnn_xlstm.py
├── transform
│   └── transforms.py
└── utils.py
```
And a quick per-file description:
- `config/custom_gnn_config.py`
    - A few additional general purpose settings - especially K-hop.
- `config/key_recall_config.py`
    - Config for the Neighbor Associative Recall, which throughout this codebase is referred to as key-recall.
- `head/ring_transfer.py`
    - We use this graph head for key-recall, as it masks out all but a single node (the central node).
- `loader/dataset/key_recall_regression.py`
    - Code for the NARR task
- `loader/dataset/key_recall.py`
    - Code for the NAR task
- `loader/master_loader.py`
    - This contains modifications to support key-recall.
- `network/gnn_xlstm.py`
    - This contains all of the code for the gLSTM model.
- `transform/transforms.py`
    - This contains code for the K-hop aggregation, which is applied as a pre-transform to datasets.

## Training

Training is via GraphGym. The entrypoint is the `main.py` file.

I have made scripts matching the configurations used in the experiments in the paper - these can all be found in the subdirectories of `./scripts`. Note these are recreations of the SLURM job files that I actually used, so there may be small mistakes.

## Sweeps

I provide the WandB sweep configuration files that I used in the `./sweeps` directory. You can create a sweep with
```bash
wandb sweep --project your_project sweeps/sweep_file.yaml
```
This will describe how to create sweep agents to perform this sweep. See [here](https://docs.wandb.ai/guides/sweeps/) for more information.

## Reproducing Our Results

### Tables

Tables can all be reproduced by running the `results.ipynb` notebook.

### Figures

Almost all figures can be reproduced by running `plot_all_nar_results` in `notebook_utils.py`. You can just do
```bash
python notebook_utils.py
```
to run this. This should populate the `figures` directory.

You can recreate the deep-vs-shallow Tree Jacobian Norms figure (Figure 2) by running the `flat_vs_deep_sensitivity.ipynb` notebook.

## Testing

I have added in some basic testing for the NAR (key-recall) synthetic task and the associated embedding function. You can find these in the `tests` directory. Note that these tests are for NAR only - not NARR.

Run these by installing `pytest` and running:
```bash
pytest tests
```