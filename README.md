Grid.ai examples training [chemprop](https://chemprop.readthedocs.io/en/latest/).
Chemprop is a message passing neural network for molecular property prediction.

## Setup

Following install from [PyPi](https://github.com/chemprop/chemprop#option-1-installing-from-pypi)
```bash
conda create --yes -n chemprop python=3.8
conda activate chemprop
conda install -c conda-forge rdkit
pip install git+https://github.com/bp-kelley/descriptastorus
pip install chemprop
# connect Jupyter notebook
pip install ipykernel # allow usage with Jupyter notebook
python -m ipykernel install --user --name=chemprop # show in Jupyter notebook
```


# prepare dataset
Datasets from MoleculeNet and a 450K subset of ChEMBL from http://www.bioinf.jku.at/research/lsc/index.html have been preprocessed and are available in data.tar.gz. 

```bash
git clone https://github.com/chemprop/chemprop.git
cd chemprop
tar xvzf data.tar.gz
```


# Training

Couple of ways to run
```bash
# command line
chemprop_train --data_path data/tox21.csv --dataset_type classification --save_dir tox21_checkpoints
# as a shell script
run.sh --data_dir data
```

on Grid.ai

```bash
# run shell script on Grid.ai using requirements.txt
grid run run.sh --data_dir grid:chemprop:1
# shell script on Grid.ai with environment.yml to build out the conda environment (will override and use base)
grid run --dependency_file environment.yml run.sh --data_dir grid:chemprop:1
```
