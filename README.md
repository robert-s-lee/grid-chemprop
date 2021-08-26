# grid-chemprop
Grid.ai example of https://github.com/chemprop/chemprop

# Session

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
grid datastore create --name chemprop --source data
```


# Training

Many ways to run
```bash
# command line
chemprop_train --data_path data/tox21.csv --dataset_type classification --save_dir tox21_checkpoints
# as a shell script
run.sh data
# run shell script on Grid.ai using requirments.txt
grid run run.sh -data_path grid:chemprop:1
# shell script on Grid.ai with environment.yml to build out the conda environment (will override and use base)
grid run --dependency_file environment.yml run.sh -data_path grid:chemprop:1

```



# Chemprop Options
```bash
% chemprop_train --help                      
usage: chemprop_train --data_path DATA_PATH [--target_columns [TARGET_COLUMNS ...]] [--ignore_columns [IGNORE_COLUMNS ...]] --dataset_type
                      {regression,classification,multiclass} [--multiclass_num_classes MULTICLASS_NUM_CLASSES] [--separate_val_path SEPARATE_VAL_PATH]
                      [--separate_test_path SEPARATE_TEST_PATH] [--data_weights_path DATA_WEIGHTS_PATH] [--target_weights [TARGET_WEIGHTS ...]]
                      [--split_type {random,scaffold_balanced,predetermined,crossval,cv,cv-no-test,index_predetermined}]
                      [--split_sizes SPLIT_SIZES SPLIT_SIZES SPLIT_SIZES] [--num_folds NUM_FOLDS] [--folds_file FOLDS_FILE]
                      [--val_fold_index VAL_FOLD_INDEX] [--test_fold_index TEST_FOLD_INDEX] [--crossval_index_dir CROSSVAL_INDEX_DIR]
                      [--crossval_index_file CROSSVAL_INDEX_FILE] [--seed SEED] [--pytorch_seed PYTORCH_SEED]
                      [--metric {auc,prc-auc,rmse,mae,mse,r2,accuracy,cross_entropy,binary_cross_entropy}]
                      [--extra_metrics [{auc,prc-auc,rmse,mae,mse,r2,accuracy,cross_entropy,binary_cross_entropy} ...]] [--save_dir SAVE_DIR]
                      [--checkpoint_frzn CHECKPOINT_FRZN] [--save_smiles_splits] [--test] [--quiet] [--log_frequency LOG_FREQUENCY]
                      [--show_individual_scores] [--cache_cutoff CACHE_CUTOFF] [--save_preds] [--resume_experiment] [--bias] [--hidden_size HIDDEN_SIZE]
                      [--depth DEPTH] [--mpn_shared] [--dropout DROPOUT] [--activation {ReLU,LeakyReLU,PReLU,tanh,SELU,ELU}] [--atom_messages]
                      [--undirected] [--ffn_hidden_size FFN_HIDDEN_SIZE] [--ffn_num_layers FFN_NUM_LAYERS] [--features_only]
                      [--separate_val_features_path [SEPARATE_VAL_FEATURES_PATH ...]] [--separate_test_features_path [SEPARATE_TEST_FEATURES_PATH ...]]
                      [--separate_val_atom_descriptors_path SEPARATE_VAL_ATOM_DESCRIPTORS_PATH]
                      [--separate_test_atom_descriptors_path SEPARATE_TEST_ATOM_DESCRIPTORS_PATH]
                      [--separate_val_bond_features_path SEPARATE_VAL_BOND_FEATURES_PATH]
                      [--separate_test_bond_features_path SEPARATE_TEST_BOND_FEATURES_PATH] [--config_path CONFIG_PATH] [--ensemble_size ENSEMBLE_SIZE]
                      [--aggregation {mean,sum,norm}] [--aggregation_norm AGGREGATION_NORM] [--reaction] [--reaction_mode {reac_prod,reac_diff,prod_diff}]
                      [--explicit_h] [--epochs EPOCHS] [--warmup_epochs WARMUP_EPOCHS] [--init_lr INIT_LR] [--max_lr MAX_LR] [--final_lr FINAL_LR]
                      [--grad_clip GRAD_CLIP] [--class_balance] [--overwrite_default_atom_features] [--no_atom_descriptor_scaling]
                      [--overwrite_default_bond_features] [--no_bond_features_scaling] [--frzn_ffn_layers FRZN_FFN_LAYERS] [--freeze_first_only]
                      [--smiles_columns [SMILES_COLUMNS ...]] [--number_of_molecules NUMBER_OF_MOLECULES] [--checkpoint_dir CHECKPOINT_DIR]
                      [--checkpoint_path CHECKPOINT_PATH] [--checkpoint_paths [CHECKPOINT_PATHS ...]] [--no_cuda] [--gpu {}]
                      [--features_generator [{morgan,morgan_count,rdkit_2d,rdkit_2d_normalized} ...]] [--features_path [FEATURES_PATH ...]]
                      [--no_features_scaling] [--max_data_size MAX_DATA_SIZE] [--num_workers NUM_WORKERS] [--batch_size BATCH_SIZE]
                      [--atom_descriptors {feature,descriptor}] [--atom_descriptors_path ATOM_DESCRIPTORS_PATH] [--bond_features_path BOND_FEATURES_PATH]
                      [--no_cache_mol] [--empty_cache] [-h]

optional arguments:
  --data_path DATA_PATH
                        (str, required) Path to data CSV file.
  --target_columns [TARGET_COLUMNS ...]
                        (List[str], default=None) Name of the columns containing target values. By default, uses all columns except the SMILES column and
                        the :code:`ignore_columns`.
  --ignore_columns [IGNORE_COLUMNS ...]
                        (List[str], default=None) Name of the columns to ignore when :code:`target_columns` is not provided.
  --dataset_type {regression,classification,multiclass}
                        (Literal['regression', 'classification', 'multiclass'], required) Type of dataset. This determines the loss function used during
                        training.
  --multiclass_num_classes MULTICLASS_NUM_CLASSES
                        (int, default=3) Number of classes when running multiclass classification.
  --separate_val_path SEPARATE_VAL_PATH
                        (str, default=None) Path to separate val set, optional.
  --separate_test_path SEPARATE_TEST_PATH
                        (str, default=None) Path to separate test set, optional.
  --data_weights_path DATA_WEIGHTS_PATH
                        (str, default=None) Path to weights for each molecule in the training data, affecting the relative weight of molecules in the loss
                        function
  --target_weights [TARGET_WEIGHTS ...]
                        (List[float], default=None) Weights associated with each target, affecting the relative weight of targets in the loss function.
                        Must match the number of target columns.
  --split_type {random,scaffold_balanced,predetermined,crossval,cv,cv-no-test,index_predetermined}
                        (Literal['random', 'scaffold_balanced', 'predetermined', 'crossval', 'cv', 'cv-no-test', 'index_predetermined'], default=random)
                        Method of splitting the data into train/val/test.
  --split_sizes SPLIT_SIZES SPLIT_SIZES SPLIT_SIZES
                        (Tuple[float, float, float], default=(0.8, 0.1, 0.1)) Split proportions for train/validation/test sets.
  --num_folds NUM_FOLDS
                        (int, default=1) Number of folds when performing cross validation.
  --folds_file FOLDS_FILE
                        (str, default=None) Optional file of fold labels.
  --val_fold_index VAL_FOLD_INDEX
                        (int, default=None) Which fold to use as val for leave-one-out cross val.
  --test_fold_index TEST_FOLD_INDEX
                        (int, default=None) Which fold to use as test for leave-one-out cross val.
  --crossval_index_dir CROSSVAL_INDEX_DIR
                        (str, default=None) Directory in which to find cross validation index files.
  --crossval_index_file CROSSVAL_INDEX_FILE
                        (str, default=None) Indices of files to use as train/val/test. Overrides :code:`--num_folds` and :code:`--seed`.
  --seed SEED           (int, default=0) Random seed to use when splitting data into train/val/test sets. When :code`num_folds > 1`, the first fold uses
                        this seed and all subsequent folds add 1 to the seed.
  --pytorch_seed PYTORCH_SEED
                        (int, default=0) Seed for PyTorch randomness (e.g., random initial weights).
  --metric {auc,prc-auc,rmse,mae,mse,r2,accuracy,cross_entropy,binary_cross_entropy}
                        (Literal['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy', 'binary_cross_entropy'], default=None) Metric
                        to use during evaluation. It is also used with the validation set for early stopping. Defaults to "auc" for classification and
                        "rmse" for regression.
  --extra_metrics [{auc,prc-auc,rmse,mae,mse,r2,accuracy,cross_entropy,binary_cross_entropy} ...]
                        (List[Literal['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy', 'binary_cross_entropy']], default=[])
                        Additional metrics to use to evaluate the model. Not used for early stopping.
  --save_dir SAVE_DIR   (str, default=None) Directory where model checkpoints will be saved.
  --checkpoint_frzn CHECKPOINT_FRZN
                        (str, default=None) Path to model checkpoint file to be loaded for overwriting and freezing weights.
  --save_smiles_splits  (bool, default=False) Save smiles for each train/val/test splits for prediction convenience later.
  --test                (bool, default=False) Whether to skip training and only test the model.
  --quiet               (bool, default=False) Skip non-essential print statements.
  --log_frequency LOG_FREQUENCY
                        (int, default=10) The number of batches between each logging of the training loss.
  --show_individual_scores
                        (bool, default=False) Show all scores for individual targets, not just average, at the end.
  --cache_cutoff CACHE_CUTOFF
                        (float, default=10000) Maximum number of molecules in dataset to allow caching. Below this number, caching is used and data loading
                        is sequential. Above this number, caching is not used and data loading is parallel. Use "inf" to always cache.
  --save_preds          (bool, default=False) Whether to save test split predictions during training.
  --resume_experiment   (bool, default=False) Whether to resume the experiment. Loads test results from any folds that have already been completed and
                        skips training those folds.
  --bias                (bool, default=False) Whether to add bias to linear layers.
  --hidden_size HIDDEN_SIZE
                        (int, default=300) Dimensionality of hidden layers in MPN.
  --depth DEPTH         (int, default=3) Number of message passing steps.
  --mpn_shared          (bool, default=False) Whether to use the same message passing neural network for all input molecules Only relevant if
                        :code:`number_of_molecules > 1`
  --dropout DROPOUT     (float, default=0.0) Dropout probability.
  --activation {ReLU,LeakyReLU,PReLU,tanh,SELU,ELU}
                        (Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'], default=ReLU) Activation function.
  --atom_messages       (bool, default=False) Centers messages on atoms instead of on bonds.
  --undirected          (bool, default=False) Undirected edges (always sum the two relevant bond vectors).
  --ffn_hidden_size FFN_HIDDEN_SIZE
                        (int, default=None) Hidden dim for higher-capacity FFN (defaults to hidden_size).
  --ffn_num_layers FFN_NUM_LAYERS
                        (int, default=2) Number of layers in FFN after MPN encoding.
  --features_only       (bool, default=False) Use only the additional features in an FFN, no graph network.
  --separate_val_features_path [SEPARATE_VAL_FEATURES_PATH ...]
                        (List[str], default=None) Path to file with features for separate val set.
  --separate_test_features_path [SEPARATE_TEST_FEATURES_PATH ...]
                        (List[str], default=None) Path to file with features for separate test set.
  --separate_val_atom_descriptors_path SEPARATE_VAL_ATOM_DESCRIPTORS_PATH
                        (str, default=None) Path to file with extra atom descriptors for separate val set.
  --separate_test_atom_descriptors_path SEPARATE_TEST_ATOM_DESCRIPTORS_PATH
                        (str, default=None) Path to file with extra atom descriptors for separate test set.
  --separate_val_bond_features_path SEPARATE_VAL_BOND_FEATURES_PATH
                        (str, default=None) Path to file with extra atom descriptors for separate val set.
  --separate_test_bond_features_path SEPARATE_TEST_BOND_FEATURES_PATH
                        (str, default=None) Path to file with extra atom descriptors for separate test set.
  --config_path CONFIG_PATH
                        (str, default=None) Path to a :code:`.json` file containing arguments. Any arguments present in the config file will override
                        arguments specified via the command line or by the defaults.
  --ensemble_size ENSEMBLE_SIZE
                        (int, default=1) Number of models in ensemble.
  --aggregation {mean,sum,norm}
                        (Literal['mean', 'sum', 'norm'], default=mean) Aggregation scheme for atomic vectors into molecular vectors
  --aggregation_norm AGGREGATION_NORM
                        (int, default=100) For norm aggregation, number by which to divide summed up atomic features
  --reaction            (bool, default=False) Whether to adjust MPNN layer to take reactions as input instead of molecules.
  --reaction_mode {reac_prod,reac_diff,prod_diff}
                        (Literal['reac_prod', 'reac_diff', 'prod_diff'], default=reac_diff) Choices for construction of atom and bond features for
                        reactions :code:`reac_prod`: concatenates the reactants feature with the products feature. :code:`reac_diff`: concatenates the
                        reactants feature with the difference in features between reactants and products. :code:`prod_diff`: concatenates the products
                        feature with the difference in features between reactants and products.
  --explicit_h          (bool, default=False) Whether H are explicitly specified in input (and should be kept this way).
  --epochs EPOCHS       (int, default=30) Number of epochs to run.
  --warmup_epochs WARMUP_EPOCHS
                        (float, default=2.0) Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
                        Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
  --init_lr INIT_LR     (float, default=0.0001) Initial learning rate.
  --max_lr MAX_LR       (float, default=0.001) Maximum learning rate.
  --final_lr FINAL_LR   (float, default=0.0001) Final learning rate.
  --grad_clip GRAD_CLIP
                        (float, default=None) Maximum magnitude of gradient during training.
  --class_balance       (bool, default=False) Trains with an equal number of positives and negatives in each batch.
  --overwrite_default_atom_features
                        (bool, default=False) Overwrites the default atom descriptors with the new ones instead of concatenating them. Can only be used if
                        atom_descriptors are used as a feature.
  --no_atom_descriptor_scaling
                        (bool, default=False) Turn off atom feature scaling.
  --overwrite_default_bond_features
                        (bool, default=False) Overwrites the default atom descriptors with the new ones instead of concatenating them
  --no_bond_features_scaling
                        (bool, default=False) Turn off atom feature scaling.
  --frzn_ffn_layers FRZN_FFN_LAYERS
                        (int, default=0) Overwrites weights for the first n layers of the ffn from checkpoint model (specified checkpoint_frzn), where n is
                        specified in the input. Automatically also freezes mpnn weights.
  --freeze_first_only   (bool, default=False) Determines whether or not to use checkpoint_frzn for just the first encoder. Default (False) is to use the
                        checkpoint to freeze all encoders. (only relevant for number_of_molecules > 1, where checkpoint model has number_of_molecules = 1)
  --smiles_columns [SMILES_COLUMNS ...]
                        (List[str], default=None) List of names of the columns containing SMILES strings. By default, uses the first
                        :code:`number_of_molecules` columns.
  --number_of_molecules NUMBER_OF_MOLECULES
                        (int, default=1) Number of molecules in each input to the model. This must equal the length of :code:`smiles_columns` (if not
                        :code:`None`).
  --checkpoint_dir CHECKPOINT_DIR
                        (str, default=None) Directory from which to load model checkpoints (walks directory and ensembles all models that are found).
  --checkpoint_path CHECKPOINT_PATH
                        (str, default=None) Path to model checkpoint (:code:`.pt` file).
  --checkpoint_paths [CHECKPOINT_PATHS ...]
                        (List[str], default=None) List of paths to model checkpoints (:code:`.pt` files).
  --no_cuda             (bool, default=False) Turn off cuda (i.e., use CPU instead of GPU).
  --gpu {}              (int, default=None) Which GPU to use.
  --features_generator [{morgan,morgan_count,rdkit_2d,rdkit_2d_normalized} ...]
                        (List[str], default=None) Method(s) of generating additional features.
  --features_path [FEATURES_PATH ...]
                        (List[str], default=None) Path(s) to features to use in FNN (instead of features_generator).
  --no_features_scaling
                        (bool, default=False) Turn off scaling of features.
  --max_data_size MAX_DATA_SIZE
                        (int, default=None) Maximum number of data points to load.
  --num_workers NUM_WORKERS
                        (int, default=8) Number of workers for the parallel data loading (0 means sequential).
  --batch_size BATCH_SIZE
                        (int, default=50) Batch size.
  --atom_descriptors {feature,descriptor}
                        (Literal['feature', 'descriptor'], default=None) Custom extra atom descriptors. :code:`feature`: used as atom features to featurize
                        a given molecule. :code:`descriptor`: used as descriptor and concatenated to the machine learned atomic representation.
  --atom_descriptors_path ATOM_DESCRIPTORS_PATH
                        (str, default=None) Path to the extra atom descriptors.
  --bond_features_path BOND_FEATURES_PATH
                        (str, default=None) Path to the extra bond descriptors that will be used as bond features to featurize a given molecule.
  --no_cache_mol        (bool, default=False) Whether to not cache the RDKit molecule for each SMILES string to reduce memory usage (cached by default).
  --empty_cache         (bool, default=False) Whether to empty all caches before training or predicting. This is necessary if multiple jobs are run within
                        a single script and the atom or bond features change.
  -h, --help            show this help message and exit
  ```


