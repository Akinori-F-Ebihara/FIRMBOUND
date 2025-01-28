# FIRMBOUND
This repository contains the PyTorch implementation of FIRMBOUND. The entire code will be available at the time of paper acceptance.

# Quickstart Guide

Follow these steps to utilize the example code provided with our paper:

### Step 1: Generate Sequential Gaussian Dataset
- Run `generate_sequential_gaussian_as_lmdb.ipynb`.
  - Update the header section to generate train, validation, and test LMDB datasets.

### Step 2: Learn to Estimate Log-Likelihood Ratios
- Execute `density_ratio_estimation_main.py`.
  - Update `./config/config_dre.py` to include paths to the LMDB datasets before running the density ratio estimation.

### Step 3: Train the Model
- Train on the generated dataset by running `backward_induction_GP_train.py` or `backward_induction_CFL_train.py`.
  - Set the folder path for saving results in `savedir`.
  - Specify the subproject name created in Step 2 at `subproject`.
  - Set the sampling cost value in `cost_pool`.

### Step 4: Test the Model
- Test on the generated dataset by running `backward_induction_GP_test.py` or `backward_induction_CFL_test.py`.

## Tested Environment
```
python      3.8.10
torch       2.0.0
notebook    6.5.3
optuna      3.1.0
```
