import numpy as np
import torch
from loguru import logger

# Main
GPU = "2"
BATCH_SIZE_TUNING = 1000
BATCH_SIZE_FITTING = 5000  # 10000
NUM_FOLDS = 5  # N-fold cross validation. Default=5.
T_MULT = 5  # Num of epochs=num of data * T_MULT; larger->smaller error. Default=3
NAME_DATASET = "synthetic"
LAMBDA_ = None  # None or a positive float(e.g., 0.25). Tuning starts if None.
TUNING_METHOD = ["original", "optuna"][1]  # original grid search or optuna

SAVEDIR = "./logs/CFLs/"
IS_LLR_AS_SUFFICIENT_STATISTIC = False
IS_SAVE_MEMORY = True  # save GPU memory if you run large-class classification
PENALTY_L = 10.0
COST_POOL_BASE = np.array(
    [0.1]
)  # 0.1, 1.0, 2.0 # the actual cost will be multiplied by penalty/time_steps

# Used if TUNING_METHOD = "optuna"
NUM_TRIALS = 30  # number of trials
RANGE_LAMBDA_OPTUNA = [1e-3, 1e1]  # search space (log uniform)
STUDY_NAME = "uniform_sampling"  # This distinguishes DB files.

# Used if TUNING_METHOD = "original"
MAX_HYPER_ITER = 10  # Max num of tuning trials
LS_LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]  # Do not change lengh
LS_LAMBDAS_MULT_LOW = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1]  # Do not change length
LS_LAMBDAS_MULT_HIGH = [1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5]  # Do not change length
LS_LAMBDAS_MULT_FINE = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]  # Can be changed


# Dataset kwargs
KWARGS_DATASETS = {
    "synthetic": {
        "NUM_DATA": 1_000,  # 20,000 gives
        "DIM_DATA": 2,
        "SCALE_NOISE": 1e-1,
        "FLAG_NOISE": True,
        "NAME_FUNCTION": "exp",
    },
    "your dataset": {
        "ADD SMTH": None,
    },
}

# Other global settings
EPSILON = 1e-12
DEFAULT_DTYPE_NP = np.float32
DEFAULT_DTYPE_PT = torch.float32

# Assert double checks
assert NAME_DATASET in [
    "synthetic",
    "your dataset",
]

# assert L and c's searchspace
for targ_cost in COST_POOL_BASE:
    if targ_cost > PENALTY_L / 2:
        logger.error("cost too high, skipping this combination...")
        logger.error(f"{PENALTY_L=}, {targ_cost=}")

# Do not forget to add new parameters after adding ones.
config = {
    "EPSILON": EPSILON,
    "DEFAULT_DTYPE_NP": DEFAULT_DTYPE_NP,
    "DEFAULT_DTYPE_PT": DEFAULT_DTYPE_PT,
    "GPU": GPU,
    "BATCH_SIZE_TUNING": BATCH_SIZE_TUNING,
    "BATCH_SIZE_FITTING": BATCH_SIZE_FITTING,
    "NUM_FOLDS": NUM_FOLDS,
    "MAX_HYPER_ITER": MAX_HYPER_ITER,
    "T_MULT": T_MULT,
    "LS_LAMBDAS": LS_LAMBDAS,
    "LS_LAMBDAS_MULT_LOW": LS_LAMBDAS_MULT_LOW,
    "LS_LAMBDAS_MULT_HIGH": LS_LAMBDAS_MULT_HIGH,
    "LS_LAMBDAS_MULT_FINE": LS_LAMBDAS_MULT_FINE,
    "KWARGS_DATASETS": KWARGS_DATASETS,
    "NUM_TRIALS": NUM_TRIALS,
    "TUNING_METHOD": TUNING_METHOD,
    "RANGE_LAMBDA_OPTUNA": RANGE_LAMBDA_OPTUNA,
    "NAME_DATASET": NAME_DATASET,
    "LAMBDA_": LAMBDA_,
    "STUDY_NAME": STUDY_NAME,
    "PENALTY_L": PENALTY_L,
    "COST_POOL_BASE": COST_POOL_BASE,
    "IS_LLR_AS_SUFFICIENT_STATISTIC": IS_LLR_AS_SUFFICIENT_STATISTIC,
    "IS_SAVE_MEMORY": IS_SAVE_MEMORY,
    "SAVEDIR": SAVEDIR,
}
