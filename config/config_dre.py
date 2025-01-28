# Assumption:
# this file is placed inside a directory named "config", under the project root directory.

import numpy as np

from utils.misc import compile_directory_paths, compile_subproject_name

""" USER MODIFIABLE PARAMS """

# Base info
CONFIG_PATH = __file__
LOG_PATH = CONFIG_PATH[: CONFIG_PATH.find("config")] + "logs/"

# Data info
NUM_CLASSES = 2
DATA_SEPARATION = "0.5"
DATA_PATH = "./tmp/"
DATA_FOLDER = f"ExampleGaussian_{NUM_CLASSES}class_offset{DATA_SEPARATION}"  # _dim2
NUM_TRAIN = 20000  # 40000  # 300000  # 25000 # 19000
NUM_VAL = 2000  # 4000  # 30000  # 4000 # # 990
NUM_TEST = 2000  # 6000  # 4000  # 30000  # 1000 # 10
FEAT_DIM = 128  # 2  # 128
IS_SAVE_GPU_MEMORY = False  # move performance metrics from cuda to cpu
TRAIN_DATA = f"{DATA_PATH}{DATA_FOLDER}/train_{NUM_TRAIN}"
VAL_DATA = f"{DATA_PATH}{DATA_FOLDER}/val_{NUM_VAL}"
TEST_DATA = f"{DATA_PATH}{DATA_FOLDER}/test_{NUM_TEST}"
IS_SHUFFLE = True  # whether to shuffle the train data
DATA_NAMES = ("data", "label", "llr")
TB_DIRNAME = "TensorBoard_events"
DB_DIRNAME = "Optuna_databases"
CKPT_DIRNAME = "checkpoints"
STDOUT_DIRNAME = "stdout_logs"

EXP_PHASE = "try"  # 'try', 'tuning', or 'stat'
SUBPROJECT_NAME_PREFIX = "Example_2clsGauss"
SUBPROJECT_NAME_SUFFIX = ""
COMMENT = ""  # for the log file name
OPTIMIZATION_CANDIDATES = (
    "macrec",
    "mabs",
)
OPTIMIZATION_TARGET = "macrec"

# if EXP_PHASE=='tuning', these params will be overwritten
# with a new parameter sampled by Optuna.
MODEL_BACKBONE = "LSTM"
# non-negative value and must be less than time_steps of the data.
ORDER_SPRT = 0
PARAM_MULTIPLET_LOSS = 0.7
LLLR_VERSION = "LLLR"
PARAM_LLR_LOSS = 0.1
AUCLOSS_VERSION = "Anorm"
AUCLOSS_BURNIN = 1500
PARAM_AUSAT_LOSS = 0.0
PARAM_SAMPLING_LOSS = 0.0

# Reproduce a particular trial or the best trial from specified subproject.
REPRODUCE_TRIAL = False  # 'best', trial index (int) or False
# Load pretrained weight from .pt file specified below.
IS_RESUME = False
SUBPROJECT_TO_RESUME = "_20230330_214519224/ckpt_step3500_target_ausat_confmx0.0011.pt"

# of frequent use
GPU = "0"
WORLD_SIZE = len(GPU.split(","))
BATCH_SIZE = 200  # The effective batch size will be WORLD_SIZE * BATCH_SIZE
NUM_TRIALS = 1
NUM_EPOCHS = 15

NUM_ITER_PER_EPOCH = int(np.ceil(NUM_TRAIN / BATCH_SIZE / WORLD_SIZE))
NUM_ITER = NUM_EPOCHS * NUM_ITER_PER_EPOCH  # e.g., 20 * 25000 // 100 = 5000

TRAIN_DISPLAY_STEP = NUM_ITER // NUM_EPOCHS
VALIDATION_STEP = TRAIN_DISPLAY_STEP
# hyperband, median, percentile, etc... set to 'none' if no pruner is needed.
PRUNER_NAME = "percentile"
PRUNER_STARTUP_TRIALS = NUM_TRIALS // 2
PRUNER_WARMUP_STEPS = NUM_ITER // 2
PRUNER_INTERVAL_STEPS = TRAIN_DISPLAY_STEP

""" USER MODIFIABLE PARAMS END """

info_for_subproject_name = {
    "DATA_SEPARATION": DATA_SEPARATION,
    "OPTIMIZATION_TARGET": OPTIMIZATION_TARGET,
    "MODEL_BACKBONE": MODEL_BACKBONE,
    "PRUNER_NAME": PRUNER_NAME,
    "ORDER_SPRT": ORDER_SPRT,
    "LLLR_VERSION": LLLR_VERSION,
    "PARAM_LLR_LOSS": PARAM_LLR_LOSS,
    "AUCLOSS_VERSION": AUCLOSS_VERSION,
    "PARAM_AUSAT_LOSS": PARAM_AUSAT_LOSS,
    "AUCLOSS_BURNIN": AUCLOSS_BURNIN,
    "PARAM_MULTIPLET_LOSS": PARAM_MULTIPLET_LOSS,
    "EXP_PHASE": EXP_PHASE,
    "SUBPROJECT_NAME_PREFIX": SUBPROJECT_NAME_PREFIX,
    "SUBPROJECT_NAME_SUFFIX": SUBPROJECT_NAME_SUFFIX,
}

SUBPROJECT_NAME = compile_subproject_name(info_for_subproject_name)

subconfig = {
    "LOG_PATH": LOG_PATH,
    "TB_DIRNAME": TB_DIRNAME,
    "DB_DIRNAME": DB_DIRNAME,
    "CKPT_DIRNAME": CKPT_DIRNAME,
    "STDOUT_DIRNAME": STDOUT_DIRNAME,
    "SUBPROJECT_NAME": SUBPROJECT_NAME,
    "SUBPROJECT_TO_RESUME": SUBPROJECT_TO_RESUME,
}

compile_directory_paths(subconfig)

config = {
    "VERBOSE": True,
    "CONFIG_PATH": CONFIG_PATH,
    "IS_SEED": False,
    "SEED": 7,
    "GPU": GPU,
    "WORLD_SIZE": WORLD_SIZE,
    "MAX_NORM": 50000,  # used for gradient clipping
    "IS_SAVE_GPU_MEMORY": IS_SAVE_GPU_MEMORY,
    # for logging
    "NAME_DATASET": "Multivariate_Gaussian",
    "COMMENT": COMMENT,
    "IS_SAVE_FIGURE": True,
    "DATA_SEPARATION": DATA_SEPARATION,
    "SUBPROJECT_NAME_PREFIX": SUBPROJECT_NAME_PREFIX,
    "SUBPROJECT_NAME_SUFFIX": SUBPROJECT_NAME_SUFFIX,
    # SDRE datasets
    "TRAIN_DATA": TRAIN_DATA,
    "VAL_DATA": VAL_DATA,
    "TEST_DATA": TEST_DATA,
    "DATA_NAMES": DATA_NAMES,
    "IS_SHUFFLE": IS_SHUFFLE,
    "NUM_TRAIN": NUM_TRAIN,
    "NUM_VAL": NUM_VAL,
    "NUM_TEST": NUM_TEST,
    "NUM_CLASSES": NUM_CLASSES,
    "FEAT_DIM": FEAT_DIM,
    "IS_CLASS_BALANCE": False,
    # SPRT-TANDEM parameters
    "ORDER_SPRT": ORDER_SPRT,
    "TIME_STEPS": 50,
    "LLLR_VERSION": LLLR_VERSION,  # LSEL
    "OBLIVIOUS": False,
    "PARAM_MULTIPLET_LOSS": PARAM_MULTIPLET_LOSS,
    "PARAM_LLR_LOSS": PARAM_LLR_LOSS,
    # Network specs
    # LSTM (B2Bsqrt-TANDEM) or Transformer (TANDEMformer)
    "MODEL_BACKBONE": MODEL_BACKBONE,
    "ACTIVATION_OUTPUT": "B2BsqrtV2",
    "ACTIVATION_FC": "relu",
    "ALPHA": 1.0,  # for B2Bsqrt activation function
    "IS_NORMALIZE": False,  # whether to use Normalization or not
    "IS_POSITIONAL_ENCODING": False,
    "IS_TRAINABLE_ENCODING": False,
    # For LSTM backbone
    "ACTIVATION_INPUT": "tanh",  # "sigmoid",
    "WIDTH_LSTM": 106,
    # For TRANSFORMER backbone
    "NUM_BLOCKS": 1,  # transformer block
    "NUM_HEADS": 2,
    "DROPOUT": 0.3,
    "FF_DIM": 32,
    "MLP_UNITS": 64,
    # For S4 backbone
    "ACTIVATION_S2F": "gelu",
    "D_STATE": 16,
    # For metanet / tapering threshold
    "ACTIVATION_THRESH": "B2Bsqrt",
    # for performance
    "IS_COMPILE": False,  # torch.compile (pytorch > 2.0)
    "MODE": "reduce-overhead",  # 'reduce-overhead', 'default', or 'max-autotune'
    "NUM_WORKERS": 0,  # num_workers argument for pytorch dataloader
    # whether to load dataset onto memory at initialization.
    "IS_LOAD_ONTO_MEMORY": True,
    # SAT curve
    "AUCLOSS_VERSION": AUCLOSS_VERSION,
    "IS_MULT_SAT": False,  # SAT curve can be truncated if False
    "NUM_THRESH": 2000,
    "IS_TAPERING_THRESHOLD": False,  # obsolete: use trainable tapering threshould
    # "linspace", "logspace", "unirandom", or "lograndom".
    "SPARSITY": "logspace",
    "BETA": 1.0,
    "PARAM_AUSAT_LOSS": PARAM_AUSAT_LOSS,
    "AUCLOSS_BURNIN": AUCLOSS_BURNIN,
    "PARAM_SAMPLING_LOSS": PARAM_SAMPLING_LOSS,
    # Training parameters
    "OPTIMIZATION_TARGET": OPTIMIZATION_TARGET,
    "OPTIMIZATION_CANDIDATES": OPTIMIZATION_CANDIDATES,
    "PRUNER_STARTUP_TRIALS": PRUNER_STARTUP_TRIALS,
    "PRUNER_WARMUP_STEPS": PRUNER_WARMUP_STEPS,
    "PRUNER_INTERVAL_STEPS": PRUNER_INTERVAL_STEPS,
    # hyperband, median, percentile, etc... set to 'none' if no pruner is needed.
    "PRUNER_NAME": PRUNER_NAME,
    "MAX_TO_KEEP": 1,
    "IS_SAVE_EVERYTIME": False,  # save checkpoints even when the result did not update the best value
    "EXP_PHASE": EXP_PHASE,
    "REPRODUCE_TRIAL": REPRODUCE_TRIAL,
    "NUM_TRIALS": NUM_TRIALS,
    "OPTUNA_BURNIN_TRIALS": int(NUM_TRIALS * 0.2),
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 0.00025186,
    "LR_DECAY_STEPS": [
        100000000,
    ],
    "SCHEDULER": "StepLR",
    "SCHEDULER_OPTIONS": {
        "Constant": {},
        "StepLR": {"step_size": 7, "gamma": 0.1},
        "MultiStepLR": {"milestones": [4, 6, 8], "gamma": 0.1},
        "ExponentialLR": {"gamma": 0.9},
        "CosineAnnealingLR": {"T_max": 20, "eta_min": 0},
        "CosineAnnealingWarmRestarts": {"T_0": 3, "T_mult": 1, "eta_min": 0},
        "ReduceLROnPlateau": {"factor": 0.5, "patience": 3, "threshold": 1e-4},
    },
    "WEIGHT_DECAY": 0.0003,
    "OPTIMIZER": "lion",
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_ITER": NUM_ITER,  # 7500
    "TRAIN_DISPLAY_STEP": TRAIN_DISPLAY_STEP,
    "VALIDATION_STEP": VALIDATION_STEP,
    "IS_RESUME": IS_RESUME,
    "IS_SKIP_NAN": False,  # raise optuna.TrialPruned() when LLR matrix contains NaN
    "IS_SKIP_OOM": False,  # raise optuna.TrialPruned() when CUDA Out Of Memory (OOM)
    "IS_REAL_WORLD": False,
    # Optuna hyperparameter search space
    # Key values starting "SPACE_" containing parameter search space information.
    # PARAM_SPACE: "float", "int", or "categorical".
    # - float: use suggest_float to suggest a float of range [LOW, HIGH], separated by STEP.
    #   if LOG=True, a float is sampled from logspace but you shall set STEP=None.
    # - int: use suggest_int to suggest an int of range [LOW, HIGH], separated by STEP.
    #   STEP should be divisor of the range, otherwise HIGH will be automatically modified.
    #   if LOG=True, an int is sampled from logspace but you shall set STEP=None.
    # - categorical: use suggest_categorical to select one category from CATEGORY_SET.
    #   Note that if the parameter is continuous (e.g., 1, 2, 3, ..., or 1.0, 0.1, 0.001, ...),
    #   it is adviseable to use float or int space because suggest_categorical treats
    #   each category independently.
    "SPACE_MODEL_BACKBONE": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": ["Transformer"],  # 'LSTM', 'Transformer', 'S4'
    },
    # common to all backbones
    "SPACE_WEIGHT_DECAY": {
        "PARAM_SPACE": "float",
        "LOW": 0.0,
        "HIGH": 0.001,
        "STEP": 0.0001,
        "LOG": False,  # log is preferable but it doesn't allow LOW==0.0
    },
    "SPACE_LEARNING_RATE": {
        "PARAM_SPACE": "float",
        "LOW": 0.000001,
        "HIGH": 0.001,
        "STEP": None,
        "LOG": True,
    },
    "SPACE_LLLR_VERSION": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": ["LLLR", "LSEL"],  # ["A", "B", "C", "D", "E", "Eplus"],
    },
    "SPACE_ACTIVATION_FC": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": ["B2Bsqrt", "B2BsqrtV2", "B2Bcbrt", "tanh", "relu", "gelu"],
    },
    "SPACE_ACTIVATION_THRESH": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": ["B2Bsqrt", "B2BsqrtV2", "B2Bcbrt", "tanh", "B2Blog"],
    },
    "SPACE_PARAM_MULTIPLET_LOSS": {
        "PARAM_SPACE": "float",
        "LOW": 0.0,
        "HIGH": 1.0,
        "STEP": 0.1,
        "LOG": False,
    },
    "SPACE_PARAM_LLR_LOSS": {
        "PARAM_SPACE": "float",
        "LOW": 0.0,
        "HIGH": 1.0,
        "STEP": 0.1,
        "LOG": False,
    },
    "SPACE_OPTIMIZER": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": ["rmsprop", "adam", "lion"],
    },
    "SPACE_SCHEDULER": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": [
            "Constant",
            "StepLR",
            "MultiStepLR",
            "ExponentialLR",
            "CosineAnnealingLR",
            "CosineAnnealingWarmRestarts",
            "ReduceLROnPlateau",
        ],
    },
    "SPACE_IS_NORMALIZE": {"PARAM_SPACE": "categorical", "CATEGORY_SET": [True, False]},
    "LSTM": {
        "SPACE_ACTIVATION_OUTPUT": {
            "PARAM_SPACE": "categorical",
            "CATEGORY_SET": ["B2Bsqrt", "B2BsqrtV2", "B2Bcbrt", "tanh", "sigmoid"],
        },
        "SPACE_ACTIVATION_INPUT": {
            "PARAM_SPACE": "categorical",
            "CATEGORY_SET": ["tanh", "sigmoid"],
        },
        "SPACE_WIDTH_LSTM": {
            "PARAM_SPACE": "int",
            "LOW": 16,
            "HIGH": 128,
            "STEP": 1,
            "LOG": True,
        },
    },
    # TANDEMformer
    "TRANSFORMER": {
        "SPACE_NUM_BLOCKS": {
            "PARAM_SPACE": "int",
            "LOW": 1,
            "HIGH": 2,
            "STEP": 1,
            "LOG": False,
        },
        "SPACE_NUM_HEADS": {  # num_heads must be a divisor of feat_dim (=embed_dim)
            "PARAM_SPACE": "int",
            "LOW": 2,
            "HIGH": 4,
            "STEP": 2,
            "LOG": False,
        },
        "SPACE_DROPOUT": {
            "PARAM_SPACE": "float",
            "LOW": 0.0,
            "HIGH": 0.5,
            "STEP": 0.1,
            "LOG": False,
        },
        "SPACE_FF_DIM": {
            "PARAM_SPACE": "int",
            "LOW": 32,
            "HIGH": 64,
            "STEP": 32,
            "LOG": False,
        },
        "SPACE_MLP_UNITS": {
            "PARAM_SPACE": "int",
            "LOW": 32,
            "HIGH": 64,
            "STEP": 32,
            "LOG": False,
        },
    },
    # Some combination of hyperparameters are incompatible or ineffective.
    "FORBIDDEN_PARAM_SETS": {
        "loss_all_zero": {
            "SPACE_PARAM_MULTIPLET_LOSS": 0.0,
            "SPACE_PARAM_LLR_LOSS": 0.0,
        },
    },
}

config.update(subconfig)
