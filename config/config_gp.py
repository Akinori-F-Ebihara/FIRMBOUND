import numpy as np

config = {
    # Shared hyperparameters
    "PENALTY_L": 10.0,
    "TIME_STEPS": 50,
    "IS_LLR_AS_SUFFICIENT_STATISTIC": True,

    # GP-specific hyperparameters
    "EPOCHS_GP": 30,
    "BATCH_SIZE_GP": 2000,
    "NUM_INDUCING_POINTS": 200,

    # Example cost pool: np.array([1.0]) / TIME_STEPS * PENALTY_L
    "COST_POOL": np.array([1.0]) / 50.0 * 10.0,  # -> array([0.2])

    # File/directory parameters
    "SAVE_DIR": "./logs/GPs/estimatedLLRs/",
    "ROOT_DIR": "./",
    "SUBPROJECT": "Example_2clsGaussoptimmacrec_try",

    # For testing
    "GPU": 0,               # GPU ID
    "IS_REAL_WORLD": False,  # Contains LLR (False) or not (True)
    "JITTER_VAL": 1e-5,     # For set_global_cholesky_jitter
}
