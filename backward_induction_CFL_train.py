import glob

import numpy as np
import torch
from loguru import logger

from config.config_cfl import config as config_cfl
from models.backward_induction_utils import CFL_training_routine, extract_llrs_from_dataset

"""USER DEFINED PARAMETERS"""
penalty_L = 10.0
time_steps = 50
is_llr_as_sufficient_statistic = False
# cost_pool = np.array([0.1, 1.0, 2.0]) / time_steps * penalty_L
cost_pool = np.array([1.0]) / time_steps * penalty_L
root_dir = "./"
subproject = "Example_2clsGaussoptimmacrec_try"
"""USER DEFINED PARAMETERS END"""

proj_path = root_dir + "logs/" + subproject + "/checkpoints/"
glob_paths = glob.glob(proj_path + "*")

cnt_model = 0
for path_modeldir in glob_paths:
    logger.info("\nModel Num {}/{}: ".format(cnt_model + 1, len(glob_paths)), path_modeldir)
    key = path_modeldir.split("/")[-1]
    # Load the train data
    llrs, labels, cnt_model, num_data, time_steps = extract_llrs_from_dataset(
        path_modeldir,
        phase="train",  # 'train', 'val', or 'test
        gpu=0,
        cnt_model=cnt_model,
    )

    for targ_cost in cost_pool:
        if targ_cost > penalty_L / 2:
            logger.info("cost too high, skipping this combination...")
            logger.info(f"{penalty_L=}, {targ_cost=}")
            continue

        model_pool = CFL_training_routine(
            torch.tensor(llrs, device="cuda", dtype=torch.float32),
            penalty_L,
            targ_cost,
            "estimatedLLRs/" + subproject + "/" + key,
            config_cfl,
        )
