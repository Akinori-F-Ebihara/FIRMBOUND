import glob
from typing import cast

import numpy as np
import torch
from loguru import logger

from config.config_cfl import config as config_cfl
from models.backward_induction_utils import (
    calc_sns,
    decision_helper_cfl,
    extract_llrs_from_dataset,
    load_cfl_models,
)

"""USER DEFINED PARAMETERS"""
penalty_L = 10.0
time_steps = 50
is_llr_as_sufficient_statistic = False
# cost_pool = np.array([0.1, 1.0, 2.0]) / time_steps * penalty_L
cost_pool = np.array([1.0]) / time_steps * penalty_L
savedir = "./logs/CFLs/estimatedLLRs/"
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
    llrs_all, labels_all, cnt_model, num_data, time_steps = extract_llrs_from_dataset(
        path_modeldir,
        phase="test",  # 'train', 'val', or 'test
        gpu=0,
        cnt_model=cnt_model,
    )
    iter_num = int(np.ceil(num_data / config_cfl["BATCH_SIZE_TUNING"]))

    # randomly select maximum batch used for computation
    assert len(llrs_all) == len(labels_all)
    max_data_num = np.max(
        (cast(int, config_cfl["BATCH_SIZE_TUNING"]), cast(int, config_cfl["BATCH_SIZE_FITTING"]))
    )
    dice = np.random.randint(0, len(llrs_all), max_data_num)
    llrs = llrs_all[dice]
    labels = labels_all[dice]

    for targ_cost in cost_pool:
        if targ_cost > penalty_L / 2:
            logger.info("cost too high, skipping this combination...")
            logger.info(f"{penalty_L=}, {targ_cost=}")
            continue

        # load model
        targ_dict = (
            f"{savedir}/{subproject}/{key}/penalty{penalty_L}_cost{targ_cost}"
            f"_bst{config_cfl['BATCH_SIZE_TUNING']}bsf{config_cfl['BATCH_SIZE_FITTING']}"
            f"_{config_cfl['NUM_FOLDS']}fold_tmult{config_cfl['T_MULT']}"
            f"_{config_cfl['RANGE_LAMBDA_OPTUNA']}_isllrs{is_llr_as_sufficient_statistic}/"
        )
        model_pool = load_cfl_models(
            model_path=targ_dict + "CFLmodels.pt",
        )

        # start batch process
        hitting_time = np.ones(num_data)
        predictions = np.ones(num_data)
        for iter in range(iter_num):
            if iter % 10 == 0:
                logger.info(f"Iterating over LLRs...{iter+1}/{iter_num}")
            idx_start = iter * cast(int, config_cfl["BATCH_SIZE_TUNING"])
            idx_end = (iter + 1) * cast(int, config_cfl["BATCH_SIZE_TUNING"])

            llrs_batch = torch.tensor(
                llrs_all[idx_start:idx_end], device="cuda", dtype=torch.float32
            )
            labels_batch = torch.tensor(
                labels_all[idx_start:idx_end], device="cuda", dtype=torch.int64
            )

            ht_batch, pred_batch = decision_helper_cfl(
                model_pool,
                penalty_L,
                targ_cost,
                llrs_batch,
                is_llr_as_sufficient_statistic,
            )
            hitting_time[idx_start:idx_end] = ht_batch.cpu().numpy()
            predictions[idx_start:idx_end] = pred_batch.cpu().numpy()

        sns = calc_sns(
            torch.tensor(labels_all, device="cpu", dtype=torch.int64),
            torch.tensor(predictions, device="cpu", dtype=torch.int64),
        )
        logger.success(f"Result with {targ_cost=}")
        logger.success(f"Average per-class error rate:{1.-torch.mean(sns)}")
        logger.success(f"Mean hitting time:{np.mean(hitting_time)}")

    logger.success("ALL DONE AND DUSTED!!")
