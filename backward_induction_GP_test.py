import glob

import numpy as np
import torch
from loguru import logger

from models.backward_induction_utils import (
    calc_sns,
    decision_helper,
    extract_llrs_from_dataset,
    load_gaussian_process_models,
    set_global_cholesky_jitter,
)

# NOTE: config file will be loaded from the subproject dir in the extract_llrs_from_dataset folder.

"""USER DEFINED PARAMETERS"""
gpu = int(0)
penalty_L = 10.0
epochs_gp = 30
batch_size_gp = 2000
num_inducing_points = 200
is_llr_as_sufficient_statistic = True
time_steps = 50
# cost_pool = np.array([0.1, 1.0, 2.0]) / time_steps * penalty_L
cost_pool = np.array([1.0]) / time_steps * penalty_L
savedir = "./logs/GPs/estimatedLLRs"  # data to be saved here
root_dir = "./"
subproject = "Example_2clsGaussoptimmacrec_try"
is_real_world = True
jitter_val = 1e-5
"""USER DEFINED PARAMETERS END"""

set_global_cholesky_jitter(jitter_val)
proj_path = root_dir + "logs/" + subproject + "/checkpoints/"
glob_paths = glob.glob(proj_path + "*")

cnt_model = 0
for path_modeldir in glob_paths:
    logger.info("\nModel Num {}/{}: ".format(cnt_model + 1, len(glob_paths)))

    key = path_modeldir.split("/")[-1]

    # Load the test data. This is the estimated LLRs: thus need to be loaded for each model
    llrs_all, labels_all, cnt_model, num_data, time_steps = extract_llrs_from_dataset(
        path_modeldir,
        phase="test",  # 'train', 'val', or 'test
        gpu=gpu,
        cnt_model=cnt_model,
    )

    iter_num = int(np.ceil(num_data / batch_size_gp))

    for targ_cost in cost_pool:
        if targ_cost > penalty_L / 2:
            logger.info("cost too high, skipping this combination...")
            logger.info(f"{penalty_L=}, {targ_cost=}")
            continue
        targ_dict = (
            f"{savedir}/{subproject}/{key}/penalty{penalty_L}_cost{targ_cost}_"
            f"batchsize{batch_size_gp}_indpnt{num_inducing_points}_epoch{epochs_gp}_"
            f"isllrs{is_llr_as_sufficient_statistic}/"
        )

        # load model and likelihood
        model_pool, likelihood_pool = load_gaussian_process_models(
            model_path=targ_dict + "GPmodels.pt",
            likelihood_path=targ_dict + "Likelihoodmodels.pt",
        )

        # start batch process
        hitting_time = np.ones(num_data)
        predictions = np.ones(num_data)
        for iter in range(iter_num):
            if iter % 10 == 0:
                logger.info(f"Iterating over LLRs...{iter+1}/{iter_num}")
            idx_start = iter * batch_size_gp
            idx_end = (iter + 1) * batch_size_gp

            llrs_batch = torch.tensor(
                llrs_all[idx_start:idx_end], device="cuda", dtype=torch.float32
            )

            labels_batch = torch.tensor(
                labels_all[idx_start:idx_end], device="cuda", dtype=torch.int64
            )

            ht_batch, pred_batch = decision_helper(
                model_pool,
                likelihood_pool,
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
        logger.success(f"Average per-class error rate:{100*(1.-torch.mean(sns))}%")
        logger.success(f"Mean hitting time:{np.mean(hitting_time)}")

logger.success("ALL DONE AND DUSTED!!")
