# import tensorboard # just for launching TensorBoard Session on VSCode

import copy
from typing import Any, Dict, Tuple

import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
from optuna.integration import TorchDistributedTrial
from optuna.trial import Trial
from termcolor import colored
from tqdm import tqdm

from config.config_definition import config as config_orig
from utils.checkpoint import finalize_objectives
from utils.hyperparameter_tuning import run_optuna, suggest_parameters
from utils.logging import ContexualLogger, create_log_folders
from utils.misc import parse_args
from utils.training import (
    ddp_setup,
    finalize_training,
    find_free_port,
    initialize_training,
    manage_single_epoch,
    prepare_for_training,
    set_world_size,
    suppress_unnecessary_warnings,
)


def objective(trial: Trial, config: Dict[str, Any]) -> Tuple[float, ...]:
    """
    Defines the objective function for Optuna to optimize.

    Args:
    - trial: An optuna Trial object.
    - config: A dictionary containing model configuration parameters.

    Returns:
    - Any: The value of the objective function.
    """
    suppress_unnecessary_warnings()

    # For distributed training
    _trial = TorchDistributedTrial(trial)

    # copy and tune for each trial
    _config = copy.deepcopy(config)

    # Suggest parameters if the experimental phase is "tuning."
    # This function may modify config in-place.
    suggest_parameters(_trial, _config)

    # This function modifies config
    create_log_folders(_config)

    # Outputs under this context will be logged into one file.
    # The log file will be investigated earch time the code exits from
    # the context. Error will be raised when error logs are found.
    with ContexualLogger(_config, is_stop_at_error=False):
        training_tools = prepare_for_training(_config)

        best, global_step = initialize_training(_config)

        for epoch in tqdm(
            range(_config["NUM_EPOCHS"]),
            mininterval=5,
            desc=colored("Epoch progress: ", "blue"),
            colour="blue",
            disable=dist.get_rank() != 0,
        ):
            logger.info(f"Starting epoch #{epoch}.")
            dist.barrier()

            best, global_step = manage_single_epoch(
                _trial, training_tools, _config, best, global_step, epoch
            )

        best, global_step = finalize_training(
            _trial, training_tools, _config, best, global_step, epoch
        )

    return finalize_objectives(best)


def main(rank: int, world_size: int, port: int, config: Dict) -> None:
    # Initialize process group and set address/port
    ddp_setup(rank, world_size, port, config)

    # Start learning!
    run_optuna(objective, config)

    # Finalize Distributed Data Parallel
    dist.destroy_process_group()


if __name__ == "__main__":
    # Use arguments to overwrite the config file
    parse_args(config_orig)

    # Get a port to communicate across GPUs
    port = find_free_port()

    # Get GPU device count
    world_size = set_world_size(config_orig)

    # Spawn multiple processes
    mp.spawn(main, args=(world_size, port, config_orig), nprocs=world_size)
