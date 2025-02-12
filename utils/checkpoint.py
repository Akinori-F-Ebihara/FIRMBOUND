import glob
import os
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from termcolor import colored

from utils.misc import convert_torch_to_numpy, extract_params_from_config


@logger.catch
def initialize_objectives(config):
    """ """
    # check if necessary parameters are defined in the config file
    requirements = set(["OPTIMIZATION_CANDIDATES", "OPTIMIZATION_TARGET"])
    conf = extract_params_from_config(requirements, config)
    if "all" in conf.optimization_target.lower():
        num_candidates = len(conf.optimization_candidates)
        return tuple([np.Inf] * num_candidates)  # multiobjectives:
    else:
        return tuple([np.Inf])


def finalize_objectives(best):
    """
    Args:
    -best (tuple): objective(s) to be minimized.
    """

    if len(best) == 1:
        return best[0]
    elif len(best) > 1:
        best0, *remaining_bests = best
        return best0, *remaining_bests


def update_and_save_result(
    model, optimizer, scheduler, config, best, performance_metrics, global_step
):
    """
    Update the best result according to the optimization target.

    Args:
    - config (dict): Configuration dictionary that contains the optimization target.
    - ckpt_manager: (TensorFlow checkpoint manager)
    - best (tuple): current best values of the objective
    - performance_metrics (dict): dictionary containing mean macro-average recall (macrec),
      AUSAT_optimization loss, and mean absolute error rate (mean_abs_error)
    - global_step (int): Current iteration number.
    Returns:
    - best (tuple): The updated best value.
    """

    def concatenate_variables(variables: List, n_digits: int) -> str:
        """ """

        def format_float(f, n_digits):
            return "{{:.{}f}}".format(n_digits).format(
                np.round(f * 10**n_digits) / 10**n_digits
            )

        # make double sure they're numpy
        variables = convert_torch_to_numpy(variables)
        variables = [format_float(v, n_digits) for v in variables]
        return "_".join(variables)

    def save_checkpoint(model, optimizer, scheduler, best, conf, global_step):
        # save only if rank:0
        if dist.get_rank() != 0:
            return

        # save a checkpoint
        model_path = (
            f"{conf.dir_ckptlogs}/"
            f"ckpt_step{global_step}_target_"
            f"{conf.optimization_target}{concatenate_variables(best, n_digits=4)}"
            f".pt"
        )

        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            model_path,
        )

        logger.info(colored("Saved checkpoint", "cyan") + f" at step {global_step}.")

    def keep_max_num_saved_models(conf: SimpleNamespace, max_to_keep: int = 1) -> None:
        """
        This function ensures that PyTorch saves a maximum number of
        checkpoint files (.pth format) in a given directory.
        If there are more saved models than the specified limit,
        it will delete the oldest ones, based on their creation time.

        Args:
        - conf (dict): stores conf.ckptlogs to specify checkpoint log directory.
        - max_to_keep (int, optional): The maximum number of checkpoint files to save in the directory. Defaults to 1.
        """
        # Get a list of all files in the directory
        models = glob.glob(f"{conf.dir_ckptlogs}/" + "*.pt")

        # Sort the files based on their creation time
        models = sorted(models, key=os.path.getmtime)
        if len(models) > max_to_keep:
            os.remove(models[0])
            logger.info(f"Removed the oldest model for {max_to_keep=}")

    def targets_sanity_check(targets, conf_targets) -> dict:
        """
        Check the optimization candidates defined in the config file and optimization targets match.
        Sort the order of targets if it is incongruent with that of the config file.

        Args:
        - targets (dict): optuna optimization targets. Note that all targets will be optimized
                          when config['OPTIMIATION_TARGET] is 'all.'
        - conf_targets (list): optimization targets listed in the config file.

        Returns:
        - sorted_targets (dict): sorted targets according to the order of conf_targets.
        """
        # Assert the number of key-value pairs of the two dictionaries are equal
        assert len(conf_targets) == len(targets)

        # Check if a set of values of 'conf' are same as a set of keys of 'target'
        assert set(conf_targets) == set(targets.keys())

        # Sort the order of target so that 'conf's keys and target's values are in the same order
        sorted_targets = {}
        for key in conf_targets:
            sorted_targets[key] = targets[key]
        return sorted_targets

    # check if necessary parameters are defined in the config file
    requirements = set(
        [
            "OPTIMIZATION_CANDIDATES",
            "OPTIMIZATION_TARGET",
            "DIR_CKPTLOGS",
            "MAX_TO_KEEP",
            "IS_SAVE_EVERYTIME",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    # Optuna *minimize* the objective by default - give it an error, not accuracy or equivalent
    targets = {
        # This is an AUSAT "loss", which should be minimized
        # "ausat_loss": performance_metrics["losses"]["AUSAT_optimization"],
        # MacRec should be larger... flip the sign
        "macrec": 1 - performance_metrics["mean_macro_recall"],
        # MABS should be smaller.
        "mabs": performance_metrics["mean_abs_error"],
        # AUSAT "curve" from confmx has accuracy as y-axis: flip it
        # "ausat_confmx": 1 - performance_metrics["ausat_from_confmx"],
    }

    targets = targets_sanity_check(targets, conf.optimization_candidates)

    if "all" in conf.optimization_target.lower():
        target = tuple(targets.values())

    elif conf.optimization_target.lower() in targets.keys():
        target = tuple([targets.get(conf.optimization_target.lower())])

    else:
        raise ValueError("Unknown optimization target!")

    indices = [i for i, (x, y) in enumerate(zip(target, best)) if x < y]  # smaller is better
    if global_step == 0:
        best = target

    elif indices:  # update the best value!
        best = tuple([target[i] if i in indices else x for i, x in enumerate(best)])
        logger.info(colored("Best value updated!", "cyan"))

        save_checkpoint(model, optimizer, scheduler, best, conf, global_step)
        keep_max_num_saved_models(conf, conf.max_to_keep)

    elif (
        conf.is_save_everytime
    ):  # save checkpoints even when the result did not update the best value
        save_checkpoint(model, optimizer, scheduler, best, conf, global_step)
        keep_max_num_saved_models(conf, conf.max_to_keep)

    return best
