import colorsys
import datetime
import os
import shutil
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.misc import (compile_comment, convert_torch_to_numpy,
                        extract_params_from_config,
                        format_common_folder_structure)
from utils.performance_metrics import (initialize_performance_metrics,
                                       summarize_performance)


def save_config_info(config, is_ddp=True):
    """
    Save config.py and config (dict) for reproducibility.
    """
    if is_ddp and dist.get_rank() != 0:
        return

    requirements = set(["CONFIG_PATH", "DIR_CONFIGS", "NOW", "EXP_PHASE"])
    conf = extract_params_from_config(requirements, config)

    comment = compile_comment(config)

    shutil.copyfile(conf.config_path, conf.dir_configs + f"/config_{conf.now}_orig.py")
    with open(conf.dir_configs + f"/config_{conf.now}_cooked.py", "w") as f:
        f.write(
            "# This is the actual config dict used for the training.\n"
            "# Simply import this file for reproducing the results.\n\n"
            "import numpy as np\n"
            "import torch \n\n"
        )
        f.write("config = {\n")
        for key, value in config.items():
            if isinstance(value, str):
                f.write(f'  "{key}": "{value}", \n')
            elif torch.is_tensor(value):
                f.write(f'  "{key}": torch.{repr(value)}, \n')
            elif isinstance(value, np.ndarray):
                f.write(f'  "{key}": np.{repr(value)}, \n')
            else:
                f.write(f'  "{key}": {value}, \n')
        f.write("}\n")
    with open(conf.dir_configs + f"/{comment}.txt", "w") as f:
        f.write("")


def create_directories_and_log(dict_path):
    """
    A simple helper function create directories and output log message.
    """
    if not os.path.exists(dict_path) and dist.get_rank() == 0:
        os.makedirs(dict_path, exist_ok=True)
        logger.info(colored("A new log directory was created: ", "yellow") + f"{dict_path}")


def setup_checkpoint_log_folder(config, conf):
    config["DIR_CKPTLOGS"] = f"{conf.root_ckptlogs}" + format_common_folder_structure(conf)
    config["DIR_CONFIGS"] = config["DIR_CKPTLOGS"] + "/configs"
    create_directories_and_log(config["DIR_CONFIGS"])


def setup_tensorboard_log_folder(config, conf):
    config["DIR_TBLOGS"] = f"{conf.root_tblogs}" + format_common_folder_structure(conf)
    create_directories_and_log(config["DIR_TBLOGS"])


def setup_stdout_log_folder(config, conf):
    # do not make individual folder because one trial contains one .log file anyways
    # + format_common_folder_structure(conf)[:-1]
    config["DIR_STDOUTLOGS"] = f"{conf.root_stdoutlogs}"
    config["STDOUTLOG_NAME"] = (
        f"{config['DIR_STDOUTLOGS']}/{format_common_folder_structure(conf)}.log"
    )
    create_directories_and_log(config["DIR_STDOUTLOGS"])


def get_timestamp_to_broadcast(config: Dict) -> None:
    """
    Get a single timestamp for a log file and broadcast across multiple workers.
    """
    rank = dist.get_rank()

    # Get a timestamp for each trial on rank 0 GPU
    if rank == 0:
        config["NOW"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        # Make sure that now_tensor is a tensor so that dist.broadcast can be used
        now_tensor = torch.tensor([int(config["NOW"])], dtype=torch.long, device=rank)
    else:
        # Allocate space for now_tensor on other ranks
        now_tensor = torch.tensor([0], dtype=torch.long, device=rank)

    # Broadcast the value to other GPUs
    dist.broadcast(now_tensor, src=0)

    # Convert back to string for non-zero ranks
    config["NOW"] = str(now_tensor.item())


def create_log_folders(config):
    """
    Create log folders for earch trial:
    - checkpoint log folder
    - tensorboard event log folder
    - stdout log folder
    Note that an optuna database log folder is created for each subproject, not each trial.
    For definition of optuna database log folder, see utils.hyperparameter_tuning:run_optuna.
    """

    # get a timestamp for each trial
    get_timestamp_to_broadcast(config)

    # check if necessary parameters are defined in the config file
    requirements = set(
        [
            "ROOT_CKPTLOGS",
            "ROOT_TBLOGS",
            "ROOT_STDOUTLOGS",
            "EXP_PHASE",
            "COMMENT",
            "CONFIG_PATH",
            "NOW",
            "MODEL_BACKBONE",
            "ORDER_SPRT",
            "LLLR_VERSION",
            "PARAM_LLR_LOSS",
            "PARAM_AUSAT_LOSS",
            "AUCLOSS_VERSION",
            "AUCLOSS_BURNIN",
            "PARAM_MULTIPLET_LOSS",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    setup_checkpoint_log_folder(config, conf)
    setup_tensorboard_log_folder(config, conf)
    setup_stdout_log_folder(config, conf)


def setup_logger(config):
    """ """

    def rank_filter(record):
        """
        Ensure logging only when rank==0.
        """
        return dist.get_rank() == 0

    def tqdm_compatible_logger(logger, logpath, rotation="10 MB", enqueue=True, level="DEBUG"):
        """
        loguru logger outputs to console and log file specified with logpath.
        lambda function is used to wrap tqdm.write(), which is required to
        output message onto the console without disrupting tqdm progress bar.

        Args:
        - logger: loguru logger object

        Return:
        - None
        """
        logger.remove()
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            enqueue=enqueue,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True,
            filter=rank_filter,
        )
        logger.add(logpath, rotation=rotation, enqueue=enqueue, level=level, filter=rank_filter)

    requirements = set(["STDOUTLOG_NAME"])
    conf = extract_params_from_config(requirements, config)

    tqdm_compatible_logger(logger, conf.stdoutlog_name)


def investigate_log(config, is_stop_at_error):
    """
    Investigate .log file and count WARNING and ERROR sign.
    Raise ValueError when non-zero ERROR are found.
    """
    requirements = set(["STDOUTLOG_NAME"])
    conf = extract_params_from_config(requirements, config)

    with open(conf.stdoutlog_name, "r") as f:
        contents = f.read()
    num_warning = contents.count("WARNING")
    num_error = contents.count("ERROR")
    logger.info(f"{num_warning=}/{num_error=} in the .log file.")

    if num_error > 0:
        if is_stop_at_error:
            raise ValueError("Found ERROR! Check the .log file for debug!")
        else:
            logger.warning("Found ERROR! check the .log file for debug!")
    elif num_warning > 0:
        logger.warning("Found WARNING! check the .log file for debug!")


class ContexualLogger:
    """
    A context manager that sets up a logger before executing a code block, and
    investigates the logger after the code block has completed.

    Usage:
    config = initialize_setups()
    with LoggingContextManager(config):
        training_my_neural_network(config)
    """

    def __init__(self, config, is_stop_at_error: bool = True):
        self.config = config
        self.is_stop_at_error = is_stop_at_error

    def __enter__(self):
        """
        Sets up a logger before entering the with block.

        Returns:
        self: The current LoggingContextManager instance.
        """
        setup_logger(self.config)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Investigates the logger after exiting the with block.

        Args:
        exc_type: The type of exception raised (if any).
        exc_value: The value of the exception raised (if any).
        traceback: The traceback object associated with the exception (if any).
        """
        investigate_log(self.config, self.is_stop_at_error)


def get_tb_writer(config):
    """
    Get a PyTorch-builtin TensorBoard writer.

    Args:
    - config (dict): a dictionary containing key "DIR_TBLOGS" and
      its value specifying the path to tensorboard log file.

    Return:
    - SummaryWriter(): PyTorch SummaryWriter object for logging.
    """
    # check if necessary parameters are defined in the config file
    requirements = set(["DIR_TBLOGS"])
    conf = extract_params_from_config(requirements, config)

    return SummaryWriter(log_dir=conf.dir_tblogs)


@logger.catch
def log_training_results(
    tb_writer,
    model,
    optimizer,
    local_step: int,
    global_step: int,
    iter_num: int,
    config: dict,
    performance_metrics: dict,
    phase: str,
    is_ddp=True,
):
    """
    Log training and evaluation results at:
    - the end of validation phase
    - evary config['TRAIN_DISPLAY_STEP'] training iteration
    """

    def subsample_dict(performance_metrics, selected_keys):
        return {k: v for k, v in performance_metrics.items() if k in selected_keys}

    # log val results only periodically
    if "val" in phase.lower() and local_step != iter_num - 1:
        return performance_metrics

    # check if necessary parameters are defined in the config file
    requirements = set(
        [
            "VERBOSE",
            "PARAM_MULTIPLET_LOSS",
            "PARAM_LLR_LOSS",
            "PARAM_AUSAT_LOSS",
            "AUCLOSS_VERSION",
            "WEIGHT_DECAY",
            "TRAIN_DISPLAY_STEP",
            "TIME_STEPS",
            "IS_REAL_WORLD",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    # Log train results only periodically
    if "train" in phase.lower() and global_step % conf.train_display_step != 0:
        # reset the performance metrics to save GPU memory
        return initialize_performance_metrics()

    _model = model.module if False else model  # depricated: conf.is_ultim

    # summarize performance_metrics
    performance_metrics = summarize_performance(performance_metrics, conf.time_steps)

    # Log only by the first GPU
    if is_ddp and dist.get_rank() == 0 or not is_ddp:
        # log the results
        output_to_terminal_and_log(conf, phase, global_step, _model, performance_metrics)

        # write to TensorBoard
        # select values of interest (I might like to modify here later)
        subset_metrics = subsample_dict(
            performance_metrics,
            [
                "losses",
                "mean_abs_error",
                "mean_macro_recall",
                # "ausat_from_confmx",
                "grad_norm",
            ],
        )
        if "train" in phase.lower():
            subset_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            # subset_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        tblog_writer(tb_writer, global_step, phase=phase, metrics=subset_metrics)

    # training phase: reset the performance metrics after the log
    # eval phase: needs postprocess, do not reset the performance metrics
    return initialize_performance_metrics() if "train" in phase.lower() else performance_metrics


def output_to_terminal_and_log(conf, phase, global_step, _model, performance_metrics):
    if conf.verbose:
        logger.info("")
        logger.info("Global Step ={:7d}".format(global_step))
        logger.info("{} mean(MacRec)_t: {}".format(phase, performance_metrics["mean_macro_recall"]))

        if not conf.is_real_world:
            logger.info("{} mean ABSerr: {}".format(phase, performance_metrics["mean_abs_error"]))
        # logger.info(
        #     "{} ausat from confmx: {}".format(
        #         phase, performance_metrics["ausat_from_confmx"]
        #     )
        # )
        logger.info(
            "{} MCE loss:{:7.5f} * {}".format(
                phase,
                performance_metrics["losses"]["multiplet_crossentropy (MCE)"],
                str(conf.param_multiplet_loss),
            )
        )
        logger.info(
            "{} LLLR :{:7.5f} * {}".format(
                phase,
                performance_metrics["losses"]["LLR_estimation (LLRE)"],
                str(conf.param_llr_loss),
            )
        )
        # logger.info(
        #     "{} AUSAT loss ver{}:{:7.5f} * {}".format(
        #         phase,
        #         conf.aucloss_version,
        #         performance_metrics["losses"]["AUSAT_optimization"],
        #         str(conf.param_ausat_loss),
        #     )
        # )
        logger.info(
            "{} weight decay:{:7.5f} * {}".format(
                phase,
                performance_metrics["losses"]["weight_decay"],
                str(conf.weight_decay),
            )
        )


def tblog_writer(tb_writer, global_step, phase, metrics):
    """
    Logs various values to TensorBoard.
    The keys in the kwargs dictionary will be used to create the names for the values in TensorBoard.

    Args:
    - tb_writer (torch.utils.tensorboard.SummaryWriter()): TensorBoard logger to use for logging the values.
    - global_step (int): Global step to use for logging the values.
    - phase (str): The phase of the model, used to prefix the names of the logged values.
    - kwargs (dict): Dictionary of values to log.
                     The keys in the dictionary will be used to create the names for the values in TensorBoard.
    Returns:
    None

    Example:
    - a keyword argument loss={'LLLR': 0.2} at a training phase will be logged as:
        'training_loss/LLLR' with value 0.2.
    """

    def write_to_tensorboard(metrics):
        for key, value in metrics.items():
            if isinstance(value, dict):
                write_to_tensorboard(value)
            else:
                name = f"{phase}/{key}"
                tb_writer.add_scalar(name, value, global_step)

    assert isinstance(metrics, dict), "metrics is not a dictionary!"
    # use a recursive function to log values
    write_to_tensorboard(metrics)


def add_prefix_to_keys(original_dict: dict, prefix: str) -> dict:
    """
    Add a prefix to all the keys in a dictionary.
    Parameters:
        original_dict (dict): The original dictionary whose keys will be modified.
        prefix (str): The prefix to add to all the keys in the original dictionary.
    Returns:
        dict: A new dictionary with the prefix added to all the keys.
    """
    assert isinstance(original_dict, dict), "original_dict should be of type dict"
    assert isinstance(prefix, str), "prefix should be of type str"
    return {prefix + key: value for key, value in original_dict.items()}


def plot_likelihood_ratio_matrix(
    llrm: np.ndarray, gt_labels: np.ndarray, num_trajectories: int = 50
) -> None:
    """
    Plot the likelihood ratio matrix for each pair of classes in llrm.

    Args:
    - llrm (ndarray): an array of shape (batch_size, num_classes, num_classes) containing the log-likelihood ratio
      matrix for each sample in the dataset.
    - gt_labels (ndarray): an array of shape (batch_size) containing the ground-truth labels.
    - num_trajectories (int, optional): the number of trajectories to plot for each pair of classes. Defaults to 50.
    """
    num_classes = llrm.shape[-1]
    colors = plt.cm.tab10(range(num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            plt.subplot(num_classes, num_classes, (j + 1) + num_classes * i)
            for k, c in zip(range(num_classes), colors):
                plt.plot(
                    np.transpose(llrm[gt_labels == k, :, i, j][:num_trajectories]),
                    color=c,
                )

    plt.subplot(num_classes, num_classes, 1)
    for (
        k,
        c,
    ) in zip(range(num_classes), colors):
        plt.plot(np.transpose(llrm[gt_labels == k, :, 0, 0][:1]), color=c, label=f"class {k}")
    plt.legend()


def plot_llrs_with_thresholds(
    llrs: np.ndarray,
    gt_labels: np.ndarray,
    thresholds: np.ndarray,
    max_traj_per_cls_pair: int = 1,
    max_traj_per_llr: int = 1,
    threshold_level: str = "max",
    plot_labels: bool = False,
):
    """
    Args:
    - llrs (np.array): array of size (batch_size, time_steps, num_classes, num_classes)
    - gt_labels (np.array): array of size (batch_size)
    - thresholds (np.array): array of size (num_thresh, batch_size, time_steps, num_classes, num_classes)
    - max_traj_per_cls_pair (int): positive integer specifies the maximum number of trajectories per class
    - max_traj_per_llr (int): positive integer specifies the maximum number of threshold trajectories per LLR
    - threshold_level (str):
        'zero', 'smallest' or 'min': use the minimum threshold. max_traj_per_cls_pair is automatically set to 1.
        'small' use the random thresholds from the first third of the data, exept the minumum threshold.
        'medium' use the random thresholds from the second third of the data.
        'large' use the random thresholds from the last third of the data, exept the maximum threshold.
        'largest' or 'max' : use the maximum threshold. max_traj_per_cls_pair is automatically set to 1.
    """
    assert threshold_level in [
        "zero",
        "smallest",
        "min",
        "small",
        "medium",
        "large",
        "largest",
        "max",
    ]
    assert max_traj_per_cls_pair > 0
    assert thresholds.shape[1:] == llrs.shape

    num_classes = len(np.unique(gt_labels))
    num_thresh = thresholds.shape[0]

    # select thresholds of size=max_traj_per_llr
    if any(s in threshold_level for s in ["zero", "smallest", "min"]):
        dice_thresh = np.array([0])
        max_traj_per_llr = 1
    elif any(s in threshold_level for s in ["largest", "max"]):
        dice_thresh = np.array([num_thresh - 1])
        max_traj_per_llr = 1
    else:
        if "small" in threshold_level:
            start_idx = 1
            end_idx = int(num_thresh / 3)
        elif "medium" in threshold_level:
            start_idx = int(num_thresh / 3)
            end_idx = int(num_thresh * 2 / 3)
        elif "large" in threshold_level:
            start_idx = int(num_thresh * 2 / 3)
            end_idx = num_thresh - 1
        dice_thresh = np.random.choice(
            range(start_idx, end_idx), size=max_traj_per_llr, replace=False
        )

    # colors for plot
    colors = matplotlib.cm.tab10(range(num_classes))
    ex_colors = expand_color_variations(colors, max_traj_per_llr)

    # find minimum number of examples across the classes
    cls_samples = []
    for k in range(num_classes):
        cls_samples.append(len(llrs[gt_labels == k]))

    # randomly selected example data indices of size=max_traj_per_cls_pair
    dice_batch = np.random.permutation(np.min(cls_samples) - 1)[:max_traj_per_cls_pair]

    for j in range(num_classes):
        for k in range(num_classes):
            if k == j:
                continue
            for i in range(max_traj_per_llr):
                i_th = dice_thresh[i]
                label = f"class {k} vs. {j} at y={k} with thresh #{i_th}"
                color = ex_colors[k, i]
                targllr = np.transpose(llrs[gt_labels == k, :, k, j][dice_batch])
                targth = np.transpose(thresholds[i_th, gt_labels == k, :, k, j][dice_batch])
                plt.plot(targllr, color=color, label=label)
                plt.plot(targth, color=color)
    if plot_labels:
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_label_set = dict(zip(labels, handles))
        plt.legend(unique_label_set.values(), unique_label_set.keys(), fontsize=10)


def rgb_to_hsv(color):
    return colorsys.rgb_to_hsv(color[0], color[1], color[2])


def hsv_to_rgb(color):
    return colorsys.hsv_to_rgb(color[0], color[1], color[2])


def create_color_variations(color, num_variations=5):
    hsv_color = rgb_to_hsv(color)
    variations = []
    for i in range(1, num_variations + 1):
        new_saturation = hsv_color[1] * (1 - i * 0.1)
        new_color = (hsv_color[0], new_saturation, hsv_color[2])
        variations.append(hsv_to_rgb(new_color))
    return variations


def expand_color_variations(base_colors, num_variations=5):
    num_colors = base_colors.shape[0]
    # Create the color variations
    color_matrix = np.empty((num_colors, num_variations, 4))
    for i, color in enumerate(base_colors[:, :3]):  # only take the RBG values
        variations = create_color_variations(color, num_variations)
        for j, variation in enumerate(variations):
            color_matrix[i, j, :3] = variation
            color_matrix[i, j, 3] = 1.0  # Set alpha channel to 1
    return color_matrix


def plot_example_trajectories(
    llrm, gt_labels, dice=[None, None], max_traj_per_cls_pair=10, max_example_cls=10
):
    """
    Summarize a log-likelihood ratio matrix (llrm) into one panel.

    Args:
    - llrm (float): array of size (batch_size, time_steps, num_classes, num_classes)
      log likelihood ratio matrix, either ground-truth or estimated.
    - gt_labels (int): array of size (batch_size,). gt_labels \in [0, 1, ... num_classes - 1]
      ground truth data labels.
    - dice (int or None): array of size (max_traj_per_cls_pair,) if provided.
      For selecting/reproducing trajectories to be plotted.
      - dice[0]: indices for trajectories.
      - dice[1]: indices for classes.
    - max_traj_per_cls_pair (int): positive integer specifies the maximum number of trajectories
      per class pair. e.g., if num_classes = 3, there will be 3*2=6 class pairs and thus
      6 * max_traj_per_cls_pair trajectories will be plotted in total. Note that a number of
      trajectories will be smaller than max_traj_per_cls_pair if llrm smaller number of examples.
    - max_example_cls (int): positive integer specifies the maximum number of classes to be plotted.
      classes are randomly selected if max_example_cls > num_classes.

    Returns:
    - dice (int): the dice that used to select example trajectories. Use it to plot to select
      the corresponding trajectories of different llrms (e.g., ground-truth llrm and estimated llrm).

    Example:
    plt.figure(figsize=(16.2, 10))

    plt.subplot(1,2,1)
    plt.title('Ground-truth LLR trajectories')
    dice = plot_example_trajectories(gt_llrm, y)

    plt.subplot(1,2,2)
    plt.title('Estimated LLR trajectories')
    _ = plot_example_trajectories(estimated_llrm, y, dice=dice)

    """

    # limit example
    unique_classes = np.unique(gt_labels)
    num_classes = len(unique_classes)

    assert max_example_cls > 1, (
        "max_example_cls must be greater than 1 to ensure likelihood ratio calculation"
        " (i.e., ratio is defined with cls > 1.)"
    )

    colors = matplotlib.cm.tab10(range(num_classes))
    if all(d is None for d in dice):
        cls_samples = []
        for k in unique_classes:
            cls_samples.append(len(llrm[gt_labels == k]))
        dice[0] = np.random.permutation(np.min(cls_samples))[:max_traj_per_cls_pair]

        # limit example classes if needed
        if num_classes > max_example_cls:
            dice[1] = np.random.permutation(num_classes)[:max_example_cls]
        else:
            dice[1] = unique_classes
    else:
        assert (
            len(dice[0]) == max_traj_per_cls_pair
        ), "length of dice[0] must be equal to max_traj_per_cls_pair!"

    for j in dice[1]:
        for k, c in zip(dice[1], colors):
            if k == j:
                continue
            plt.plot(
                np.transpose(llrm[gt_labels == k, :, k, j][dice[0]]),
                color=c,
                label=f"class {k} vs. {j} at y={k}",
            )
    # labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_label_set = dict(zip(labels, handles))
    plt.legend(unique_label_set.values(), unique_label_set.keys(), fontsize=10)

    return dice


def save_sdre_imgs(
    config, target, tb_writer, global_step, last_example_to_plot, performance_metrics
):
    """
    Save the results of sequential density ratio estimation (SDRE).
    """
    # check if necessary parameters are defined in the config file
    requirements = set(
        [
            "IS_SAVE_FIGURE",
            "IS_MULT_SAT",
            "OPTIMIZATION_TARGET",
            "AUCLOSS_VERSION",
            "PARAM_MULTIPLET_LOSS",
            "PARAM_LLR_LOSS",
            "PARAM_AUSAT_LOSS",
            "NUM_CLASSES",
            "BATCH_SIZE",
            "IS_REAL_WORLD",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    if not conf.is_save_figure or dist.get_rank() != 0:
        return

    y_batch, gt_llrs, llrs = last_example_to_plot

    performance_metrics = convert_torch_to_numpy(performance_metrics)
    aesthetic_num_traj = np.max([1, 20 // conf.num_classes])

    plt.rcParams["font.size"] = 15
    fig, ax = plt.subplots(figsize=(32, 10))

    plt.subplot(1, 2, 1)
    plt.title("Estimated LLR trajectories")
    dice = plot_example_trajectories(
        llrs, y_batch, dice=[None, None], max_traj_per_cls_pair=aesthetic_num_traj
    )
    plt.subplot(1, 2, 2)
    plt.title("Ground-truth LLR trajectories")
    if conf.is_real_world:
        plt.text(0.35, 0.5, "Real-world data. GT-LLRs not available")
    else:
        _ = plot_example_trajectories(
            gt_llrs, y_batch, dice, max_traj_per_cls_pair=aesthetic_num_traj
        )

    targval = str(np.array([t for t in target]))
    name = f"FlagMult{conf.is_mult_sat}_mla{conf.param_multiplet_loss}"
    f"{conf.param_llr_loss}{conf.param_ausat_loss}_target{conf.optimization_target}{targval}"

    # for paper figures
    # plt.savefig(f"Iter{global_step}_SAT_targval{targval}.pdf", format="pdf")

    # save in the TensorBoard log
    tb_writer.add_figure(name, fig, global_step=global_step)

    plt.close("all")
    logger.info("Figures saved.")

