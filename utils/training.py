import datetime
import os
import re
import socket
import time
import warnings
from typing import Any, Dict, Tuple, Union

import numpy as np
import optuna
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from optuna.integration import TorchDistributedTrial
from optuna.trial import Trial
from termcolor import colored
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LambdaLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.losses import compute_loss_and_metrics
from models.optimizers import initialize_optimizer, initialize_scheduler, step_scheduler
from models.temporal_integrators import import_model, load_pretrained_states
from utils.checkpoint import initialize_objectives, update_and_save_result
from utils.data_processing import lmdb_dataloaders, move_data_to_device
from utils.hyperparameter_tuning import report_to_pruner
from utils.logging import get_tb_writer, log_training_results, save_config_info, save_sdre_imgs
from utils.misc import (
    catch_all_class_methods,
    convert_torch_to_numpy,
    extract_params_from_config,
    fix_random_seed,
    sanity_check,
    set_cpu_workers,
)
from utils.performance_metrics import accumulate_performance, initialize_performance_metrics


def initialize_training(_config):
    best = initialize_objectives(_config)
    global_step = 0
    return best, global_step


def finalize_training(
    _trial,
    training_tools,
    _config,
    best,
    global_step,
    epoch,
):
    logger.info(f"Final validation after {epoch + 1} epochs.")

    model, optimizer, scheduler, data_loaders, tb_writer = training_tools

    best, global_step = run_one_epoch(
        _trial,
        model,
        optimizer,
        scheduler,
        data_loaders,
        tb_writer,
        _config,
        best,
        global_step,
        epoch,
        phase="val",
    )
    return best, global_step


def handle_runtime_error(_config, e):
    if _config["IS_SKIP_OOM"]:
        raise optuna.exceptions.TrialPruned(f"Trial pruned due to RuntimeError:\n{e}")
    else:
        raise optuna.exceptions.OptunaError(f"Exception due to RuntimeError:\n{e}")


def manage_single_epoch(
    _trial,
    training_tools,
    _config,
    best,
    global_step,
    epoch,
):
    model, optimizer, scheduler, data_loaders, tb_writer = training_tools
    try:
        best, global_step = run_one_epoch(
            _trial,
            model,
            optimizer,
            scheduler,
            data_loaders,
            tb_writer,
            _config,
            best,
            global_step,
            epoch,
        )
    except RuntimeError as e:
        handle_runtime_error(_config, e)

    step_scheduler(scheduler, best)

    return best, global_step


@catch_all_class_methods
class PyTorchPhaseManager:
    """
    A context manager for managing the training and evaluation phases of a PyTorch model.

    Args:
    - phase (str): The phase to manage ('train' or 'val').

    Methods:
    - __init__(self, phase)
    - __enter__(self)
    - __exit__(self, *args)
    - training_preprocess(self)
    - training_postprocess(self)
    - evaluation_preprocess(self)
    - evaluation_postprocess(self)

    Required variables:
    - model
    - optimizer
    - loss

    Remark:
    - assumption for the loss function is that it takes arguments in order of:
      (model, ..., config).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim,
        phase: str,
    ):
        """
        Initializes a PyTorchPhaseManager object with the specified phase.

        Args:
        - phase (str): The phase to manage ('train' or 'val').
        """

        self.model = model
        self.optimizer = optimizer
        self.phase = phase
        self.loss = None
        self.scaler = torch.cuda.amp.GradScaler()

    def __enter__(self):
        """
        Enters the context for the current phase and performs any necessary preprocessing.

        Returns:
        - self: The PyTorchPhaseManager object.
        """

        if "train" in self.phase:
            self.mode = torch.enable_grad()
            self.training_preprocess()
        elif "val" in self.phase:
            self.mode = torch.no_grad()
            self.evaluation_preprocess()
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

        self.mode.__enter__()

        return self

    def __exit__(self, *args):
        """
        Exits the context for the current phase and performs any necessary postprocessing.

        Args:
        - *args: The arguments passed to __exit__.
        """
        if "train" in self.phase:
            self.training_postprocess()
            self.grad_norm = self.monitor_gradient()
        elif "val" in self.phase:
            self.evaluation_postprocess()
            self.grad_norm = torch.tensor(0.0)
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

        self.mode.__exit__(*args)

    def training_preprocess(self):
        """
        Sets the model to training mode and clears the gradients of the optimizer.
        """
        self.model.train()
        self.optimizer.zero_grad()

    def training_postprocess(self):
        """
        Computes the gradients of the loss with respect to the model parameters
        using loss.backward() and performs the optimization step using optimizer.step().
        """

        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def evaluation_preprocess(self):
        """
        Sets the model to evaluation mode.
        """
        self.model.eval()

    def evaluation_postprocess(self):
        """
        Performs no post-processing steps, as gradients are not computed during evaluation.
        """
        pass

    def monitor_gradient(self):
        """ """

        grad_raw = [
            torch.norm(p.grad.detach()) for p in self.model.parameters() if p.grad is not None
        ]

        if not grad_raw:
            grad_norm = torch.tensor(0.0)
        else:
            grad_norm = torch.norm(torch.stack(grad_raw), 2)

        return grad_norm


@logger.catch
def prepare_for_training(
    config: Dict[str, Any], is_load_test: bool = False, is_ddp: bool = True
) -> Tuple[
    nn.Module,
    optim.Optimizer,
    Union[
        LambdaLR,
        StepLR,
        MultiStepLR,
        ExponentialLR,
        CosineAnnealingLR,
        ReduceLROnPlateau,
    ],
    Dict[str, DataLoader],
    SummaryWriter,
]:
    """
    Prepare the network and data for training.

    Args:
    - trial: An object used by Optuna to generate trial parameters. Unused if not using Optuna.
    - config: A dictionary containing various configuration settings.
    - device: The device to run the computation on, e.g. 'cpu' or 'cuda'.

    Returns:
    - model: The initialized network.
    - optimizer: The optimizer used to update the network parameters during training.
    - data_loaders: A dictionary containing the train, validation, and test data loaders.
    - tb_writer: A SummaryWriter object for logging training progress to TensorBoard.
    """
    # Accurate matrix multiplication when working with 32-bit floating-point values
    torch.set_float32_matmul_precision("high")

    # Optimize cuda computation if network structure is static, sacrificing reproducibility
    torch.backends.cudnn.benchmark = True

    # Set number of CPU threads (workers)
    set_cpu_workers(config)

    # Set random seeds (optional)
    fix_random_seed(config)

    # Save config.py and config (dict) for reproducibility
    save_config_info(config, is_ddp=is_ddp)

    # Setup Tensorboard writer
    tb_writer = get_tb_writer(config)

    # Initialize the network or load a pretrained one
    # tb_writer is optional: provide if you want to have a model graph on TB
    model = import_model(config, tb_writer=None, is_ddp=is_ddp)

    # Setup the optimizer
    model, optimizer = initialize_optimizer(model, config)

    # Setup the learning rate scheduler
    scheduler = initialize_scheduler(optimizer, config)

    # (Optional) Load pretrained state_dicts
    load_pretrained_states(model, optimizer, scheduler, config)

    # Load train, val, and test data
    data_loaders = lmdb_dataloaders(config, load_test=is_load_test, is_ddp=is_ddp)

    return model, optimizer, scheduler, data_loaders, tb_writer


# Do not use @logger.catch here to allow pruning
def iterating_over_dataset(
    trial: Trial,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    data_loaders: Dict[str, DataLoader],
    tb_writer: SummaryWriter,
    config: Dict[str, Any],
    best: Tuple[float, ...],
    global_step: int,
    phase: str,
) -> Tuple[Dict[str, float], int, Tuple[float], Tuple[float, ...]]:
    """
    Runs the model for a given phase (training, validation, or test) and logs results.

    Args:
    - trial: An object used by Optuna to generate trial parameters. Unused if not using Optuna.
    - model: The PyTorch model to be run.
    - optimizer: The PyTorch optimizer used to update model parameters during training.
    - data_loaders: A dictionary containing train, validation, and test data loaders.
    - tb_writer: A TensorBoard SummaryWriter to log results.
    - config: A dictionary containing model configuration parameters.
    - best: A tuple of the current best performance metrics.
    - global_step: The current iteration number.
    - phase: The phase for which the model is being run. Can be 'train', 'validation', or 'test'.

    Returns:
    - performance_metrics: A dictionary containing the performance metrics.
    - global_step: The updated iteration number.
    - last_example_to_plot: An object for plotting results.
    - best: A tuple of the updated best performance metrics.
    """

    is_train, iter_num, performance_metrics, barcolor = training_setup(phase, config)

    for local_step, data in tqdm(
        enumerate(data_loaders[phase]),
        mininterval=2,
        total=iter_num,
        desc=colored(f"{phase} in progress...", barcolor),
        colour=barcolor,
        leave=False,
        disable=dist.get_rank() != 0,
    ):
        if config["IS_REAL_WORLD"]:
            x_batch, y_batch = move_data_to_device(data)

            # TODO: need to wrap and manage device
            batch_size, time_steps, _ = x_batch.shape
            gt_llrs_batch = torch.zeros(
                batch_size,
                time_steps,
                config["NUM_CLASSES"],
                config["NUM_CLASSES"],
                device="cuda",
            )
        else:
            x_batch, y_batch, gt_llrs_batch = move_data_to_device(data)

        # a periodic validation is inserted between training iterations.
        # This function is recursively called under is_train=True, with phase='eval'.
        if is_train and global_step % config["VALIDATION_STEP"] == 0:
            best, global_step = run_one_epoch(
                trial,
                model,
                optimizer,
                scheduler,
                data_loaders,
                tb_writer,
                config,
                best,
                global_step=global_step,
                phase="val",
            )

        # Train phase: run preprocesses (model.train(), optimizer.zero_grad()),
        #              and postprocesses (optimizer.step(), loss.backward())
        # Eval phase: run preprocesses (model.eval()), enter torch.no_grad() mode,
        #             no postprocess
        with PyTorchPhaseManager(model, optimizer, phase=phase) as p:
            monitored_values = compute_loss_and_metrics(p.model, x_batch, y_batch, config)
            sanity_check(monitored_values, config["IS_SKIP_NAN"])
            # monitored_values = ausat_minibatch_learning(
            #     y_batch,
            #     monitored_values,
            #     global_step,
            #     config,
            # )
            p.loss = monitored_values["losses"]["total_loss"]

        # Store performance metrics
        performance_metrics = accumulate_performance(
            performance_metrics,
            y_batch,
            gt_llrs_batch,
            monitored_values,
            phase_manager=p,
            is_save_gpu_memory=config["IS_SAVE_GPU_MEMORY"],
        )

        # log results periodically. Return immediately when the condition is not met
        # performance_metrics is reset at training loop to avoid running out of memory
        performance_metrics = log_training_results(
            tb_writer,
            model,
            optimizer,
            local_step,
            global_step,
            iter_num,
            config,
            performance_metrics,
            phase,
        )

        # for figures
        if local_step == iter_num - 1 and "val" in phase:
            last_example_to_plot = convert_torch_to_numpy(
                [
                    y_batch,
                    gt_llrs_batch,
                    monitored_values["llrs"],
                ]
            )
        else:
            last_example_to_plot = None

        # increment global step that traces training process. skip if eval phase
        global_step = global_step + 1 if "train" in phase else global_step

    return performance_metrics, global_step, last_example_to_plot, best


@logger.catch
def eval_postprocess(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    best: Tuple[float, ...],
    performance_metrics: Dict[str, float],
    last_example_to_plot: Tuple[float],
    config: Dict[str, Any],
    tb_writer: SummaryWriter,
    global_step: int,
) -> Tuple[float, ...]:
    """
    Updates the status and saves a checkpoint.

    Args:
    - model: The PyTorch model.
    - optimizer: The PyTorch optimizer.
    - best: A tuple of the current best performance metrics.
    - performance_metrics: A dictionary containing the performance metrics.
    - last_example_to_plot: An object for plotting results.
    - config: A dictionary containing model configuration parameters.
    - global_step: The current iteration number.

    Returns:
    - best: A tuple of the updated best performance metrics.
    """
    # torch.distributed: wait for validation end on every MPI
    dist.barrier()

    # update status and save a checkpoint
    best = update_and_save_result(
        model,
        optimizer,
        scheduler,
        config,
        best,
        performance_metrics,
        global_step=global_step,
    )

    # save trajectory figures if needed
    save_sdre_imgs(config, best, tb_writer, global_step, last_example_to_plot, performance_metrics)

    return best


def run_one_epoch(
    trial: TorchDistributedTrial,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    data_loaders: Dict[str, DataLoader],
    tb_writer: SummaryWriter,
    config: Dict[str, Any],
    best: Tuple[float, ...],
    global_step: int,
    epoch: int = 0,
    phase: str = "train",
) -> Tuple[Tuple[float, ...], int]:
    """
    Runs one epoch of the model.

    Args:
    - trial: An optuna Trial object.
    - model: The PyTorch model.
    - optimizer: The PyTorch optimizer.
    - data_loaders: A dictionary containing the data loaders for
                    the train, validation, and test sets.
    - tb_writer: A tensorboard SummaryWriter to log results.
    - config: A dictionary containing model configuration parameters.
    - best: A tuple of the current best performance metrics.
    - global_step: The current iteration number.
    - phase: The phase for which the model is being run. Can be 'train', 'validation', or 'test'.

    Returns:
    - best: A tuple of the updated best performance metrics.
    - global_step: The current iteration number.
    """
    # For Distributed Data Parallel (DDP). Ensure random sampling at each epoch
    if phase == "train":
        data_loaders["train"].sampler.set_epoch(epoch)

    (
        performance_metrics,
        global_step,
        last_example_to_plot,
        best,
    ) = iterating_over_dataset(
        trial,
        model,
        optimizer,
        scheduler,
        data_loaders,
        tb_writer,
        config,
        best,
        global_step,
        phase,
    )

    # postprocesses at an eval phase
    if "val" in phase:
        best = eval_postprocess(
            model,
            optimizer,
            scheduler,
            best,
            performance_metrics,
            last_example_to_plot,
            config,
            tb_writer,
            global_step,
        )

        # optuna pruner for early stopping
        report_to_pruner(trial, best, global_step, config)

    return best, global_step


def find_free_port() -> int:
    with socket.socket() as s:
        # Bind to a free port provided by the host.
        s.bind(("", 0))
        port = s.getsockname()[1]  # Return the port number assigned.
        logger.info(f"Find a free port {port} to be used for communication across GPUs.")
        return port


def ddp_setup(rank: int, world_size: int, port: int, config: Dict) -> None:
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """

    # for AIGIS
    # os.environ["RANK"] = os.getenv("OMPI_COMM_WORLD_RANK")
    # os.environ["LOCAL_RANK"] = os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
    # os.environ["WORLD_SIZE"] = os.getenv("OMPI_COMM_WORLD_SIZE")
    # os.environ["MASTER_ADDR"] = os.getenv("CGM_RANK0_SERVER_IP")
    # os.environ["MASTER_PORT"] = os.getenv("CGM_TORCH_DISTRIBUTED_MASTER_PORT")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)  # "12355"
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(0, seconds=30),
    )
    torch.cuda.set_device(rank)

    learning_rate_linear_scaling(config, world_size)


def learning_rate_linear_scaling(config: Dict, world_size: int) -> None:
    """
    For distributed processing. Scale the learning rate by the factor of world_size.
    If learning rate's search space is defined for optuna, the space will be scaled as well.

    Args:
    - config: dictionary variable contains training setups including hyperparameters.
    - world_size: note that dist.get_world_size() cannot use here because
      this function is called before processes are spawned

    Returns:
    - None: config dictionary will be overwritten in-place.
    """

    config["LEARNING_RATE"] *= float(world_size)

    if "SPACE_LEARNING_RATE" in config:
        config["SPACE_LEARNING_RATE"]["LOW"] *= float(world_size)
        config["SPACE_LEARNING_RATE"]["HIGH"] *= float(world_size)

    return


def sanity_check_gpu_config(conf_gpu: np.string_):
    assert isinstance(conf_gpu, str), "config['GPU'] must be a string"
    pattern = re.compile(r"^\d+(, \d+)*$")
    assert pattern.match(conf_gpu), "config['GPU'] must contain only integers separated by commas"


def set_world_size(config: Dict) -> Tuple[int, str]:
    """ """

    # check if necessary parameters are defined in the config file
    requirements = set(["GPU", "WORLD_SIZE"])
    conf = extract_params_from_config(requirements, config)

    # sanity check
    sanity_check_gpu_config(conf.gpu)

    # restrict visible device(s)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf.gpu)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using " + colored(f"{device} ", "yellow") + "device.")

    world_size = conf.world_size

    return world_size


@logger.catch
def training_setup(
    phase: str,
    config: Dict,
) -> Tuple[bool, int, Dict, str, Dict]:
    """
    Returns:
    - is_train (bool): train flag to decide if trainable weights are updated.
    - iter_num (int): an index of the last full-size batch.
    - performance_metrics (dict): dictionary of model performance metrics
                                  initialized with zeros or empty list.
    - barcolor (str): specifies tqdm bar color for each phase.
    """
    # check if necessary parameters are defined in the config file
    requirements = set(["NUM_TRAIN", "NUM_VAL", "NUM_TEST", "BATCH_SIZE"])
    conf = extract_params_from_config(requirements, config)

    world_size = dist.get_world_size()
    is_train = True if "train" in phase.lower() else False
    barcolor = "cyan" if "train" in phase.lower() else "yellow"

    if "train" in phase.lower():
        iter_num = np.ceil(conf.num_train / conf.batch_size / world_size).astype("int")
    elif "val" in phase.lower():
        iter_num = np.ceil(conf.num_val / conf.batch_size / world_size).astype("int")
    elif "test" in phase.lower():
        iter_num = np.ceil(conf.num_test / conf.batch_size / world_size).astype("int")
    else:
        raise ValueError("Unknown phase!")
    performance_metrics = initialize_performance_metrics()

    return is_train, iter_num, performance_metrics, barcolor


def suppress_unnecessary_warnings():
    """
    Suppress unnecessary warnings that is confirmed to be benign.
    """
    # suppress warnings at gradient clipping
    warnings.filterwarnings(
        "ignore",
        message="The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )
    # Suppress FutureWarning from optuna.integration (for distributed coding)
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="optuna.integration.pytorch_distributed",
    )

    # Suppress UserWarning from torchinfo
    warnings.filterwarnings("ignore", category=UserWarning, module="torchinfo.torchinfo")

    # Suppress UserWarning from torch.storage
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.storage")


def finalize_ddp() -> None:
    rank = dist.get_rank()
    # torch.distributed: destroy process group and open ports
    if rank == 0:
        logger.info("Destroying all the processes. Wait 10 secs...")
        time.sleep(10)
    dist.destroy_process_group()
