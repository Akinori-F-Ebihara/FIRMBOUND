from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
from lion_pytorch import Lion
from loguru import logger
from torch.optim import AdamW, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)

from utils.misc import extract_params_from_config


def get_optimizer(conf):
    if "adam" in conf.optimizer.lower():
        return AdamW
    elif "rmsprop" in conf.optimizer.lower():
        return RMSprop
    elif "lion" in conf.optimizer.lower():
        return Lion
    else:
        raise ValueError(f'Optimizer "{conf.optimizer}" is not implemented!')


def initialize_optimizer(
    model: torch.nn.Module, config: Dict[str, Any]
) -> Tuple[torch.nn.Module, optim.Optimizer]:
    """
    Initializes the optimizer for training the model, based on the configuration parameters.

    Args:
    - model: The model to be trained.
    - config: A dictionary containing model configuration parameters.

    Returns:
    - Tuple[Module, optim.Optimizer]: A tuple containing the initialized model and optimizer.
    """
    # check if necessary parameters are defined in the config file
    requirements = set(
        [
            "LEARNING_RATE",
            "LR_DECAY_STEPS",
            "OPTIMIZER",
            "WEIGHT_DECAY",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    base_optimizer = get_optimizer(conf)

    optimizer = base_optimizer(
        model.parameters(),
        weight_decay=conf.weight_decay,
        lr=conf.learning_rate,
    )

    logger.info(f"Optimizer:\n{optimizer}")

    return model, optimizer


def initialize_scheduler(
    optimizer: optim.Optimizer,
    config: Dict,
) -> Union[
    LambdaLR,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
]:
    # check if necessary parameters are defined in the config file
    requirements = set(["SCHEDULER", "SCHEDULER_OPTIONS"])
    conf = extract_params_from_config(requirements, config)

    if "constant" in conf.scheduler.lower():
        return LambdaLR(optimizer, lambda _: 1.0)
    elif "steplr" in conf.scheduler.lower() and "multi" not in conf.scheduler.lower():
        return StepLR(optimizer, **conf.scheduler_options[conf.scheduler])
    elif "multisteplr" in conf.scheduler.lower():
        return MultiStepLR(optimizer, **conf.scheduler_options[conf.scheduler])
    elif "exponential" in conf.scheduler.lower():
        return ExponentialLR(optimizer, **conf.scheduler_options[conf.scheduler])
    elif "cosineannealing" in conf.scheduler.lower() and "restart" not in conf.scheduler.lower():
        return CosineAnnealingLR(optimizer, **conf.scheduler_options[conf.scheduler])
    elif "cosineannealing" in conf.scheduler.lower() and "restart" in conf.scheduler.lower():
        return CosineAnnealingWarmRestarts(optimizer, **conf.scheduler_options[conf.scheduler])
    elif "plateau" in conf.scheduler.lower():
        return ReduceLROnPlateau(optimizer, **conf.scheduler_options[conf.scheduler])
    else:
        raise ValueError(f"Invalid scheduler name: {conf.scheduler}")


def step_scheduler(
    scheduler: Union[
        LambdaLR,
        StepLR,
        MultiStepLR,
        ExponentialLR,
        CosineAnnealingLR,
        ReduceLROnPlateau,
    ],
    best: Union[Tuple[float], Tuple[float, ...]],
) -> None:
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(np.mean(best))
    else:
        scheduler.step()
