from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from utils.data_processing import sequential_slice_labels
from utils.misc import extract_params_from_config
from utils.performance_metrics import calc_llrs, calc_oblivious_llrs

"""
DRE loss functions Adapted from the following papers:

- A. F. Ebihara et al. Sequential density ratio estimation for simultaneous
optimization of speed and accuracy. In ICLR, 2021.
- T. Miyagawa and A. F. Ebihara. The power of log-sum-exp:
 Sequential density ratio matrix estimation for speed-accuracy optimization. In ICML, 2021.
- A. F. Ebihara et al. Toward asymptotic optimality: Sequential unsupervised
regression of density ratio for early classification. In ICASSP, 2023.

  GitHub: https://github.com/Akinori-F-Ebihara/SPRT-TANDEM-PyTorch
"""


def compute_llrs(logits_concat: Tensor, oblivious: bool = False) -> Tensor:
    """
    Args:
        logits_concat: A logit Tensor with shape
            (batch_size, (time_steps - order_sprt), order_sprt + 1, num classes).
            This is the output from
            utils.data_processing.sequential_concat(logit_slice, labels_slice)
        oblivious: A bool, whether to use the oblivion formula or not (= TANDEM formula).

    Returns:
        llrs: A Tensor with shape (batch size, time_steps, num cls, num cls).
    """

    if oblivious:
        # Miyagawa and Ebihara, ICML 2021
        llrs = calc_oblivious_llrs(logits_concat)
    else:
        # compute with TANDEM formula: Ebihara+, ICLR 2021
        llrs = calc_llrs(logits_concat)

    return llrs  # (batch_size, time_steps, num cls, num cls)


def compute_loss_for_llr(llrs: Tensor, labels_concat: Tensor, config: dict) -> Tensor:
    """
    Compute the loss for log-likelihood ratio estimation (LLLR).
    Args:
        llrs: A Tensor with shape (batch size, time_steps, num cls, num cls).
        labels_concat: A Tensor with shape (batch size,).
        version: A string, which version of the loss to compute.

    Returns:
        loss: A scalar Tensor.
    """
    shapes = llrs.shape
    assert shapes[-1] == shapes[-2]
    num_classes = shapes[-1]

    # check if necessary parameters are defined in the config file
    requirements = set(
        [
            "LLLR_VERSION",
            "IS_CLASS_BALANCE",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    labels_oh = F.one_hot(labels_concat, num_classes).to(torch.float32)
    # (batch_size, num_classes)
    labels_oh_reshaped = labels_oh.reshape(-1, 1, num_classes, 1)
    # (batch_size, 1, num_classes, 1)

    if conf.lllr_version == "LLLR":  # LLLR, Ebihara+, ICLR 2021
        lllr = torch.abs(labels_oh_reshaped - torch.sigmoid(llrs))
        # (batch_size, time_steps, num_classes, num_classes)
        lllr = 0.5 * (num_classes / (num_classes - 1.0)) * torch.mean(lllr)

    elif conf.lllr_version == "LSEL":  # LSEL, Miyagawa and Ebihara, ICML 2021
        llrs = llrs * labels_oh_reshaped
        # (batch_size, time_steps, num_classes, num_classes)
        llrs = torch.sum(llrs, dim=2)
        # (batch_size, time_steps, num_classes)
        llrs = llrs.permute(1, 0, 2)
        # (time_steps, batch_size, num_classes)
        llrs = llrs.reshape(-1, num_classes)
        # (time_steps * batch_size, num_classes)
        minllr = torch.min(llrs, dim=1, keepdim=True)[0].detach()
        # (time_steps * batch_size, 1)
        llrs = llrs - minllr
        # avoids over-/under-flow
        # (time_steps * batch_size, num_classes)
        expllrs = torch.sum(torch.exp(-llrs), dim=1)
        # (time_steps * batch_size, )
        lllr = -minllr.squeeze() + torch.log(expllrs + 1e-12)
        # (time_steps * batch)
        lllr = torch.mean(lllr)
        # scalar

    else:
        raise ValueError(f"Unknown version {conf.lllr_version} found for LLR estimation loss.")

    return lllr


def restore_lost_significance(llrs):
    # Restore lost significance with epsilon
    tri = torch.ones_like(llrs)
    triu = torch.triu(tri)  # Upper triangular part.
    tril = torch.tril(tri)  # Lower triangular part.
    llrs_restore = llrs - 1e-10 * (triu - tril)
    # (batch_size, time_steps, num cls, num cls)
    # To avoid double hit due to the values exactly equal to 0
    # in scores or when doing truncation, LLRs of the last frame.

    return llrs_restore


def calc_weight_decay(model):
    """ """
    wd_reg = 0.0
    for name, param in model.named_parameters():
        if "bias" not in name:  # Exclude bias terms from weight decay
            wd_reg += torch.norm(param, p=2) ** 2
    return wd_reg


def retrieve_posteriors_from_llrm(llrm: Tensor, prior_ratio_matrix=None):
    """
    Args:
    - llrm: pytorch Tensor of size (batch_size, time_steps, num_classes, num_classes).
            Strictly antisymmetric matrix, diagnonal must be all-zero.
            lambda_kl = log(p(x | label_k) / p(x | label_l))
            note: (batch_size, num_classes, num_classes) will also work.
    - prior_ratio_matrix: pytorch Tensor of size (num_classes, num_classes)
                          Strictly antisymmetric matrix, diagnonal must be all-one.
                          chi_kl = p(label_k) / p(label_l)
    Returns:
    - posterior: A tensor of size (batch_size, time_steps, num_classes)

    """
    if prior_ratio_matrix is None:
        prior_ratio_matrix = torch.ones_like(llrm)
    else:
        llrm_shape = llrm.shape
        prior_ratio_matrix = prior_ratio_matrix.repeat(llrm_shape[0], llrm_shape[1], 1, 1)

    # No need to neither add 1 to the denominator or avoid the diagonal summation,
    # because the diagonal is explog(p_k/p_k) == 1 and will cancel out
    posterior = 1 / torch.sum(torch.exp(llrm) * prior_ratio_matrix, dim=-2)
    return posterior


def compute_llrm_from_posteriors(
    posteriors: torch.Tensor, prior_ratio_matrix: Optional[torch.Tensor] = None
):
    """
    Args:
    - posteriors: PyTorch Tensor of size (batch_size, time_steps, num_classes).
                  Contains the posterior probabilities for each class.
    - prior_ratio_matrix: Optional PyTorch Tensor of size (num_classes, num_classes).
                          If provided, must be strictly antisymmetric matrix, diagonal must be all-ones.
                          chi_kl = p(label_k) / p(label_l)
                          If not provided, uniform priors are assumed, simplifying to 1.
    Returns:
    - llrm: PyTorch Tensor of size (batch_size, time_steps, num_classes, num_classes).
            Contains log likelihood ratios, computed as lambda_kl = log(p(x | label_k) / p(x | label_l)).
    """
    num_classes = posteriors.shape[-1]
    if prior_ratio_matrix is None:
        # Assuming uniform priors, which simplifies the prior ratio to 1
        prior_ratio_matrix = torch.ones(num_classes, num_classes, device=posteriors.device)
        torch.diagonal(prior_ratio_matrix).fill_(1)  # Ensuring diagonal is all-ones

    # Expand posteriors to enable computation of ratios between all class pairs
    p_k = posteriors.unsqueeze(-1)  # Shape: (batch_size, time_steps, num_classes, 1)
    p_l = posteriors.unsqueeze(2)  # Shape: (batch_size, time_steps, 1, num_classes)

    # Compute the matrix of posterior ratios p(x | label_k) / p(x | label_l)
    posterior_ratio_matrix = p_k / p_l  # Shape: (batch_size, time_steps, num_classes, num_classes)

    # Compute the log-likelihood ratio matrix (LLRM)
    # lambda_kl = log((p(x | label_k) / p(x | label_l)) * (p(label_k) / p(label_l)))
    # Here, prior_ratio_matrix needs to be expanded to match the batch and time_steps dimensions
    llrm_shape = (
        posteriors.shape[0],
        posteriors.shape[1],
        prior_ratio_matrix.shape[0],
        prior_ratio_matrix.shape[1],
    )
    expanded_prior_ratio_matrix = (
        prior_ratio_matrix.unsqueeze(0).unsqueeze(0).repeat(llrm_shape[0], llrm_shape[1], 1, 1)
    )

    llrm = torch.log(posterior_ratio_matrix * expanded_prior_ratio_matrix)

    return llrm


def multiplet_crossentropy(logits_slice, labels_slice, config):
    """Multiplet loss for density estimation of time-series data.
    Args:
        model: A model.backbones_lstm.LSTMModel object.
        logits_slice: An logit Tensor with shape
            ((effective) batch size, order of SPRT + 1, num classes).
            This is the output from LSTMModel.call(inputs, training).
        labels_slice: A label Tensor with shape ((effective) batch size,)
    Returns:
        xent: A scalar Tensor. Sum of multiplet losses.
    """
    # check if necessary parameters are defined in the config file
    requirements = set(
        [
            "IS_CLASS_BALANCE",
            "NUM_CLASSES",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    # Calc multiplet losses
    logits = logits_slice.permute(1, 0, 2)
    logits = logits.reshape(-1, logits.shape[2])
    labels = labels_slice.repeat(logits_slice.shape[1])

    if conf.is_class_balance:
        xent = F.cross_entropy(logits, labels, reduction="none")
        # labels_oh = F.one_hot(labels, conf.num_classes).to(torch.float32)
        # weight xent matrix with class histogram
        xent = torch.mean(xent)
    else:
        xent = F.cross_entropy(logits, labels)

    return xent


@logger.catch
def compute_loss_and_metrics(model, x, labels, config):
    """Calculate loss and gradients.
    Args:
        model: A tf.keras.Model object.
        x: A Tensor. A batch of time-series input data
            without sequential_slice and sequential_concat.
        y: A Tensor. A batch of labels
            without sequential_slice and sequential_concat.
        training: A boolean. Training flag.
        order_sprt: An int. The order of the SPRT.
        time_steps: An int. Num of frames in a sequence.
        num_thresh: An int. Num of thresholds for AUC-SAT.
        beta: A positive float for beta-sigmoid in AUC-SAT.
        param_multiplet_loss: A float. Loss weight.
        param_llr_loss: A float. Loss weight.
        param_aucsat_loss: A float. Loss weight.
        param_wd: A float. Loss weight.
        flag_wd: A boolean. Weight decay or not.
    Returns:
        gradients: A Tensor or None.
        losses: A list of loss Tensors; namely,
            total_loss: A scalar Tensor or 0 if not calc_grad.
                The weighted total loss.
            multiplet_loss: A scalar Tensor.
            llr_loss: A scalar Tensor.
            aucsat_loss: A scalar Tensor.
            wd_reg: A scalar Tensor.
        logits_concat: A logit Tensor with shape
            (batch_size, (time_steps - order_sprt), order_sprt + 1, num classes).
            This is the output from
            utils.data_processing.sequential_concat(logit_slice, labels_slice).
    Remarks:
        - All the losses below will be calculated if not calc_grad
          for TensorBoard logs.
            total_loss
            multiplet_loss
            llr_loss
            aucsat_loss
            wd_reg
    """
    # check if necessary parameters are defined in the config file
    requirements = set(
        [
            "ORDER_SPRT",
            "TIME_STEPS",
            "PARAM_MULTIPLET_LOSS",
            "PARAM_LLR_LOSS",
            "IS_TAPERING_THRESHOLD",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    y_slice = sequential_slice_labels(labels, conf.time_steps, conf.order_sprt)

    if conf.is_tapering_threshold:
        llrs, logits_slice, kappa = model.forward(x)
    else:
        llrs, logits_slice = model.forward(x)

    lllr = compute_loss_for_llr(llrs, labels, config)

    # multiplet_loss
    mce = multiplet_crossentropy(logits_slice, y_slice, config)

    # L2 weight decay regularization
    wd_reg = calc_weight_decay(model)

    # use constant weights predefined in config file
    total_loss = conf.param_multiplet_loss * mce + conf.param_llr_loss * lllr

    # store loss values
    losses = {
        "total_loss": total_loss,
        "multiplet_crossentropy (MCE)": mce,
        "LLR_estimation (LLRE)": lllr,
        "weight_decay": wd_reg,
    }

    # store performance metrics and training status with the losses
    monitored_values = {
        "losses": losses,
        "llrs": llrs,
    }
    if conf.is_tapering_threshold:
        monitored_values["kappa"] = kappa

    return monitored_values
