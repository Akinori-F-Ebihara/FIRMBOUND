import itertools
import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, cast

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from loguru import logger
from termcolor import colored
from torch import Tensor
from tqdm import tqdm

from models.losses import retrieve_posteriors_from_llrm
from utils.misc import StopWatch, extract_params_from_config, import_config_from_path
from utils.performance_metrics import get_upper_triangle, seqconfmx_to_macro_ave_sns
from utils.training import PyTorchPhaseManager, move_data_to_device, prepare_for_training


def set_global_cholesky_jitter(jitter_val):
    gpytorch.settings.cholesky_jitter(jitter_val)


class GPModel(ApproximateGP):
    """
    Adapted from:
    https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html
    """

    def __init__(self, inducing_points):
        """
        This method constructs a mean module, a kernel module, a variational distribution object,
        and a variational strategy object. This method should also be responsible for construting
        whatever other modules might be necessary.
        """
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        """
        Args:
        - x: input data matrix of size (batch_size, data_dim).

        Returns:
        A MultivariateNormal with the prior mean and covariance evaluated at x.
        Specifically, the vector mu_x and matrix K_xx (batch_size, batch_size)
        representing the prior mean and covariance matrix of the GP.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def stochastic_variational_ELBO_maximization(
    model, likelihood, stat, G_min, num_epochs, batch_size
):
    assert len(stat) == len(G_min)
    num_iter = int(np.ceil(len(stat) // batch_size))

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=0.02,
    )

    # Our loss object. We're using the VariationalELBO
    ELBO = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=G_min.size(0))

    # epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
    loss_pool = []
    for i in range(num_epochs):
        # Within each iteration, we will go over each minibatch of data
        for iter in range(num_iter):
            x_batch = stat[iter * batch_size : (iter + 1) * batch_size]
            y_batch = G_min[iter * batch_size : (iter + 1) * batch_size]

            optimizer.zero_grad()
            output = model(x_batch)
            loss = -ELBO(output, y_batch)
            if np.isnan(loss.item()):
                raise ValueError("Got NaN in the loss!")
            loss_pool.append(loss.item())
            loss.backward()
            optimizer.step()
        if i == 0 or (i + 1) % 10 == 0:
            logger.info(f"Epoch {i}, {loss.item()=}")
    return loss_pool, model


def conditional_expectation_G_given_lmbda(model, likelihood, stat):
    # BOTH model AND likelihood must enter the eval mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # compute likelihood and confidence from the predictive
        prediction = likelihood(model(stat))
        lower, upper = prediction.confidence_region()

    return prediction.mean, lower, upper


def conditional_expectation_G_given_lmbda_cfl(model, stat):
    with torch.no_grad():
        prediction_mean = -model.predict(stat)

    return prediction_mean  # convert the convex to concave (orig)


def initialize_Gaussian_process_model(stat, num_inducing_points=100):
    # CAUTION: too many inducing points may lead to nan value in ELBO
    device = stat.device
    dice = np.random.randint(0, len(stat), num_inducing_points)
    inducing_points = stat[dice]
    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = model.to(device)
    likelihood = likelihood.to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=0.02,
    )

    return model, likelihood, optimizer


def get_sufficient_statistic(llrs):
    """
    Args:
    - llrs: Tensor of shape (batch_size, time_steps, num_classes, num_classes)

    Returns:
    - sufficient_statistic: Tensor of shape (batch_size, time_steps, num_classes*(num_classes-1)/2)
    """
    # Get the indices of the upper triangular part of the tensor
    rows, cols = torch.triu_indices(llrs.shape[-2], llrs.shape[-1], offset=1)

    # Use these indices to index into the tensor
    sufficient_statistic = llrs[:, :, rows, cols]

    return sufficient_statistic


def G_st(posteriors, penalty=1.0):
    """
    Instantaneous stopping risk (or negative gain) function.

    Args:
    -posteriors: posteriors p(label_y | data_x).
                   PyTorch tensor of shape (batch_size, time_steps, num_classes).
    -penalty_matrix (optional): penalty for choosing each class.
                                PyTorch tensor of shape (num_classes,)
                                All-ones if not provided.
    Returns:
    - risk: Tensor of shape (batch_size, time_steps). G_st
    - predicted_labels: Tensor of shape (batch_size, time_steps). argmax(G_st)
    """

    penalty_matrix = torch.ones_like(posteriors) * penalty

    assert posteriors.shape[-1] == penalty_matrix.shape[-1]

    risk, predicted_labels = torch.min(penalty_matrix * (1 - posteriors), dim=-1)
    return risk, predicted_labels


@logger.catch
def backward_induction(
    llrs,
    num_epochs,
    batch_size,
    num_inducing_points,
    penalty=1.0,
    cost=0.1,
    is_llr_as_sufficient_statistic=True,
    dirname="../logs/GPs/",
    device=None,
):
    ###

    data_num, time_steps, num_classes, _ = llrs.shape
    if not device:
        device = llrs.device

    # (data_num, time_steps, num_classes)
    posteriors = retrieve_posteriors_from_llrm(llrs)

    # precompute G_st, the stopping risk (data_num, time_steps) \in {0, 1, ..., num_classes}
    stopping_risk, predicted_labels = G_st(posteriors, penalty)
    continuation_risk = torch.zeros((data_num, time_steps)).to(device)
    minimal_risk = torch.zeros((data_num, time_steps)).to(device)

    # Initialize the last G
    continuation_risk[:, -1] = stopping_risk[:, -1]
    minimal_risk[:, -1] = stopping_risk[:, -1]

    loss_pool = []
    model_pool = []
    likelihood_pool = []
    for t in tqdm(
        reversed(range(time_steps - 1)),
        mininterval=5,
        desc=colored("Reverse time step progress: ", "blue"),
        colour="blue",
        # disable=dist.get_rank() != 0,
    ):
        G_min = minimal_risk[:, t + 1]  # expected risk at the next time step (data_num)

        if is_llr_as_sufficient_statistic:
            sufficient_statistic = get_upper_triangle(llrs)[
                :, t
            ]  # (data_num, kC2), where k is num_classes. current sufficient statistic at hand
        else:
            sufficient_statistic = posteriors[
                :, t
            ]  # (data_num, num_classes) current sufficient statistic at hand

        model, likelihood, _ = initialize_Gaussian_process_model(
            sufficient_statistic, num_inducing_points
        )

        loss, model = stochastic_variational_ELBO_maximization(
            model, likelihood, sufficient_statistic, G_min, num_epochs, batch_size
        )
        loss_pool.append(loss)
        model_pool.append(model)
        likelihood_pool.append(likelihood)

        (
            cond_mean,
            _,  # confidence_lower,
            _,  # confidence_upper,
        ) = conditional_expectation_G_given_lmbda(
            model,
            likelihood,
            sufficient_statistic,
        )

        continuation_risk[:, t] = cond_mean + torch.tensor(cost)
        minimal_risk[:, t] = torch.min(stopping_risk[:, t], continuation_risk[:, t])

    return (
        continuation_risk,
        stopping_risk,
        predicted_labels,
        loss_pool,
        model_pool,
        likelihood_pool,
    )


def posterior_grid_to_llrs_upperdiag(x_new):
    assert x_new.ndim == 2
    x_new += 1e-16
    batch_size, num_classes = x_new.shape
    device = x_new.device
    kC2 = int(num_classes * (num_classes - 1) / 2)
    llrs_upperdiag = torch.zeros(batch_size, kC2, device=device)
    ind = 0
    for i in range(num_classes):
        for j in range(num_classes):
            if j <= i:
                continue
            llrs_upperdiag[:, ind] = x_new[:, i] / x_new[:, j]
            ind += 1

    return torch.log(llrs_upperdiag)


def predictive_continuation_risk(x_new, likelihood, model, cost):
    likelihood.eval()
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return likelihood(model(x_new)).mean.detach().cpu().numpy() + cost


def predictive_continuation_risk_cfl(x_new, model, cost):
    with torch.no_grad():
        return -model.predict(x_new).detach().cpu().numpy() + cost


def predictive_stopping_risk(x_new, penalty):
    G, _ = G_st(x_new, penalty)
    return G.cpu().numpy()


def posterior_grid(num_classes, device="cuda"):
    """
    Note: The number of combinations can be found with the Stars and Bars Theorem:
    The problem is boiled down to assign k indistinguishable items into n distinguishable bins.
    In our case k = 1.0 (sum of prob) / step and n = num_classes.
    Thus the number of combinations, len(x_new), is n+k-1Ck = n+k-1Cn-1
    """
    if num_classes == 2:
        step = 0.001
    elif num_classes == 3:
        step = 0.1
    else:
        raise ValueError("Classes > 3 is not supported due to computational load.")

    # All discrete values from 0.0 to 1.0 with 0.1 step
    values = np.arange(0.0, 1.00 + step, step)

    combinations = list(itertools.product(values, repeat=num_classes))

    # Convert the list of combinations
    x_new = np.array(combinations)
    x_new = x_new[np.isclose(x_new.sum(axis=1), 1.0)]
    x_new = torch.tensor(x_new).to(device).float()

    return x_new


def decision_helper(
    model_pool,
    likelihood_pool,
    penalty,
    cost,
    llrs,
    is_llr_as_sufficient_statistic=False,
):
    """
    Args:
    - model_pool (list): of length time_steps containing GPyTorch models.
    - likelihood_pool (list): of length time_steps containing GPyTorch likelihoods.
    - cost (float): sampling cost
    - llrs (Tensor): of size (batch_size, time_steps, num_classes, num_classes)

    Returns:
    - (masked) hitting_time (Tensor): of size (batch_size, time_steps) of binary elements.
       1. indicates the hit time. Only the first hit time is selected.
    """
    batch_size, time_steps, num_classes, num_classes_ = llrs.shape
    assert num_classes == num_classes_
    assert time_steps == len(model_pool) + 1
    assert time_steps == len(likelihood_pool) + 1

    # posteriors: A tensor of size (batch_size, time_steps, num_classes)
    posteriors = retrieve_posteriors_from_llrm(llrs)
    assert posteriors.shape[1] == time_steps

    # stopping risk and predicted labels: both size of (batch_size, time_steps)
    Gst, predictions = G_st(posteriors, penalty)

    if is_llr_as_sufficient_statistic:
        sufficient_statistic = get_upper_triangle(llrs)
    else:
        sufficient_statistic = posteriors
    hitting_time = torch.zeros(batch_size, time_steps).int()
    for t in range(1, time_steps):
        # current time: t-1
        current_statistic = sufficient_statistic[:, t - 1, :]
        current_Gst = Gst[:, t - 1]

        # continuation risk at time t
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            Gtilde = likelihood_pool[-t](model_pool[-t](current_statistic)).mean + cost

        hitting_time[:, t - 1] = current_Gst <= Gtilde
    hitting_time[:, t] = torch.ones(batch_size)
    # _, predictions[:, t] = G_st(posteriors[:, t, :], penalty)

    hitting_time = torch.argmax(hitting_time, dim=1).float()
    predictions = predictions[range(len(predictions)), hitting_time.int()]

    hitting_time += 1
    assert torch.min(hitting_time) > 0 and torch.max(hitting_time) <= time_steps

    return hitting_time, predictions


def decision_helper_cfl(
    model_pool,
    penalty,
    cost,
    llrs,
    is_llr_as_sufficient_statistic=False,
):
    """
    Args:
    - model_pool (list): of length time_steps containing GPyTorch models.
    - likelihood_pool (list): of length time_steps containing GPyTorch likelihoods.
    - cost (float): sampling cost
    - llrs (Tensor): of size (batch_size, time_steps, num_classes, num_classes)

    Returns:
    - (masked) hitting_time (Tensor): of size (batch_size, time_steps) of binary elements.
       1. indicates the hit time. Only the first hit time is selected.
    """
    batch_size, time_steps, num_classes, num_classes_ = llrs.shape
    assert num_classes == num_classes_
    assert time_steps == len(model_pool) + 1

    # posteriors: A tensor of size (batch_size, time_steps, num_classes)
    posteriors = retrieve_posteriors_from_llrm(llrs)
    assert posteriors.shape[1] == time_steps

    # stopping risk and predicted labels: both size of (batch_size, time_steps)
    Gst, predictions = G_st(posteriors, penalty)

    if is_llr_as_sufficient_statistic:
        sufficient_statistic = get_upper_triangle(llrs)
    else:
        sufficient_statistic = posteriors
    hitting_time = torch.zeros(batch_size, time_steps).int()
    for t in range(1, time_steps):
        # current time: t-1
        current_statistic = sufficient_statistic[:, t - 1, :]
        current_Gst = Gst[:, t - 1]

        # continuation risk at time t
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            Gtilde = -model_pool[-t].predict(current_statistic) + cost

        hitting_time[:, t - 1] = current_Gst <= Gtilde
    hitting_time[:, t] = torch.ones(batch_size)
    # _, predictions[:, t] = G_st(posteriors[:, t, :], penalty)

    hitting_time = torch.argmax(hitting_time, dim=1).float()
    predictions = predictions[range(len(predictions)), hitting_time.int()]

    hitting_time += 1
    assert torch.min(hitting_time) > 0 and torch.max(hitting_time) <= time_steps

    return hitting_time, predictions


def scale_cost_with_timesteps(cost_pool_base, time_steps, penalty_L):
    cost_pool = cost_pool_base / time_steps * penalty_L
    return cost_pool


def save_gaussian_process_models(
    model_pool, model_path, likelihood_pool, likelihood_path, is_entire_model=True
):
    """
    Args:
    - model_pool: a list of length time_steps - 1, containing GPyTorch models.
    - likelihood_pool: a list of length time_steps - 1, containing GPyTorch likelihoods.
    """
    if is_entire_model:
        torch.save(model_pool, model_path)
        torch.save(likelihood_pool, likelihood_path)
    else:
        state_dicts = [model.state_dict() for model in model_pool]
        torch.save(state_dicts, model_path)

        state_dicts = [likelihood.state_dict() for likelihood in likelihood_pool]
        torch.save(state_dicts, likelihood_path)

    logger.success("Successfully saved GPyTorch models and Likelihoods")


def save_cfl_models(model_pool, model_path, is_entire_model=True):
    """
    Args:
    - model_pool: a list of length time_steps - 1, containing GPyTorch models.
    - likelihood_pool: a list of length time_steps - 1, containing GPyTorch likelihoods.
    """
    if is_entire_model:
        torch.save(model_pool, model_path)
    else:
        state_dicts = [model.state_dict() for model in model_pool]
        torch.save(state_dicts, model_path)

    logger.success("Successfully saved CFL models")


def load_gaussian_process_models(model_path, likelihood_path):
    """
    Args:
    - model_pool: a list of length time_steps - 1, containing GPyTorch models.
    - likelihood_pool: a list of length time_steps - 1, containing GPyTorch likelihoods.
    """

    model = torch.load(model_path)
    likelihood = torch.load(likelihood_path)

    logger.success("Successfully loaded GPyTorch models and Likelihoods")

    return model, likelihood


def load_cfl_models(model_path):
    """
    Args:
    - model_pool: a list of length time_steps - 1, containing GPyTorch models.
    - likelihood_pool: a list of length time_steps - 1, containing GPyTorch likelihoods.
    """

    model = torch.load(model_path)

    logger.success("Successfully loaded CFL models")

    return model


def save_gp_model(
    model_pool,
    likelihood_pool,
    continuation_risk,
    stopping_risk,
    predicted_labels,
    loss_pool,
    dirname,
):
    save_gaussian_process_models(
        model_pool,
        dirname + "/GPmodels.pt",
        likelihood_pool,
        dirname + "/Likelihoodmodels.pt",
    )
    np.save(dirname + "/continuation_risk", continuation_risk.cpu())
    np.save(dirname + "/stopping_risk", stopping_risk.cpu())
    np.save(dirname + "/predicted_labels", predicted_labels.cpu())
    loss_pool = np.stack(loss_pool)
    # plt.plot(loss_pool.T)
    plt.imshow(loss_pool, aspect="auto")
    plt.colorbar()
    plt.ylabel("iteration")
    plt.xlabel("time step (w/o the horizon)")
    plt.savefig(f"{dirname}/Negative_ELBO.pdf")
    plt.close()
    np.save(dirname + "/loss_pool", loss_pool)


def save_cfl_model(
    model_pool,
    continuation_risk,
    stopping_risk,
    predicted_labels,
    loss_pool,
    dirname,
):
    save_cfl_models(
        model_pool,
        dirname + "/CFLmodels.pt",
    )
    np.save(dirname + "/continuation_risk", continuation_risk.cpu())
    np.save(dirname + "/stopping_risk", stopping_risk.cpu())
    np.save(dirname + "/predicted_labels", predicted_labels.cpu())
    loss_pool = torch.stack(loss_pool).cpu().numpy()
    plt.plot(loss_pool)
    plt.xlabel("Time steps")
    plt.ylabel("Train loss")
    plt.savefig(f"{dirname}/Train_score.pdf")
    plt.close()
    np.save(dirname + "/loss_pool", loss_pool)


def GP_training_routine(
    llrs,
    penalty,
    cost,
    subproject,
    num_epochs=30,
    batch_size=2000,
    num_inducing_points=200,
    is_llr_as_sufficient_statistic=False,
    savedir="./logs/GPs/",
):
    shapes = llrs.shape
    assert shapes[2] == shapes[3]
    # num_classes = shapes[2]

    dirname = (
        f"{savedir}/{subproject}/penalty{penalty}_cost{cost}_batchsize{batch_size}"
        f"_indpnt{num_inducing_points}_epoch{num_epochs}_isllrs{is_llr_as_sufficient_statistic}"
    )
    Path(dirname).mkdir(parents=True, exist_ok=True)

    (
        continuation_risk,
        stopping_risk,
        predicted_labels,
        loss_pool,
        model_pool,
        likelihood_pool,
    ) = backward_induction(
        llrs,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_inducing_points=num_inducing_points,
        penalty=penalty,
        cost=cost,
        is_llr_as_sufficient_statistic=is_llr_as_sufficient_statistic,
        dirname=dirname,
    )
    save_gp_model(
        model_pool,
        likelihood_pool,
        continuation_risk,
        stopping_risk,
        predicted_labels,
        loss_pool,
        dirname,
    )

    return model_pool, likelihood_pool


def CFL_training_routine(
    llrs,
    penalty,
    cost,
    subproject,
    config_cfl,
):
    shapes = llrs.shape
    assert shapes[2] == shapes[3]
    # num_classes = shapes[2]

    requirements = set(
        [
            "BATCH_SIZE_TUNING",
            "BATCH_SIZE_FITTING",
            "NUM_FOLDS",
            "T_MULT",
            "RANGE_LAMBDA_OPTUNA",
            "IS_LLR_AS_SUFFICIENT_STATISTIC",
            "SAVEDIR",
        ]
    )
    conf = extract_params_from_config(requirements, config_cfl)

    dirname = (
        f"{conf.savedir}/{subproject}/penalty{penalty}_cost{cost}"
        f"_bst{conf.batch_size_tuning}bsf{conf.batch_size_fitting}"
        f"_{conf.num_folds}fold_tmult{conf.t_mult}_{conf.range_lambda_optuna}"
        f"_isllrs{conf.is_llr_as_sufficient_statistic}"
    )
    Path(dirname).mkdir(parents=True, exist_ok=True)

    (
        continuation_risk,
        stopping_risk,
        predicted_labels,
        loss_pool,
        model_pool,
    ) = backward_induction_convexfunc(
        llrs,
        penalty,
        cost,
        conf.is_llr_as_sufficient_statistic,
        dirname,
        config_cfl,
    )
    save_cfl_model(
        model_pool,
        continuation_risk,
        stopping_risk,
        predicted_labels,
        loss_pool,
        dirname,
    )

    return model_pool


def backward_induction_convexfunc(
    llrs,
    penalty,
    cost,
    is_llr_as_sufficient_statistic,
    dirname,
    config_cfl,
):
    requirements = set(["BATCH_SIZE_TUNING", "BATCH_SIZE_FITTING", "IS_SAVE_MEMORY"])
    conf = extract_params_from_config(requirements, config_cfl)
    data_num, time_steps, num_classes, _ = llrs.shape

    device = llrs.device

    if conf.is_save_memory:
        if is_llr_as_sufficient_statistic:
            raise ValueError("Cannot use LLR as sufficient statistic while is_save_memory=True!")
        posteriors = retrieve_posteriors_from_llrm(llrs.cpu())
        del llrs
        posteriors = posteriors.to(device)
    else:
        # (data_num, time_steps, num_classes)
        posteriors = retrieve_posteriors_from_llrm(llrs)

    # precompute G_st, the stopping risk (data_num, time_steps) \in {0, 1, ..., num_classes}
    stopping_risk, predicted_labels = G_st(posteriors, penalty)
    continuation_risk = torch.zeros((data_num, time_steps)).to(device)
    minimal_risk = torch.zeros((data_num, time_steps)).to(device)

    # Initialize the last G
    continuation_risk[:, -1] = stopping_risk[:, -1]
    minimal_risk[:, -1] = stopping_risk[:, -1]

    loss_pool = []
    model_pool = []
    for t in tqdm(
        reversed(range(time_steps - 1)),
        mininterval=5,
        desc=colored("Reverse time step progress: ", "blue"),
        colour="blue",
        # disable=dist.get_rank() != 0,
    ):
        G_min = minimal_risk[:, t + 1]  # expected risk at the next time step (data_num)

        if is_llr_as_sufficient_statistic:
            sufficient_statistic = get_upper_triangle(llrs)[
                :, t
            ]  # (data_num, kC2), where k is num_classes. current sufficient statistic at hand
        else:
            sufficient_statistic = posteriors[
                :, t
            ]  # (data_num, num_classes) current sufficient statistic at hand

        # tuning with Optuna
        seed_stat, seed_Gmin = select_seed_data(sufficient_statistic, G_min, conf.batch_size_tuning)
        study = tuning_cfl(seed_stat, seed_Gmin, config_cfl, dirname, current_time=t)

        # fitting
        seed_stat, seed_Gmin = select_seed_data(
            sufficient_statistic, G_min, conf.batch_size_fitting
        )
        model, loss = fit_cfl(seed_stat, seed_Gmin, config_cfl, study)

        loss_pool.append(loss)
        model_pool.append(model)

        cond_mean = conditional_expectation_G_given_lmbda_cfl(model, sufficient_statistic)

        continuation_risk[:, t] = cond_mean + torch.tensor(cost)
        minimal_risk[:, t] = torch.min(stopping_risk[:, t], continuation_risk[:, t])

    return continuation_risk, stopping_risk, predicted_labels, loss_pool, model_pool


def select_seed_data(sufficient_statistic, G_min, batch_size_cfl):
    # select seed data (equivalent to the inducing points of Gaussian Process Regression)
    dice = np.random.permutation(len(sufficient_statistic))[:batch_size_cfl]
    seed_stat = sufficient_statistic[dice]
    seed_Gmin = G_min[dice]
    return seed_stat, seed_Gmin


def initialize_cfl_model(config_cfl):
    requirements = set(
        [
            "MAX_HYPER_ITER",
            "NUM_FOLDS",
            "LS_LAMBDAS",
            "LS_LAMBDAS_MULT_LOW",
            "LS_LAMBDAS_MULT_HIGH",
            "LS_LAMBDAS_MULT_FINE",
        ]
    )
    conf = extract_params_from_config(requirements, config_cfl)

    model = ConvexRegressionModel(
        conf.max_hyper_iter,
        conf.num_folds,
        conf.ls_lambdas,
        conf.ls_lambdas_mult_low,
        conf.ls_lambdas_mult_high,
        conf.ls_lambdas_mult_fine,
    )
    return model


def objective_cfl(
    trial,
    X_train: Tensor,
    y_train: Tensor,
    dim_data: int,
    num_data: int,
    lam_low: float,
    lam_high: float,
    config_cfl: Dict,
):
    requirements = set(
        [
            "NUM_FOLDS",
        ]
    )
    conf = extract_params_from_config(requirements, config_cfl)

    """For Optuna"""
    print("Suggest lambda_")
    lambda_ = trial.suggest_float(name="lambda_", low=lam_low, high=lam_high, log=True)
    # check if necessary parameters are defined in the config file

    model = initialize_cfl_model(config_cfl)
    model.update_lambda_rho(d=dim_data, lambda_=lambda_, n=num_data)

    with StopWatch(unit="sec"):
        mean_loss = model.cross_validate(
            X_train, y_train, num_folds=conf.num_folds
        )  # cross validation

    # Verbose
    a_result, y_hat_result = model.show_params()
    logger.success(f"lambda_, rho = {model.lambda_}, {model.rho}")

    return mean_loss


def tuning_cfl(
    seed_stat: Tensor, seed_G_min: Tensor, config_cfl: Dict, dirname: str, current_time: int
):
    requirements = set(
        [
            "STUDY_NAME",
            "RANGE_LAMBDA_OPTUNA",
            "NUM_TRIALS",
        ]
    )
    conf = extract_params_from_config(requirements, config_cfl)

    study_name = conf.study_name + f"_time{current_time}"
    path_db = dirname + f"/{study_name}.db"
    storage_name = "sqlite:///" + path_db

    study = optuna.create_study(
        storage=storage_name,
        study_name=study_name,
        load_if_exists=True,
        direction="minimize",
    )
    if len(study.trials) >= conf.num_trials:
        logger.info("Tuning was already done. Skipping this timestep...")
        return study

    lam_low = conf.range_lambda_optuna[0]
    lam_high = conf.range_lambda_optuna[1]
    num_data, dim_data = seed_stat.shape
    study.optimize(
        lambda trial: objective_cfl(
            trial,
            X_train=seed_stat,
            y_train=-seed_G_min,
            dim_data=dim_data,
            num_data=num_data,
            lam_low=lam_low,
            lam_high=lam_high,
            config_cfl=config_cfl,
        ),
        n_trials=conf.num_trials,
        timeout=None,
        n_jobs=1,
        gc_after_trial=True,  # Set True if you observe memory leak over trials
        show_progress_bar=True,
    )
    return study


def fit_cfl(seed_stat, seed_G_min, config_cfl, study):
    model = initialize_cfl_model(config_cfl)

    with StopWatch(unit="sec"):
        model.fit(seed_stat, -seed_G_min, lambda_=study.best_params["lambda_"])
        train_loss = model.score(seed_stat, -seed_G_min)

    a_result, y_hat_result = model.show_params()
    logger.info(f"lambda_, rho = {model.lambda_}, {model.rho}")
    logger.info(f"Show params: a = {a_result}, y_hat = {y_hat_result}")

    return model, train_loss


def extract_llrs_from_dataset(
    path_modeldir,
    phase,  # 'train', 'val', or 'test
    gpu,
    cnt_model=0,
):
    # find the model ckpt
    for s in os.listdir(path_modeldir):
        if "ckpt_" in s:
            ckpt_path = path_modeldir + "/" + s
    if "ckpt_path" not in locals():
        logger.error("ckpt is missing in this folder...")
    else:
        cnt_model += 1

    # setup config
    config_dir = path_modeldir + "/configs/"  # representative config path
    for s in os.listdir(config_dir):
        if "cooked.py" in s:
            config_path = config_dir + s

    config = import_config_from_path(config_path)
    config["GPU"] = int(gpu)
    config["IS_LOAD_ONTO_MEMORY"] = False

    if "train" in phase:
        phase = "train"
        num_data = config["NUM_TRAIN"]
    elif "val" in phase:
        phase = "val"
        num_data = config["NUM_VAL"]
    elif "test" in phase:
        phase = "test"
        num_data = config["NUM_TEST"]

    batch_size = config["BATCH_SIZE"]
    time_steps = config["TIME_STEPS"]
    num_classes = config["NUM_CLASSES"]
    config["IS_RESUME"] = True  # of course
    config["PATH_RESUME"] = ckpt_path
    config["IS_LOAD_ONTO_MEMORY"] = False

    # load model, optimizer and their weights
    model, optimizer, scheduler, data_loaders, tb_writer = prepare_for_training(
        config, is_load_test=True, is_ddp=False
    )
    model.eval()

    # compute LLRs
    llrs_all = np.zeros((num_data, time_steps, num_classes, num_classes), dtype="float16")
    labels_all = np.zeros(num_data, dtype="int32")
    logger.info("iterating over dataset to compute the LLR...")

    for iter_b, feats in enumerate(data_loaders[phase]):
        result = move_data_to_device(feats, is_ddp=False)

        if config["IS_REAL_WORLD"]:
            x_batch, y_batch = cast(Tuple[Tensor, Tensor], result)
        else:
            x_batch, y_batch, _ = cast(Tuple[Tensor, Tensor, Tensor], result)

        with PyTorchPhaseManager(
            model,
            optimizer,
            phase="eval",
        ) as p:
            llrs = model.forward(x_batch)[0]

        # store LLRs and labels
        llrs_all[iter_b * batch_size : (iter_b + 1) * batch_size] = llrs.cpu().numpy()
        labels_all[iter_b * batch_size : (iter_b + 1) * batch_size] = y_batch.cpu().numpy()
    return llrs_all, labels_all, cnt_model, num_data, time_steps


def calc_sns(labels_concat, preds):
    """ """
    _device = labels_concat.device
    device = preds.device
    assert (
        _device == device
    ), f"labels and predictions must be on the same device but got labels on {_device} and predictions on {device}."

    num_classes = torch.max(labels_concat) + 1
    preds = F.one_hot(preds, num_classes=num_classes).to(device)
    # (batch_size, num_classes)

    labels_oh = F.one_hot(labels_concat, num_classes=num_classes)
    # (batch_size, num_classes)

    labels_oh = labels_oh.unsqueeze(-1).unsqueeze(0)
    # (1 (=current num_thresh), batch_size, num_classes, 1)

    preds = preds.unsqueeze(-2).unsqueeze(0)
    # (1 (=current num_thresh), batch_size, 1, num_classes)

    confmx = torch.sum(labels_oh * preds, dim=1, dtype=torch.int32)
    # seqconfmx: A series of confusion matrix Tensors
    #        with shape (series length (arbitrary), num classes, num classes).
    ls_sns = seqconfmx_to_macro_ave_sns(confmx)

    return ls_sns


""" [Convex function learning utils START]

Adapted from:
- Siahkamari, Ali, et al. "Faster algorithms for learning convex functions
  International Conference on Machine Learning. PMLR, 2022. https://arxiv.org/abs/2111.01348
  GitHub: https://github.com/Siahkamari/Piecewise-linear-regression/blob/master/Python/piecewise_linear_estimation_v0.py

# Remarks
- The original algorithm assumes sum_i x_i = sum_i y_i = 0 so that sum_i hat{y}_i = 0 to simplify the solution,
  but the original code given in the GitHub does not assume it.
- The tuning method is a bit modified from the original code. Specifically, the definition of the best lambda_
  during auto_tune is different from the original one. Our code takes overfitting into consideration.

"""


def calc_from_to(index: int, batch_size: int) -> Tuple[int, int]:
    from_ = batch_size * index
    to = batch_size * (index + 1)
    return from_, to


def calc_num_iter(num_data: int, batch_size: int) -> int:
    num_iter = num_data // batch_size
    if num_data % batch_size != 0:
        num_iter += 1
    return num_iter


def calc_caninocal_rho(d: int, lambda_: float, n: int) -> float:
    return (np.sqrt(d) * lambda_**2) / n


def calc_Sigma_i(init_Sigma_i, num_data, X_train, y_train, eye) -> Tuple[Tensor, Tensor]:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """
    sumjxjxjT = torch.matmul(X_train.T, X_train)  # [dim_data,dim_data]
    X_bar = torch.mean(X_train, dim=0)  # [dim_data,] #Isn't this =0?
    y_bar = torch.mean(y_train)  # scalar #Is't this =0?

    Sigma_i = init_Sigma_i
    for i in range(num_data):
        nxxT_sumxxT = (
            num_data * torch.outer(X_train[i], X_train[i])
            + sumjxjxjT
            - num_data * (torch.outer(X_bar, X_train[i]) + torch.outer(X_train[i], X_bar))
        )  # [dim_data, dim_data]
        Sigma_i[i, :, :] = torch.inverse(nxxT_sumxxT + eye)  # [dim_data, dim_data]

    return Sigma_i, y_bar  # [num_data,dim_data,dim_data],scalar


@torch.no_grad()
def y_hat_update(alpha, s, a, X_train, num_data, y_bar, rho, y_train) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """
    temp1 = (
        torch.sum(
            alpha.T - alpha + s.T - s + torch.matmul(X_train, a.T) - torch.matmul(a, X_train.T),
            dim=1,
        )
        + num_data * torch.sum(a * X_train, dim=1)
        - torch.sum(a * X_train)
        + num_data * 2 * y_bar
    )  # [num_data,num_data] -> [num_data,]

    y_hat = (
        2 / (2 + num_data * rho) * y_train + rho / (2 + num_data * rho) / 2 * temp1
    )  # [num_data,]
    return y_hat


@torch.no_grad()
def y_hat_z_update_dc(
    alpha, s, a, X_train, num_data, y_bar, beta, t, b, rho, y_train
) -> Tuple[Tensor, Tensor]:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """
    temp1 = (
        torch.sum(
            alpha.T - alpha + s.T - s + torch.matmul(X_train, a.T) - torch.matmul(a, X_train.T),
            dim=1,
        )
        + num_data * torch.sum(a * X_train, dim=1)
        - torch.sum(a * X_train)
        + num_data * 2 * y_bar
    )  # [num_data,num_data] -> [num_data,]
    temp2 = (
        torch.sum(
            beta.T - beta + t.T - t - torch.matmul(b, X_train.T) + torch.matmul(X_train, b.T), dim=1
        )
        + num_data * torch.sum(b * X_train, dim=1)
        - torch.sum(b * X_train)
    )

    y_hat = (
        2 / (2 + num_data * rho) * y_train
        + rho / (2 + num_data * rho) / 2 * temp1
        - rho / (2 + num_data * rho) / 2 * temp2
    )
    z = (
        -1 / (2 + num_data * rho) * y_train
        + 1 / (2 * num_data) / (2 + num_data * rho) * temp1
        + (1 + num_data * rho) / (2 * num_data) / (2 + num_data * rho) * temp2
    )
    return y_hat, z


@torch.no_grad()
def a_update(p, eta, alpha, s, y_hat, X, Sigma_i, num_data, dim_data) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    a = p - eta  # [num_data, dim_data]
    a += torch.sum(alpha + s + y_hat.reshape(-1, 1) - y_hat.reshape(1, -1), dim=1).reshape(
        -1, 1
    ) * X - torch.matmul(alpha + s + y_hat.reshape(-1, 1) - y_hat.reshape(1, -1), X)
    a = torch.matmul(Sigma_i, a.reshape(num_data, dim_data, 1)).reshape(
        num_data, dim_data
    )  # (num_data, dim_data)
    return a


@torch.no_grad()
def a_update_dc(p, eta, alpha, s, y_hat, z, X, Sigma_i, num_data, dim_data) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    a = p - eta  # [num_data, dim_data]
    a += torch.sum(
        alpha
        + s
        + y_hat.reshape(-1, 1)
        - y_hat.reshape(1, -1)
        + z.reshape(-1, 1)
        - z.reshape(1, -1),
        dim=1,
    ).reshape(-1, 1) * X - torch.matmul(
        alpha
        + s
        + y_hat.reshape(-1, 1)
        - y_hat.reshape(1, -1)
        + z.reshape(-1, 1)
        - z.reshape(1, -1),
        X,
    )
    a = torch.matmul(Sigma_i, a.reshape(num_data, dim_data, 1)).reshape(
        num_data, dim_data
    )  # (num_data, dim_data)
    return a


@torch.no_grad()
def b_update_dc(q, zeta, beta, t, z, X, Sigma_i, num_data, dim_data) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    b = q - zeta
    b += torch.sum(beta + t + z.reshape(-1, 1) - z.reshape(1, -1), dim=1).reshape(
        -1, 1
    ) * X - torch.matmul(beta + t + z.reshape(-1, 1) - z.reshape(1, -1), X)
    b = torch.matmul(Sigma_i, b.reshape(num_data, dim_data, 1)).reshape(num_data, dim_data)
    return b


@torch.no_grad()
def p_update(p, a, eta, L, u, gamma) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    temp1 = 1 / 2 * (a + eta)  # [num_data,dim_data]
    temp2 = 1 / 2 * (L - u - gamma - torch.abs(p))  # [num_data,dim_data]
    p = torch.sign(temp1) * torch.relu(torch.abs(temp1) + temp2)  # [num_data,dim_data]
    return p


@torch.no_grad()
def p_update_dc(p, a, eta, L, u, gamma) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    temp1 = 1 / 2 * (a + eta)
    temp2 = 1 / 2 * (L - u - gamma - torch.abs(p))
    p = torch.sign(temp1) * torch.relu(torch.abs(temp1) + temp2)
    return p


@torch.no_grad()
def q_update_dc(b, zeta, L, u, gamma, q) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    temp1 = 1 / 2 * (b + zeta)
    temp2 = 1 / 2 * (L - u - gamma - torch.abs(q))
    q = torch.sign(temp1) * torch.relu(torch.abs(temp1) + temp2)
    return q


@torch.no_grad()
def L_update(num_data, rho, gamma, u, p, lambda_) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    L = -1 / (num_data * rho) * lambda_
    L += 1 / num_data * torch.sum(gamma + u + torch.abs(p), dim=0).reshape(1, -1)
    return L  # [1, num_data]


@torch.no_grad()
def L_update_dc(num_data, rho, gamma, u, p, q, lambda_) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    L = -1 / (num_data * rho) * lambda_
    L += 1 / num_data * torch.sum(gamma + u + torch.abs(p) + torch.abs(q), dim=0).reshape(1, -1)
    return L  # [1, num_data]


@torch.no_grad()
def s_update(alpha, y_hat, a, X) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    s = (
        -alpha
        - y_hat.reshape(-1, 1)
        + y_hat.reshape(1, -1)
        + torch.sum(a * X, dim=1).reshape(-1, 1)
        - torch.matmul(a, X.T)
    )  # [num_data,num_data]
    s = torch.relu(s)  # [num_data,num_data]
    return s  # [num_data,num_data]


@torch.no_grad()
def s_update_dc(alpha, y_hat, z, a, X) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    s = (
        -alpha
        - y_hat.reshape(-1, 1)
        + y_hat.reshape(1, -1)
        - z.reshape(-1, 1)
        + z.reshape(1, -1)
        + torch.sum(a * X, dim=1).reshape(-1, 1)
        - torch.matmul(a, X.T)
    )  # [num_data,num_data]
    s = torch.relu(s)  # [num_data,num_data]
    return s  # [num_data,num_data]


@torch.no_grad()
def t_update_dc(beta, z, b, X) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """
    t = (
        -beta
        - z.reshape(-1, 1)
        + z.reshape(1, -1)
        + torch.sum(b * X, dim=1).reshape(-1, 1)
        - torch.matmul(b, X.T)
    )
    t = torch.relu(t)
    return t  # [num_data,num_data]


@torch.no_grad()
def u_update(gamma, L, p) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    u = -gamma + L - torch.abs(p)  # [num_data, dim_data]
    u = torch.relu(u)
    return u  # [num_data, dim_data]


@torch.no_grad()
def u_update_dc(gamma, L, q, p) -> Tensor:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    u = -gamma + L - torch.abs(q) - torch.abs(p)
    u = torch.relu(u)
    return u


@torch.no_grad()
def dual_update(alpha, s, y_hat, a, X, gamma, u, L, p, eta) -> Tuple[Tensor, Tensor, Tensor]:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    alpha += (
        s
        + y_hat.reshape(-1, 1)
        - y_hat.reshape(1, -1)
        - torch.sum(a * X, dim=1).reshape(-1, 1)
        + torch.matmul(a, X.T)
    )  # [num_data,num_data]
    gamma += u - L + torch.abs(p)  # [num_data,dim_data]
    eta += a - p  # [num_data,dim_data]
    return alpha, gamma, eta


@torch.no_grad()
def dual_update_dc(
    alpha, s, y_hat, z, a, X, beta, t, b, gamma, u, L, p, q, eta, zeta
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    - y_hat: [num_data,]
    - z: [num_data,]
    - a: [num_data, dim_data]
    - b: [num_data, dim_data]
    - p: [num_data, dim_data]
    - q: [num_data, dim_data]
    - L: [1, dim_data]
    - s: [num_data, num_data]
    - t: [num_data, num_data]
    - u: [num_data, dim_data]
    - alpha: [num_data, num_data]
    - beta: [num_data, num_data]
    - gamma: [num_data, dim_data]
    - eta: [num_data, dim_data]
    - zeta: [num_data, dim_data]
    """

    alpha += (
        s
        + y_hat.reshape(-1, 1)
        - y_hat.reshape(1, -1)
        + z.reshape(-1, 1)
        - z.reshape(1, -1)
        - torch.sum(a * X, dim=1).reshape(-1, 1)
        + torch.matmul(a, X.T)
    )
    beta += (
        t
        + z.reshape(-1, 1)
        - z.reshape(1, -1)
        - torch.sum(b * X, dim=1).reshape(-1, 1)
        + torch.matmul(b, X.T)
    )
    gamma += u - L + torch.abs(p) + torch.abs(q)
    eta += a - p
    zeta += b - q
    return alpha, beta, gamma, eta, zeta


class BaseClassCFL(metaclass=ABCMeta):
    """Base class of convex function learning models"""

    @abstractmethod
    def __init__(
        self,
        MAX_HYPER_ITER,
        NUM_FOLDS,
        LS_LAMBDAS,
        LS_LAMBDAS_MULT_LOW,
        LS_LAMBDAS_MULT_HIGH,
        LS_LAMBDAS_MULT_FINE,
        T_mult: int = 3,
    ) -> None:
        """
        # Args
        - T_mult:  An int. The number of epochs is T_mult * num_data.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")  # type:ignore
        self.default_dtype = torch.get_default_dtype()

        self.lambda_ = 1.0
        self.rho = 1.0
        self.T_mult = T_mult

        self.max_hyper_iter = MAX_HYPER_ITER
        self.num_folds = NUM_FOLDS
        self.ls_lambdas = LS_LAMBDAS
        self.ls_lambdas_mult_low = LS_LAMBDAS_MULT_LOW
        self.ls_lambdas_mult_high = LS_LAMBDAS_MULT_HIGH
        self.ls_lambdas_mult_fine = LS_LAMBDAS_MULT_FINE

        raise NotImplementedError

    @abstractmethod
    def fit(
        self, X_train: Tensor, y_train: Tensor, lambda_: Optional[float] = None, *args, **kwargs
    ) -> None:
        """
        Fit parameters.

        # Args
        - X_train: A Tensor with shape [num_data, dim_data].
        - y_train: A Tensor with shape [num_data,].
        - lambda_: A float. Lambda in the original paper.
        """
        raise NotImplementedError

    def update_lambda_rho(self, d: int, lambda_: float, n: int) -> None:
        self.lambda_ = lambda_
        self.rho = calc_caninocal_rho(d, lambda_, n)
        logger.info(f"self.lambda_ and self.rho updated: {self.lambda_}, {self.rho}")

    @torch.no_grad()
    def auto_tune(self, X: Tensor, y: Tensor, *args, **kwargs) -> None:
        """Cross validation
        # Args
        - X: A Tensor with shape [num_data, dim_data].
        - y: A Tensor with shape [num_data,].
        - max_hyper_iter: An int. The maximum number of tuning iterations.
        """
        max_hyper_iter = self.max_hyper_iter
        num_folds = self.num_folds

        # Initialize ls_lambdas; will be updated in the for-loop
        ls_lambdas = self.ls_lambdas

        # Start tuning
        for it_tuning in range(max_hyper_iter):
            num_lambdas = len(ls_lambdas)

            # 0. Cross-validation
            loss = torch.zeros(num_lambdas)
            logger.info(f"Tuning iter: {it_tuning + 1} / max{max_hyper_iter}")
            for it_ind, it_lambda_ in tqdm(enumerate(ls_lambdas)):
                self.update_lambda_rho(d=X.shape[1], lambda_=it_lambda_, n=X.shape[0])
                loss[it_ind] = self.cross_validate(X, y, num_folds)
                arg_min = torch.argmin(loss)
            min_lambda_ = ls_lambdas[arg_min]

            # 1. Best lambda_ is the minimum lambda_ in ls_lambdas
            if min_lambda_ == ls_lambdas[0]:
                # Make the candidate lambdas smaller
                ls_lambdas = [k * min_lambda_ for k in self.ls_lambdas_mult_low]

            # 2. Best lambda_ is the maximum lambda_ in ls_lambdas
            elif min_lambda_ == ls_lambdas[-1]:
                # Make the candidate lambdas larger
                ls_lambdas = [k * min_lambda_ for k in self.ls_lambdas_mult_high]

            # 3. Otherwise, start the second stage of the tuning,
            # which consists of two phases:
            else:
                # Phase 1: Make the lambdas finer
                if len(ls_lambdas) == 7:  # 7 is = len of first ls_lambdas.
                    # Update ls_lambdas
                    ls_lambdas = [k * min_lambda_ for k in self.ls_lambdas_mult_fine]

                # Phase 2: Set the best lambda_, fit the other parameters, and finish the tuning.
                else:
                    self.update_lambda_rho(d=X.shape[1], lambda_=min_lambda_, n=X.shape[0])
                    logger.info("********** Final fitting **********")
                    self.fit(X, y, min_lambda_)
                    break

            if it_tuning == max_hyper_iter - 1:
                logger.info("********** Final fitting **********")
                self.fit(X, y, min_lambda_)

    @abstractmethod
    def predict(self, X: Tensor, *args, **kwargs) -> Tensor:
        """
        Outputs \hat{f}, i.e., y_hat

        # Args
        - X: A Tensor with shape [arbitrary number of data, dim_data].

        # Returns
        - pred: A Tensor with shape [X.shape[0],]
        """
        raise NotImplementedError

    @torch.no_grad()
    def cross_validate(self, X: Tensor, y: Tensor, num_folds: int, *args, **kwargs) -> Tensor:
        """N-fold cross validation"""
        num, _ = X.shape

        # Permute the rows of X and y
        rp = torch.randperm(num)
        y = y[rp]
        X = X[rp]

        # Initializing different measure
        loss = torch.zeros(num_folds)

        for i in range(num_folds):
            # Splitting the data to test (leave-out validation data) and train data
            test_start = int(torch.ceil(torch.tensor(num / num_folds * i)))
            test_end = int(torch.ceil(torch.tensor(num / num_folds * (i + 1))))

            I_test = [i for i in range(test_start, test_end)]
            I_train = [i for i in range(test_start)] + [i for i in range(test_end, num)]

            # Learning with the X_train and predicting with it
            self.fit(X[I_train], y[I_train], self.lambda_)

            y_hat_test = self.predict(X[I_test])
            loss[i] = torch.mean((y_hat_test - y[I_test]) ** 2)

        return torch.mean(loss)

    @torch.no_grad()
    def score(self, X: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute R^2 (the coefficient of determination)
        """
        y = y.clone()
        return 1.0 - torch.mean((self.predict(X) - y) ** 2) / torch.var(y)


class ConvexRegressionModel(BaseClassCFL):
    """Algorithm 2 in the original paper. X_train.mean != 0 or y_train != 0 are not currently supported."""

    def __init__(
        self,
        MAX_HYPER_ITER,
        NUM_FOLDS,
        LS_LAMBDAS,
        LS_LAMBDAS_MULT_LOW,
        LS_LAMBDAS_MULT_HIGH,
        LS_LAMBDAS_MULT_FINE,
        T_mult: int = 3,
        *args,
        **kwargs,
    ) -> None:
        """
        # Args
        - T_mult:  An int. The number of epochs is T_mult * num_data.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")  # type:ignore
        self.default_dtype = torch.get_default_dtype()

        self.lambda_ = 1.0
        self.rho = 1.0
        self.T_mult = T_mult

        self.max_hyper_iter = MAX_HYPER_ITER
        self.num_folds = NUM_FOLDS
        self.ls_lambdas = LS_LAMBDAS
        self.ls_lambdas_mult_low = LS_LAMBDAS_MULT_LOW
        self.ls_lambdas_mult_high = LS_LAMBDAS_MULT_HIGH
        self.ls_lambdas_mult_fine = LS_LAMBDAS_MULT_FINE

    @torch.no_grad()
    def fit(
        self, X_train: Tensor, y_train: Tensor, lambda_: Optional[float] = None, *args, **kwargs
    ) -> None:
        """
        Fit parameters.

        # Args
        - X_train: A Tensor with shape [num_data, dim_data].
        - y_train: A Tensor with shape [num_data,].
        - lambda_: A float. Lambda in the original paper.
        """
        # Initialization 1
        self.X_train = X_train
        self.y_train = y_train
        self.num_data = self.X_train.shape[0]
        self.dim_data = self.X_train.shape[1]
        self.num_epochs = int(self.T_mult * self.num_data)
        self.y_hat = torch.zeros(self.num_data, device=self.device, dtype=self.default_dtype)
        self.a = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )

        # Auto-tuning lambda_ with cross-validation or training with a given lambda_
        if lambda_ is None:
            self.auto_tune(self.X_train, self.y_train)
            return
        else:
            self.update_lambda_rho(d=self.dim_data, lambda_=lambda_, n=self.num_data)

        # raise NotImplementedError
        # Initialization 2
        # Primal
        y_hat = torch.zeros(
            self.num_data, device=self.device, dtype=self.default_dtype
        )  # (num_data,)
        a = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )  # (num_data,dim_data)
        p = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )  # (num_data,dim_data)
        # q = torch.zeros(self.num_data, self.dim_data,
        #                 device=self.device, dtype=self.default_dtype)  # (num_data,dim_data)

        L = torch.zeros(1, self.dim_data, device=self.device, dtype=self.default_dtype)

        # Slack
        s = torch.zeros(
            self.num_data, self.num_data, device=self.device, dtype=self.default_dtype
        )  # (num_data,num_data)
        # t = torch.zeros(self.num_data, self.num_data,
        #                 device=self.device, dtype=self.default_dtype)  # (num_data,num_data)
        u = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )  # (num_data,dim_data)

        # Dual
        alpha = torch.zeros(
            self.num_data, self.num_data, device=self.device, dtype=self.default_dtype
        )
        # beta = torch.zeros(
        #     self.num_data, self.num_data,
        #     device=self.device, dtype=self.default_dtype)
        gamma = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )
        eta = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )
        # zeta = torch.zeros(
        #     self.num_data, self.dim_data,
        #     device=self.device, dtype=self.default_dtype)

        init_Sigma_i = torch.zeros(
            [self.num_data, self.dim_data, self.dim_data], device=self.device
        )  # [num_data, dim_data, dim_data]
        # [dim_data,dim_data]
        eye = torch.eye(self.dim_data, device=self.device)

        # Preprocess1
        # =========================================
        Sigma_i, y_bar = calc_Sigma_i(init_Sigma_i, self.num_data, X_train, y_train, eye)

        # ADMM iteration
        # ========================================
        best_score = -float("inf")
        y_hat_best = y_hat
        a_best = a
        for it_t in range(self.num_epochs):
            # Primal updates
            # y_hat update
            y_hat = y_hat_update(
                alpha, s, a, self.X_train, self.num_data, y_bar, self.rho, self.y_train
            )

            # a update
            a = a_update(
                p, eta, alpha, s, y_hat, self.X_train, Sigma_i, self.num_data, self.dim_data
            )

            # p updates
            p = p_update(p, a, eta, L, u, gamma)

            # L update
            L = L_update(self.num_data, self.rho, gamma, u, p, self.lambda_)

            # Slack updates
            # s update
            s = s_update(alpha, y_hat, a, self.X_train)

            # u update
            u = u_update(gamma, L, p)

            # Dual updates
            alpha, gamma, eta = dual_update(alpha, s, y_hat, a, self.X_train, gamma, u, L, p, eta)

            # Save best parameters and verbose
            if (it_t + 1) % 1000 == 0 or it_t == 0 or it_t + 1 == self.num_epochs:
                # Calc parameters
                y_hat_current = y_hat - torch.sum(a * self.X_train, dim=1)
                a_current = a
                # z_current = z - torch.sum(b * self.X_train, dim=1)
                # b_current = b

                # Update parameters
                self.y_hat = y_hat_current
                self.a = a_current
                # self.z = z_current
                # self.b = b_current

                # Calc score
                current_training_score = 1.0 - torch.mean(
                    (self.predict(self.X_train) - self.y_train) ** 2
                ) / torch.var(self.y_train)

                if best_score < current_training_score:
                    # Update the best score
                    best_score = current_training_score
                    _tmp_msg = " Best score updated!"

                    # Save the current best parameters
                    y_hat_best = y_hat_current
                    a_best = a_current
                    # z_best = z_current
                    # b_best = b_current

                else:
                    _tmp_msg = ""

                logger.info(
                    f"Step {it_t + 1} / {self.num_epochs}: Train score = {current_training_score}, "
                    f"Best score = {best_score}{_tmp_msg}"
                )

        # Final result parameters
        self.y_hat = y_hat_best
        self.a = a_best
        # self.z = z_best
        # self.b = b_best

    @torch.no_grad()
    def predict(self, X: Tensor, *args, **kwargs) -> Tensor:
        """
        Outputs Eq. (16.5) in the original paper.

        # Args
        - X: A Tensor with shape [arbitrary number of data, dim_data].

        # Returns
        - pred: A Tensor with shape [X.shape[0],]
        """
        pred, _ = torch.max(
            torch.matmul(X, self.a.T) + self.y_hat.reshape(1, -1), dim=1
        )  # Shape: max([X.shape[0],num_data] + [1,num_data], dim=1) = [X.shape[0],]

        return pred

    def show_params(self):
        return self.a, self.y_hat


class DCRegressionModel(BaseClassCFL):
    """
    Algorithm 4 in the original paper
    DC functions are a very rich class—for instance,
    they are known to contain all C2 functions.
    """

    def __init__(self, T_mult: int = 3) -> None:
        """
        # Args
        - rho: A float. Rho in the original paper.
        - T_mult:  An int. The number of epochs is T_mult * num_data.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")  # type:ignore
        self.default_dtype = torch.get_default_dtype()

        self.lambda_ = 1.0
        self.rho = 1.0
        self.T_mult = T_mult

    @torch.no_grad()
    def fit(
        self, X_train: Tensor, y_train: Tensor, lambda_: Optional[float] = None, *args, **kwargs
    ) -> None:
        """
        # Args
        - X_train: A Tensor with shape [num_data, dim_data].
        - y_train: A Tensor with shape [num_data,].
        - lambda_: A float. Lambda in the original paper.
        """
        # Initialization 1
        self.X_train = X_train
        self.y_train = y_train
        self.num_data = self.X_train.shape[0]
        self.dim_data = self.X_train.shape[1]
        self.num_epochs = int(self.T_mult * self.num_data)

        self.y_hat = torch.zeros(
            self.num_data, device=self.device, dtype=self.default_dtype
        )  # fitting variable1/4
        self.z = torch.zeros(
            self.num_data, device=self.device, dtype=self.default_dtype
        )  # fitting variable2/4
        self.a = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )  # fitting variable3/4
        self.b = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )  # fitting variable4/4

        # Auto-tuning lambda_ with cross-validation or training with a given lambda_
        if lambda_ is None:
            self.auto_tune(self.X_train, self.y_train)
            return
        else:
            self.update_lambda_rho(d=self.dim_data, lambda_=lambda_, n=self.num_data)

        rho = self.rho

        # Initialization 2
        # Primal
        y_hat = torch.zeros(
            self.num_data, device=self.device, dtype=self.default_dtype
        )  # (num_data,)
        z = torch.zeros(self.num_data, device=self.device, dtype=self.default_dtype)  # (num_data,)
        a = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )  # (num_data,dim_data)
        b = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )  # (num_data,dim_data)
        p = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )  # (num_data,dim_data)
        q = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )  # (num_data,dim_data)
        L = torch.zeros(1, self.dim_data, device=self.device, dtype=self.default_dtype)

        # Slack
        s = torch.zeros(
            self.num_data, self.num_data, device=self.device, dtype=self.default_dtype
        )  # (num_data,num_data)
        t = torch.zeros(
            self.num_data, self.num_data, device=self.device, dtype=self.default_dtype
        )  # (num_data,num_data)
        u = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )  # (num_data,dim_data)

        # Dual
        alpha = torch.zeros(
            self.num_data, self.num_data, device=self.device, dtype=self.default_dtype
        )  # [num_data,num_data]
        beta = torch.zeros(
            self.num_data, self.num_data, device=self.device, dtype=self.default_dtype
        )  # [num_data,num_data]
        gamma = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )  # [num_data,dim_data]
        eta = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )  # [num_data,dim_data]
        zeta = torch.zeros(
            self.num_data, self.dim_data, device=self.device, dtype=self.default_dtype
        )  # [num_data,dim_data]

        init_Sigma_i = torch.zeros(
            [self.num_data, self.dim_data, self.dim_data], device=self.device
        )  # [num_data, dim_data, dim_data]

        eye = torch.eye(self.dim_data, device=self.device)  # [dim_data,dim_data]

        # Preprocess1
        # =========================================
        Sigma_i, y_bar = calc_Sigma_i(init_Sigma_i, self.num_data, X_train, y_train, eye)

        # ADMM iteration (computational bottleneck)
        # ========================================
        best_score = -float("inf")
        y_hat_best = y_hat
        z_best = z
        a_best = a
        b_best = b
        for it_t in range(self.num_epochs):
            # Primal updates
            # y_hat & z update (y_hat for phi1 and z for phi2 where f = phi1 - phi2)
            y_hat, z = y_hat_z_update_dc(
                alpha, s, a, self.X_train, self.num_data, y_bar, beta, t, b, rho, self.y_train
            )

            # a update (a^q)
            a = a_update_dc(
                p, eta, alpha, s, y_hat, z, self.X_train, Sigma_i, self.num_data, self.dim_data
            )

            # b update (a^q)
            b = b_update_dc(
                q, zeta, beta, t, z, self.X_train, Sigma_i, self.num_data, self.dim_data
            )

            # p updates (p^q)
            p = p_update_dc(p, a, eta, L, u, gamma)

            # q updates (p^q)
            q = q_update_dc(b, zeta, L, u, gamma, q)

            # L update
            L = L_update_dc(self.num_data, rho, gamma, u, p, q, self.lambda_)

            # Slack updates
            # s & t update
            s = s_update_dc(alpha, y_hat, z, a, self.X_train)
            t = t_update_dc(beta, z, b, self.X_train)

            # u update
            u = u_update_dc(gamma, L, q, p)

            # Dual updates
            alpha, beta, gamma, eta, zeta = dual_update_dc(
                alpha, s, y_hat, z, a, self.X_train, beta, t, b, gamma, u, L, p, q, eta, zeta
            )

            # Save best parameters and verbose
            if (it_t + 1) % 500 == 0 or it_t == 0 or it_t + 1 == self.num_epochs:
                # Calc parameters
                y_hat_current = y_hat + z - torch.sum(a * self.X_train, dim=1)
                z_current = z - torch.sum(b * self.X_train, dim=1)
                a_current = a
                b_current = b

                # Update parameters
                self.y_hat = y_hat_current
                self.z = z_current
                self.a = a_current
                self.b = b_current

                # Calc score
                current_training_score = 1.0 - torch.mean(
                    (self.predict(self.X_train) - self.y_train) ** 2
                ) / torch.var(self.y_train)

                if best_score < current_training_score:
                    # Update the best score
                    best_score = current_training_score
                    _tmp_msg = " Best score updated!"

                    # Save the current best parameters
                    y_hat_best = y_hat_current
                    z_best = z_current
                    a_best = a_current
                    b_best = b_current

                else:
                    _tmp_msg = ""

                logger.info(
                    f"Step {it_t + 1} / {self.num_epochs}: Train score = {current_training_score},"
                    f"Best score = {best_score}{_tmp_msg}"
                )

        # Final result parameters
        self.y_hat = y_hat_best
        self.z = z_best
        self.a = a_best
        self.b = b_best

    @torch.no_grad()
    def predict(self, X: Tensor, *args, **kwargs) -> Tensor:
        """
        Outputs Eq. (16.5) in the original paper.

        # Args
        - X: A Tensor with shape [arbitrary number of data, dim_data].

        # Returns
        - pred: A Tensor with shape [X.shape[0],]
        """
        f1, _ = torch.max(self.y_hat.reshape(1, -1) + torch.matmul(X, self.a.T), dim=1)
        f2, _ = torch.max(self.z.reshape(1, -1) + torch.matmul(X, self.b.T), dim=1)
        pred = f1 - f2

        return pred
