"""
Main script for the SSKRR (Sparse Spectral Kernel Ridge Regression) method.
This script converts the main_SSKRR.ipynb notebook into a modular command-line tool.
"""

import argparse
import logging
import yaml
import numpy as np
import torch
import pyro
import pyro.contrib.gp as gp
import matplotlib.pyplot as plt
from matplotlib import rc
import sklearn.preprocessing
from scipy import stats
from scipy.special import eval_legendre, factorial
from smt.sampling_methods import LHS as LHS_sampling
import spgl1
from typing import Tuple, List, Optional, Any, Dict, Union

from lib import MultiIndex, MPJacn, JNodeWt

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_matplotlib(use_latex: bool = True) -> None:
    """
    Configure matplotlib to use LaTeX.

    Args:
        use_latex: If True, attempts to use LaTeX for text rendering.
    """
    if use_latex:
        try:
            rc("text", usetex=True)
            plt.rc("axes", labelsize=25)
            plt.rcParams["text.usetex"] = True
            plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"
            plt.rcParams["font.family"] = "Latin Modern Roman"
            logger.info("Matplotlib configured with LaTeX.")
        except Exception as e:
            logger.warning(f"Could not configure LaTeX for matplotlib: {e}. Using plain text.")
            plt.rcParams["text.usetex"] = False
    else:
        plt.rcParams["text.usetex"] = False
        logger.info("Matplotlib configured without LaTeX.")


def ishigami_function(x: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    """
    Compute the Ishigami function.

    Args:
        x: Input points of dimension (N, 3).
        a: Parameter a of the Ishigami function.
        b: Parameter b of the Ishigami function.

    Returns:
        Function values at points x.
    """
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    return np.sin(x1) + a * (np.sin(x2)) ** 2 + b * (x3**4) * np.sin(x1)


def rosenbrock_function(x: np.ndarray) -> np.ndarray:
    """
    Compute the Rosenbrock function for any dimension.

    Args:
        x: Input points of dimension (N, D).

    Returns:
        Function values at points x.
    """
    dimension = x.shape[1]
    rosenbrock = 0.0
    for d in range(dimension - 1):
        rosenbrock += 100 * (x[:, d + 1] - x[:, d] ** 2) ** 2 + (1 - x[:, d]) ** 2
    return rosenbrock


def legendre_poly(x: np.ndarray, order: int) -> np.ndarray:
    """
    Compute the normalized Legendre polynomial at a given order.

    Args:
        x: Evaluation points.
        order: Polynomial order.

    Returns:
        Polynomial values at points x.
    """
    norm = np.sqrt(2 / (2 * order + 1))
    return eval_legendre(order, x) / norm


def prod_legendre_poly(x: np.ndarray, indx: np.ndarray) -> np.ndarray:
    """
    Compute the product of Legendre polynomials according to provided indices.

    Args:
        x: Evaluation points (N, D).
        indx: Order indices for each dimension (D,).

    Returns:
        Product of polynomials (N, 1).
    """
    n_ld, dimension = x.shape
    prod = np.ones((n_ld, 1))
    for dim in range(dimension):
        poly = legendre_poly(x[:, dim], int(indx[dim]))
        prod *= poly.reshape((-1, 1))
    return prod


def compute_rho_torch_pyro(
    xobs: torch.Tensor,
    f_vals: torch.Tensor,
    pif: np.ndarray,
    pic: np.ndarray,
    nugget: torch.Tensor,
    kernel: Any,
) -> torch.Tensor:
    """
    Compute the rho metric for GPR optimization.

    Args:
        xobs: Observation points.
        f_vals: Function values at observation points.
        pif: Mask for 'fine' points.
        pic: Mask for 'coarse' points.
        nugget: Regularization parameter.
        kernel: Gaussian process kernel.

    Returns:
        Rho value.
    """
    # Xf points
    kf = kernel.forward(xobs[pif, :]) + nugget * torch.eye(sum(pif))
    ff = f_vals[pif]

    # Xc points
    kc = kernel.forward(xobs[pic, :]) + nugget * torch.eye(sum(pic))
    fc = f_vals[pic]

    # Compute rho
    den = fc @ torch.linalg.solve(kc, fc)
    num = ff @ torch.linalg.solve(kf, ff)
    rho = 1 - den / num

    if rho < 0 or rho > 1:
        raise ValueError(f"Error in rho: {rho.item()}. Value must be between 0 and 1.")

    return rho


def compute_rms_from_data(x: np.ndarray) -> float:
    """
    Compute the root mean square distance from a set of positions.

    Args:
        x: Set of observations (N, D) or (N,).

    Returns:
        RMS value.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n_obs, dimension = x.shape
    squared_distance = []

    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            dist = np.sum((x[i] - x[j]) ** 2)
            squared_distance.append(dist)

    rms = np.sqrt((2 / (n_obs * (n_obs - 1))) * np.sum(squared_distance))
    logger.debug(f"RMS computed from {n_obs} observations of dimension {dimension}: {rms}")
    return rms


def create_pif_pic(
    num_obs: int, nf: int, nc: int, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create Pif and Pic masks for multi-fidelity learning.

    Args:
        num_obs: Total number of observations.
        nf: Number of 'fine' points.
        nc: Number of 'coarse' points.
        seed: Random seed.

    Returns:
        Masks (Pif, Pic).
    """
    rng = np.random.default_rng(seed)

    # Pif points
    pif = np.full(num_obs, False)
    pif[:nf] = True
    rng.shuffle(pif)

    # Pic points
    pic = np.full(num_obs, False)
    fine_indices = np.where(pif == True)[0]
    pic[rng.choice(fine_indices, size=nc, replace=False)] = True

    return pif, pic


def compute_pce_norm_moment_order(alpha: List[int], beta: List[int], dimension: int) -> float:
    """
    Compute the norm for the PCE moment order.

    Args:
        alpha: Jacobi polynomial alpha parameters.
        beta: Jacobi polynomial beta parameters.
        dimension: Problem dimension.

    Returns:
        Computed norm.
    """
    norm = 1.0
    for dim in range(dimension):
        num = (2 ** (alpha[dim] + beta[dim] + 1)) * factorial(alpha[dim]) * factorial(beta[dim])
        den = factorial(alpha[dim] + beta[dim] + 1)
        norm *= num / den
    return np.sqrt(norm)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the config.yaml file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_sskrr_and_pce(
    cfg: Dict[str, Any],
    case_cfg: Dict[str, Any],
    func: Any,
    scaler: sklearn.preprocessing.MinMaxScaler,
    bounds: np.ndarray,
    dimension: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Execute SSKRR and Sparse PCE algorithms.

    Returns:
        (y_sskrr_list, y_pce_list, y_td_list)
    """
    num_seeds = cfg["general"]["number_of_seeds"]
    n_td = cfg["general"]["n_td"]
    nuggets_cfg = cfg["sskrr"]
    nuggets = 10 ** (np.linspace(nuggets_cfg["nuggets_start"], nuggets_cfg["nuggets_end"], num=nuggets_cfg["nuggets_num"]))

    total_order = case_cfg["total_order"]
    alpha = [0] * dimension
    beta = [0] * dimension
    indexes, _ = MultiIndex.MultiIndex(total_order, dimension)

    y_sskrr_td_list = []
    y_pce_td_list = []
    y_td_list = []

    for n_obs in case_cfg["number_of_observations_ld_list"]:
        for seed in range(num_seeds):
            logger.info(f"SSKRR/PCE - Seed {seed}, N_obs {n_obs}")

            sampling = LHS_sampling(xlimits=bounds, criterion="maximin", random_state=seed)
            x_ld = sampling(n_obs)
            x_ld_scaled = scaler.transform(x_ld)
            y_ld = func(x_ld)

            kappa_var = np.var(y_ld)

            # SPGL1
            theta = np.zeros((n_obs, len(indexes)))
            for obs in range(n_obs):
                theta[obs, :] = MPJacn.MPJacn(x_ld_scaled[obs, :], total_order, alpha, beta, indexes)

            c, _, _, _ = spgl1.spg_bpdn(
                theta,
                y_ld.flatten(),
                case_cfg["spgl1"]["sigma"],
                verbosity=0,
                iter_lim=int(1e5),
                opt_tol=case_cfg["spgl1"]["opt_tol"],
            )

            ck_value = c
            r_s = len(c)
            e_sparse = theta

            # SSKRR eigenvalues
            den = np.sum(np.abs(ck_value))
            sigma_min = np.array([kappa_var * (np.abs(ck) / den) for ck in ck_value]).flatten()

            # Test dataset
            np.random.seed(seed)
            x_td_scaled = np.random.uniform(-1, 1, size=(n_td, dimension))
            x_td = scaler.inverse_transform(x_td_scaled)
            y_td = func(x_td)
            y_td_list.append(y_td)

            # PCE Prediction
            e_pce = np.zeros((n_td, r_s))
            for idx in range(r_s):
                e_pce[:, idx] = prod_legendre_poly(x_td_scaled, indexes[idx]).flatten()
            y_pce = e_pce @ np.array(ck_value)
            y_pce_td_list.append(y_pce)

            # SSKRR Optimization (Nugget Grid Search)
            rmse_optm_list = []
            y_best_predict_optm = None
            min_rmse = float("inf")

            # Pre-calculate constant parts of K_OBS and K_PREDICT
            sigma_diag = np.diag(sigma_min)
            k_obs_base = e_sparse @ sigma_diag @ e_sparse.T
            k_predict_base = e_pce @ sigma_diag @ e_sparse.T

            for nugget in nuggets:
                k_obs = k_obs_base + nugget * np.eye(n_obs)
                y_predict = k_predict_base @ np.linalg.solve(k_obs, y_ld)
                rmse = np.sqrt(np.mean((y_td - y_predict) ** 2))
                rmse_optm_list.append(rmse)

                if rmse < min_rmse:
                    min_rmse = rmse
                    y_best_predict_optm = y_predict

            y_sskrr_td_list.append(y_best_predict_optm)
            logger.info(f"Log10(RMSE) SSKRR: {np.log10(min_rmse):.4f}, PCE: {np.log10(np.sqrt(np.mean((y_pce - y_td)**2))):.4f}")

    return y_sskrr_td_list, y_pce_td_list, y_td_list


def run_gpr(
    cfg: Dict[str, Any],
    case_cfg: Dict[str, Any],
    func: Any,
    scaler: sklearn.preprocessing.MinMaxScaler,
    bounds: np.ndarray,
    dimension: int,
) -> List[np.ndarray]:
    """
    Execute Gaussian Process Regression (GPR) with Pyro.

    Returns:
        y_gpr_list
    """
    num_seeds = cfg["general"]["number_of_seeds"]
    n_td = cfg["general"]["n_td"]
    ite_max = case_cfg["gpr"]["ite_max"]
    ite_save = case_cfg["gpr"]["ite_save"]

    y_gpr_list = []

    for n_obs in case_cfg["number_of_observations_ld_list"]:
        for seed in range(num_seeds):
            logger.info(f"GPR - Seed {seed}, N_obs {n_obs}")

            sampling = LHS_sampling(xlimits=bounds, criterion="maximin", random_state=seed)
            x_ld = sampling(n_obs)
            x_ld_scaled = scaler.transform(x_ld)
            y_ld = func(x_ld)

            # Test dataset
            np.random.seed(seed)
            x_td_scaled = np.random.uniform(-1, 1, size=(n_td, dimension))
            y_td = func(scaler.inverse_transform(x_td_scaled))

            # Initial lengthscale
            gamma = [compute_rms_from_data(x_ld_scaled[:, d]) / np.sqrt(2) for d in range(dimension)]

            nf, nc = n_obs, n_obs // 2
            pif, pic = create_pif_pic(n_obs, nf, nc, seed=None)

            variable_gamma = torch.tensor(gamma, requires_grad=True)
            nugget_optm = torch.tensor([0.0], requires_grad=True)

            kernel = gp.kernels.RBF(
                input_dim=dimension,
                variance=torch.tensor(1.0),
                lengthscale=variable_gamma
            )

            # Pyro parameter store cleanup
            pyro.clear_param_store()

            optimizer = torch.optim.Adam(
                [kernel.lengthscale_unconstrained, nugget_optm],
                lr=3e-4
            )

            variable_gamma_optm_history = [variable_gamma.detach().numpy().copy()]
            rmse_history = []

            # Initial RMSE
            with torch.no_grad():
                gpr_tmp = gp.models.GPRegression(
                    torch.from_numpy(x_ld_scaled).float(),
                    torch.from_numpy(y_ld).float(),
                    kernel,
                    noise=torch.tensor(0.0),
                    jitter=nugget_optm
                )
                val_torch, _ = gpr_tmp(torch.from_numpy(x_td_scaled).float(), full_cov=False, noiseless=True)
                rmse_history.append(np.sqrt(np.mean((y_td - val_torch.numpy()) ** 2)))

            for k in range(ite_max):
                optimizer.zero_grad()
                loss = compute_rho_torch_pyro(
                    torch.from_numpy(x_ld_scaled).float(),
                    torch.from_numpy(y_ld).float(),
                    pif, pic, nugget_optm, kernel
                )
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    kernel.lengthscale[:] = kernel.lengthscale.clamp(1e-6, 1e6)
                    nugget_optm[:] = nugget_optm.clamp(1e-12, 1e6)

                    if (k + 1) % ite_save == 0:
                        gpr_tmp = gp.models.GPRegression(
                            torch.from_numpy(x_ld_scaled).float(),
                            torch.from_numpy(y_ld).float(),
                            kernel,
                            noise=torch.tensor(0.0),
                            jitter=nugget_optm
                        )
                        val_torch, _ = gpr_tmp(torch.from_numpy(x_td_scaled).float(), full_cov=False, noiseless=True)
                        current_rmse = np.sqrt(np.mean((y_td - val_torch.numpy()) ** 2))
                        rmse_history.append(current_rmse)
                        variable_gamma_optm_history.append(kernel.lengthscale.detach().numpy().copy())
                        logger.info(f"GPR Iter {k+1}/{ite_max} - Log10(RMSE): {np.log10(current_rmse):.4f}")

                pif, pic = create_pif_pic(n_obs, nf, nc, seed=None)

            # Best hyperparameters based on history
            idx_best = np.argmin(rmse_history)
            best_gamma = variable_gamma_optm_history[idx_best]

            kernel_best = gp.kernels.RBF(
                input_dim=dimension,
                variance=torch.tensor(1.0),
                lengthscale=torch.from_numpy(best_gamma)
            )
            gpr_best = gp.models.GPRegression(
                torch.from_numpy(x_ld_scaled).float(),
                torch.from_numpy(y_ld).float(),
                kernel_best,
                noise=torch.tensor(0.0),
                jitter=nugget_optm
            )
            val_best, _ = gpr_best(torch.from_numpy(x_td_scaled).float(), full_cov=False, noiseless=True)
            y_gpr_list.append(val_best.detach().numpy())

    return y_gpr_list


def run_fully_tensorized_pce(
    cfg: Dict[str, Any],
    case_cfg: Dict[str, Any],
    func: Any,
    scaler: sklearn.preprocessing.MinMaxScaler,
    dimension: int,
) -> List[np.ndarray]:
    """
    Execute fully tensorized PCE (Ishigami only).
    """
    num_seeds = cfg["general"]["number_of_seeds"]
    n_td = cfg["general"]["n_td"]
    total_order = case_cfg["total_order"]

    # Quadrature parameters
    n_nodes = int(np.around((total_order * 2 + 3) / 2))
    choice_quad = 2  # GJL quadrature
    quad, weights = JNodeWt.JNodeWt(n_nodes - 1, choice_quad, 0, 0)

    n_ld = n_nodes**dimension
    x_ld_scaled = np.zeros((n_ld, dimension))

    # Grid generation (specific to 3D for Ishigami)
    if dimension != 3:
        logger.warning("run_fully_tensorized_pce is optimized for 3D.")

    count = 0
    multi_dim_weights = np.zeros((n_ld, dimension))
    for i in range(n_nodes):
        for j in range(n_nodes):
            for k in range(n_nodes):
                x_ld_scaled[count, :] = quad[i], quad[j], quad[k]
                multi_dim_weights[count, :] = weights[i], weights[j], weights[k]
                count += 1

    prod_weights = np.prod(multi_dim_weights, axis=1)
    diag_weights = np.diag(prod_weights)

    x_ld = scaler.inverse_transform(x_ld_scaled)
    y_ld = func(x_ld)

    indexes, _ = MultiIndex.MultiIndex(total_order, dimension)
    alpha = [0] * dimension
    beta = [0] * dimension

    theta = np.zeros((n_ld, len(indexes)))
    for obs in range(n_ld):
        theta[obs, :] = MPJacn.MPJacn(x_ld_scaled[obs, :], total_order, alpha, beta, indexes)

    # Expansion coefficients
    c = y_ld.T @ diag_weights @ theta
    ck_value = c
    r_s = len(c)

    y_ft_pce_list = []
    for seed in range(num_seeds):
        np.random.seed(seed)
        x_td_scaled = np.random.uniform(-1, 1, size=(n_td, dimension))
        y_td = func(scaler.inverse_transform(x_td_scaled))

        e_pce = np.zeros((n_td, r_s))
        for idx in range(r_s):
            e_pce[:, idx] = prod_legendre_poly(x_td_scaled, indexes[idx]).flatten()

        y_pce = e_pce @ np.array(ck_value)
        y_ft_pce_list.append(y_pce)
        logger.info(f"FT-PCE Seed {seed} - Log10(RMSE): {np.log10(np.sqrt(np.mean((y_pce - y_td)**2))):.4f}")

    return y_ft_pce_list


def plot_boxplot(
    y_td_list: List[np.ndarray],
    results: Dict[str, List[np.ndarray]],
    title: str
) -> None:
    """
    Generate and display error boxplots.
    """
    nrmse_results = {}
    rmse_results = {}

    for name, predictions in results.items():
        nrmse_list = []
        rmse_list = []
        for i in range(len(y_td_list)):
            y_td = y_td_list[i].flatten()
            y_pred = predictions[i].flatten()
            nrmse = np.sqrt(np.sum((y_td - y_pred) ** 2) / np.sum(y_td**2))
            rmse = np.sqrt(np.mean((y_td - y_pred) ** 2))
            nrmse_list.append(np.log10(nrmse))
            rmse_list.append(np.log10(rmse))

        nrmse_results[name] = nrmse_list
        rmse_results[name] = rmse_list

    labels = list(nrmse_results.keys())
    data_nrmse = [nrmse_results[lab] for lab in labels]
    data_rmse = [rmse_results[lab] for lab in labels]

    plt.figure()
    plt.boxplot(data_nrmse)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.grid(True, linestyle="-.", linewidth=0.5)
    plt.ylabel(r"$\log_{10}(e_{\mathrm{NRMSE}})$")
    plt.title(f"NRMSE - {title}")
    plt.show()

    plt.figure()
    plt.boxplot(data_rmse)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.grid(True, linestyle="-.", linewidth=0.5)
    plt.ylabel(r"$\log_{10}(e_{\mathrm{RMSE}})$")
    plt.title(f"RMSE - {title}")
    plt.show()


def plot_pdf(
    y_td: np.ndarray,
    results: Dict[str, List[np.ndarray]],
    title: str
) -> None:
    """
    Generate and display Probability Density Function (PDF) plots.
    """
    y_td = y_td.flatten()
    test_gt = np.linspace(np.min(y_td), np.max(y_td), num=1000)

    plt.figure(figsize=(10, 6))
    pdf_gt = stats.gaussian_kde(y_td)(test_gt)
    plt.plot(test_gt, pdf_gt, "k", linestyle="dotted", label="Ground truth")

    colors = ["r", "b", "c", "m", "g"]
    markers = ["D", "*", "^", "s", "o"]

    for i, (name, predictions) in enumerate(results.items()):
        y_pred = predictions[0].flatten()  # Use first seed
        pdf_pred = stats.gaussian_kde(y_pred)(test_gt)
        plt.plot(
            test_gt, pdf_pred,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linestyle="-.",
            linewidth=0.5,
            markevery=50,
            markersize=4,
            label=name
        )

    plt.grid(True, linestyle="-.", linewidth=0.2)
    plt.legend()
    plt.title(f"PDF - {title}")
    plt.show()


def run_ishigami(cfg: Dict[str, Any]) -> None:
    """
    Execute the Ishigami test case.
    """
    logger.info("Starting Ishigami test case")
    case_cfg = cfg["ishigami"]
    bounds = np.array(case_cfg["bounds"])
    scaler = sklearn.preprocessing.MinMaxScaler((-1, 1))
    scaler.fit(bounds.T)

    y_sskrr, y_pce, y_td = run_sskrr_and_pce(
        cfg, case_cfg, ishigami_function, scaler, bounds, 3
    )

    y_gpr = run_gpr(
        cfg, case_cfg, ishigami_function, scaler, bounds, 3
    )

    y_ft_pce = run_fully_tensorized_pce(
        cfg, case_cfg, ishigami_function, scaler, 3
    )

    results = {
        "SSKRR": y_sskrr,
        "Sparse gPC": y_pce,
        "Fully tensorized gPC": y_ft_pce,
        "KRR": y_gpr
    }

    plot_boxplot(y_td, results, "Ishigami")
    plot_pdf(y_td[0], results, "Ishigami")


def run_rosenbrock(cfg: Dict[str, Any]) -> None:
    """
    Execute the Rosenbrock test case.
    """
    logger.info("Starting Rosenbrock test case")
    case_cfg = cfg["rosenbrock"]
    dim = case_cfg["dimension"]
    bounds = np.zeros((dim, 2))
    bounds[:, 0], bounds[:, 1] = case_cfg["bounds_min"], case_cfg["bounds_max"]
    scaler = sklearn.preprocessing.MinMaxScaler((-1, 1))
    scaler.fit(bounds.T)

    y_sskrr, y_pce, y_td = run_sskrr_and_pce(
        cfg, case_cfg, rosenbrock_function, scaler, bounds, dim
    )

    y_gpr = run_gpr(
        cfg, case_cfg, rosenbrock_function, scaler, bounds, dim
    )

    results = {
        "SSKRR": y_sskrr,
        "Sparse gPC": y_pce,
        "KRR": y_gpr
    }

    plot_boxplot(y_td, results, "Rosenbrock")
    plot_pdf(y_td[0], results, "Rosenbrock")


def main() -> None:
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(description="SSKRR Method Analysis")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--test-case", type=str, choices=["ishigami", "rosenbrock", "both"], default="both", help="Test case to run")
    parser.add_argument("--no-latex", action="store_true", help="Disable LaTeX rendering")

    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_matplotlib(use_latex=not args.no_latex and cfg["general"]["use_latex"])

    if args.test_case in ["ishigami", "both"]:
        run_ishigami(cfg)

    if args.test_case in ["rosenbrock", "both"]:
        run_rosenbrock(cfg)


if __name__ == "__main__":
    main()