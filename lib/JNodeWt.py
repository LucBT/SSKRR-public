"""Module for computing Jacobi quadrature points and weights."""

from typing import Tuple, List
import numpy as np
from scipy.special import gamma
from .JacobiGQ import JacobiGQ
from .JacobiGL import JacobiGL
from .JacobiGR import JacobiGR


def JNodeWt(
    NN: int,
    IP: int,
    alpha: float,
    beta: float
) -> List[np.ndarray]:
    """
    Compute interpolation points and weights from Jacobi polynomials.

    Args:
        NN: Order of interpolation (number of points - 1).
        IP: Quadrature type:
            1: Gauss quadrature (exact for orders up to 2*NN+1)
            2: Gauss-Lobatto quadrature (exact for orders up to 2*NN-1)
            3: Gauss-Radau quadrature with X_0 = -1 (exact for orders up to 2*NN)
            4: Gauss-Radau quadrature with X_N = +1 (exact for orders up to 2*NN)
            5: Equidistant points
        alpha: Jacobi polynomial parameter alpha.
        beta: Jacobi polynomial parameter beta.

    Returns:
        A list [Xinter, Winter] containing quadrature points and weights.
    """
    N1 = NN + 1
    N2 = 2.0 * NN

    # Chebyshev quadratures using exact formulae (alpha = beta = -0.5)
    if alpha == -0.5 and beta == -0.5:
        ij = np.arange(0, NN + 1)[:, np.newaxis]
        if IP == 1:  # Gauss quadrature
            x_inter = -np.cos((2 * ij + 1) * np.pi / (N2 + 2))
            w_inter = (np.pi / N1) * np.ones((N1, 1))
        elif IP == 2:  # Gauss-Lobatto quadrature
            x_inter = -np.cos(np.pi * ij / NN)
            w_inter = (np.pi / NN) * np.ones((N1, 1))
            w_inter[0] *= 0.5
            w_inter[-1] *= 0.5
        elif IP == 3:  # Gauss-Radau quadrature with X_0 = -1
            x_inter = -np.cos(2.0 * np.pi * ij / (N2 + 1))
            w_inter = (2.0 * np.pi / (N2 + 1)) * np.ones((N1, 1))
            w_inter[0] *= 0.5
        elif IP == 4:  # Gauss-Radau quadrature with X_N = +1
            x_inter = np.flipud(np.cos(2.0 * np.pi * ij / (N2 + 1)))
            w_inter = (2.0 * np.pi / (N2 + 1)) * np.ones((N1, 1))
            w_inter[-1] *= 0.5
        elif IP == 5:  # Equidistant
            x_inter = 2.0 * np.pi * ij / N1
            w_inter = (2.0 * np.pi / N1) * np.ones((N1, 1))
        else:
            raise ValueError(f"JNodeWt: option {IP} not implemented for Chebyshev case")

    else:
        # General Jacobi case using numerical methods
        if IP == 1:  # Gauss quadrature
            x_inter, w_inter = JacobiGQ(alpha, beta, NN)
        elif IP == 2:  # Gauss-Lobatto quadrature
            x_inter, w_inter = JacobiGL(alpha, beta, NN)
        elif IP == 3:  # Gauss-Radau quadrature with X_0 = -1
            x_inter, w_inter = JacobiGR(alpha, beta, NN, "L")
        elif IP == 4:  # Gauss-Radau quadrature with X_N = +1
            x_inter, w_inter = JacobiGR(alpha, beta, NN, "R")
        elif IP == 5:  # Equidistant
            x_inter = np.linspace(-1, 1, N1).reshape(-1, 1)
            w_inter = (2.0 / N1) * np.ones((N1, 1))
        else:
            raise ValueError(f"JNodeWt: option {IP} not implemented")

    # Zero-th moment of Gauss-Jacobi weight w(x) = (1-x)^alpha * (1+x)^beta
    mu0 = (
        2 ** (alpha + beta + 1)
        * gamma(alpha + 1)
        * gamma(beta + 1)
        / gamma(alpha + beta + 2)
    )

    # Sanity check for weights
    if not np.isclose(np.sum(w_inter), mu0, atol=1e-10):
        print(f"JNodeWt warning: sum of weights {np.sum(w_inter)} != mu0 {mu0}")

    return [x_inter, w_inter]
