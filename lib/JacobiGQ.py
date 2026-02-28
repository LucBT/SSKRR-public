"""Module for computing Gauss-Jacobi quadrature points and weights."""

from typing import Tuple

import numpy as np
from scipy.special import gamma


def JacobiGQ(alpha: float, beta: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate N-th order Gauss quadrature points x and weights w.

    Associated with Jacobi polynomials of type (alpha, beta) > -1.
    If N is the order, the number of points is N+1.

    Ref: Hesthaven, Warburton (2008), pp.447-448.

    Args:
        alpha: Jacobi polynomial parameter alpha.
        beta: Jacobi polynomial parameter beta.
        N: Order of the quadrature.

    Returns:
        A tuple (x, w) where x are the quadrature points and w are the weights.
    """
    if N == 0:
        x = np.zeros((1, 1))
        w = np.zeros((1, 1))
        x[0, 0] = (beta - alpha) / (alpha + beta + 2)
        w[0, 0] = (
            2 ** (alpha + beta + 1)
            * gamma(alpha + 1)
            * gamma(beta + 1)
            / gamma(alpha + beta + 2)
        )
    else:
        # Form symmetric Jacobi matrix for recurrence
        h1 = 2 * np.arange(0, N + 1) + alpha + beta
        j_diag = np.zeros(N + 1)
        j_diag[0] = -0.5 * (alpha - beta) / (h1[0] + 2)
        j_diag[1:] = -0.5 * (alpha**2 - beta**2) / (h1[1 : (N + 1)] + 2) / h1[1 : (N + 1)]

        # Off-diagonal elements
        indices = np.arange(1, N + 1)
        j_off_diag = (
            2
            / (h1[0:N] + 2)
            * np.sqrt(
                indices
                * (indices + alpha + beta)
                * (indices + alpha)
                * (indices + beta)
                / (h1[0:N] + 1)
                / (h1[0:N] + 3)
            )
        )

        # Construct symmetric matrix J
        J = np.diag(j_diag) + np.diag(j_off_diag, 1) + np.diag(j_off_diag, -1)

        # Compute quadrature by eigenvalue decomposition
        # D contains the eigenvalues (points), V contains eigenvectors
        D, V = np.linalg.eigh(J)
        x = np.array([D]).T

        # Calculate weights from the first component of each eigenvector
        # normalized by the zero-th moment.
        # w_i = V[0, i]^2 * mu_0
        mu0 = (
            2 ** (alpha + beta + 1)
            / (alpha + beta + 1)
            * gamma(alpha + 1)
            * gamma(beta + 1)
            / gamma(alpha + beta + 1)
        )
        w = (V[0, :]) ** 2 * mu0
        w = np.array([w]).T

    return x, w
