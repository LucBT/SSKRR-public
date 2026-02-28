"""1D Jacobi polynomials evaluation using recurrence relation."""

from typing import Union
import numpy as np
from scipy.special import gamma


def JacobiP(
    x: Union[np.ndarray, list, float],
    alpha: float,
    beta: float,
    N: int
) -> np.ndarray:
    """
    Evaluate Jacobi polynomial of type (alpha, beta) > -1 at points x for order N.

    Polynomials are normalized to be orthonormal.
    Uses forward recurrence relation.

    Notes:
    - Bug fix (ES 12/10/2009): Case alpha = beta = -0.5 for N = 0 is handled.
    - Reference: Savin, Faverjon (2017) "Computation of higher-order moments
      of generalized polynomial chaos expansions".

    Args:
        x: Points at which to evaluate the polynomial (array-like or scalar).
        alpha: Jacobi polynomial parameter alpha > -1.
        beta: Jacobi polynomial parameter beta > -1.
        N: Order of the polynomial to evaluate.

    Returns:
        P: Values of the polynomial of order N at points x.
           Returned as a 1D array.
    """
    # Ensure x is an array for consistent handling
    xp = np.atleast_1d(x)
    num_obs = xp.size

    # Array to store polynomials up to degree N: pl[order, observation]
    pl = np.zeros((N + 1, num_obs))

    # Initial value for order 0: P_0(x)
    if alpha == -0.5 and beta == -0.5:
        # Special case for Chebyshev polynomials
        gamma0 = np.pi
    else:
        # Standard Jacobi normalization factor for order 0
        gamma0 = (
            2 ** (alpha + beta + 1)
            / (alpha + beta + 1)
            * gamma(alpha + 1)
            * gamma(beta + 1)
            / gamma(alpha + beta + 1)
        )

    pl[0, :] = 1.0 / np.sqrt(gamma0)

    if N == 0:
        return pl[0, :]

    # Initial value for order 1: P_1(x)
    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    pl[1, :] = ((alpha + beta + 2) * xp / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)

    if N == 1:
        return pl[1, :]

    # Recurrence relation constants
    aold = 2 / (alpha + beta + 2) * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))

    # Forward recurrence to compute P_i(x) for i from 2 up to N
    for i in range(1, N):
        h1 = 2 * i + alpha + beta
        anew = (
            2
            / (h1 + 2)
            * np.sqrt(
                (i + 1)
                * (i + 1 + alpha + beta)
                * (i + 1 + alpha)
                * (i + 1 + beta)
                / (h1 + 1)
                / (h1 + 3)
            )
        )
        bnew = -(alpha**2 - beta**2) / (h1 * (h1 + 2))

        # Recurrence step
        pl[i + 1, :] = (1 / anew) * (-aold * pl[i - 1, :] + (xp - bnew) * pl[i, :])
        aold = anew

    # Return the values for order N
    return pl[N, :]
