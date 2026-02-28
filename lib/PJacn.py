"""Jacobi polynomials and their derivatives evaluation."""

from typing import Tuple
import numpy as np
from .JacobiP import JacobiP


def PJacn(
    x: np.ndarray,
    NN: int,
    alpha: float,
    beta: float,
    filter_user: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate Jacobi polynomials of type (alpha, beta) > -1 and their derivatives.

    The computation is performed at points x up to the order NN.
    A filter is applied to compute filtered Jacobi polynomials.

    Notes:
    - They are normalized to be orthonormal.
    - The computation of Q = dP/dx uses Hesthaven, Warburton:
      Nodal Discontinuous Galerkin Methods,
      Springer, New York (2008), Eq.(A.2) p.445.

    Args:
        x: Array of points to evaluate at.
        NN: Maximum order of the polynomials.
        alpha: Jacobi polynomial parameter alpha > -1.
        beta: Jacobi polynomial parameter beta > -1.
        filter_user: Filter coefficients to apply to each order.

    Returns:
        A tuple (P, Q) where:
        - P is a (NN+1, nx) array of filtered Jacobi polynomials.
        - Q is a (NN+1, nx) array of their derivatives.
    """
    if np.shape(filter_user)[0] == 1:
        filter_user = np.transpose(filter_user)

    nx = int(len(x))
    N1 = NN + 1
    P = np.zeros((N1, nx))
    Q = np.zeros((N1, nx))

    # Evaluate the zero-th order polynomial
    # JacobiP returns values at points x for order 0
    P[0, 0:nx] = filter_user[0] * JacobiP(x, alpha, beta, 0)

    for iN in range(2, N1 + 1):
        # Normalized polynomials
        norm = np.sqrt((iN + alpha + beta) * (iN - 1))
        # Non-normalized polynomials: norm = (iN + alpha + beta) / 2

        # P[iN-1, :] is the polynomial of order iN-1
        P[iN - 1, 0:nx] = filter_user[iN - 1] * JacobiP(x, alpha, beta, iN - 1)

        # Q[iN-1, :] is the derivative of the polynomial of order iN-1
        Q[iN - 1, 0:nx] = norm * (
            filter_user[iN - 1] * JacobiP(x, alpha + 1, beta + 1, iN - 2)
        )

    return P, Q
