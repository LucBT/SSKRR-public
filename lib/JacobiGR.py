"""Module for computing Gauss-Radau-Jacobi quadrature points and weights."""

from typing import Tuple

import numpy as np
from scipy.special import gamma

from .JacobiGQ import JacobiGQ


def JacobiGR(
    alpha: float, beta: float, N: int, IR: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate N-th order Gauss-Radau quadrature points x and weights w.

    Associated with Jacobi polynomials of type (alpha, beta) > -1.
    If N is the order, the number of points is N+1.

    Ref: F.L. Gozman (2005).

    Args:
        alpha: Jacobi polynomial parameter alpha.
        beta: Jacobi polynomial parameter beta.
        N: Order of the quadrature.
        IR: Side to include the fixed point ('L' for -1, 'R' for +1).

    Returns:
        A tuple (x, w) where x are the quadrature points and w are the weights.
    """
    # Constant B0 used in weight factor calculations Ref: Gozman (2005)
    b0 = (
        2 ** (alpha + beta + 1)
        * np.prod(np.arange(1, N + 1))
        / gamma(alpha + beta + N + 2)
    )
    b1 = (
        b0
        * gamma(beta + 1)
        * gamma(beta + 2)
        * gamma(alpha + N + 1)
        / gamma(beta + N + 2)
    )
    b2 = (
        b0
        * gamma(alpha + 1)
        * gamma(alpha + 2)
        * gamma(beta + N + 1)
        / gamma(alpha + N + 2)
    )

    if IR == "L":
        # Fixed point at -1. Interior points are Gauss quadrature points
        # for parameters (alpha, beta+1).
        xint, wint = JacobiGQ(alpha, beta + 1, N - 1)
        x = np.block([[-1.0], [xint]])
        w = np.block([[b1], [wint / (1.0 + xint)]])
    elif IR == "R":
        # Fixed point at +1. Interior points are Gauss quadrature points
        # for parameters (alpha+1, beta).
        xint, wint = JacobiGQ(alpha + 1, beta, N - 1)
        x = np.block([[xint], [1.0]])
        w = np.block([[wint / (1.0 - xint)], [b2]])
    else:
        raise ValueError("IR must be 'L' or 'R'")

    return x, w
