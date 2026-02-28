"""Module for computing Gauss-Lobatto-Jacobi quadrature points and weights."""

from typing import Tuple

import numpy as np
from scipy.special import gamma

from .JacobiGQ import JacobiGQ


def JacobiGL(alpha: float, beta: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate N-th order Gauss-Lobatto quadrature points x and weights w.

    Associated with Jacobi polynomials of type (alpha, beta) > -1.
    If N is the order, the number of points is N+1.

    Ref: Hesthaven, Gottlieb, Gottlieb (2007) and Gozman (2005).

    Args:
        alpha: Jacobi polynomial parameter alpha.
        beta: Jacobi polynomial parameter beta.
        N: Order of the quadrature.

    Returns:
        A tuple (x, w) where x are the quadrature points and w are the weights.
    """
    x = np.zeros((N + 1, 1))
    w = np.zeros((N + 1, 1))

    # Weight factors calculation Ref: Eq.(5.15) Hesthaven et al (2007)
    # and Eqs.(1.10), (2.9-10) Gozman (2005)
    b0 = (
        2 ** (alpha + beta + 1)
        * np.prod(np.arange(1, N))
        / gamma(alpha + beta + N + 2)
    )
    b1 = (
        b0
        * gamma(beta + 1)
        * gamma(beta + 2)
        * gamma(alpha + N + 1)
        / gamma(beta + N + 1)
    )
    b2 = (
        b0
        * gamma(alpha + 1)
        * gamma(alpha + 2)
        * gamma(beta + N + 1)
        / gamma(alpha + N + 1)
    )

    if N == 1:
        x[0, 0] = -1.0
        x[1, 0] = 1.0
        w[0, 0] = b1
        w[1, 0] = b2
        return x, w

    # Interior points are roots of the derivative of the Jacobi polynomial
    # which correspond to Gauss quadrature points for parameters (alpha+1, beta+1)
    xint, wint = JacobiGQ(alpha + 1, beta + 1, N - 2)

    # Combine endpoints with interior points
    x = np.block([[-1.0], [xint], [1.0]])
    w = np.block([[b1], [wint / (1.0 - xint**2)], [b2]])

    return x, w
