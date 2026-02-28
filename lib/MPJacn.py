"""Multivariate Jacobi polynomials evaluation."""

from typing import Optional, Union
import numpy as np
from scipy.special import comb
from .JacobiP import JacobiP


def MPJacn(
    x: Union[np.ndarray, list],
    NN: int,
    alpha: Union[np.ndarray, list, float],
    beta: Union[np.ndarray, list, float],
    Mindices: np.ndarray
) -> Optional[np.ndarray]:
    """
    Evaluate multivariate Jacobi polynomials at point x for order NN.

    The multivariate polynomial is constructed as a product of 1D Jacobi
    polynomials according to the provided multi-indices.

    Notes:
    - Polynomials are normalized to be orthonormal.
    - Bug fix (ES 12/10/2009): Case alpha = beta = -0.5 for N = 0 is handled in JacobiP.

    Args:
        x: A multi-dimensional point (array-like of size nx).
        NN: Maximum total order of the polynomials.
        alpha: Jacobi polynomial parameters alpha (array-like of size nx).
        beta: Jacobi polynomial parameters beta (array-like of size nx).
        Mindices: Multi-indices matrix of size (P1, nx) where P1 is the number
                  of multivariate basis functions.

    Returns:
        XP: Array of size (1, P1) containing the evaluation of multivariate
            polynomials at point x, or None if input sizes are incorrect.
    """
    # Ensure inputs are numpy arrays
    x = np.atleast_1d(x)
    alpha = np.atleast_1d(alpha)
    beta = np.atleast_1d(beta)

    nx = x.size
    na = alpha.size
    nb = beta.size

    # Check if alpha and beta match the dimension of x
    if na != nx or nb != nx:
        print("MPJacn: the size of alpha/beta is incorrect")
        return None

    N1 = NN + 1
    # PJ stores 1D polynomials: PJ[order, dimension]
    PJ = np.zeros((N1, nx))

    # Pre-calculate 1D Jacobi polynomials for each dimension up to order NN
    for ix in range(nx):
        for iN in range(N1):
            # JacobiP returns the value of the polynomial of degree iN at point x[ix]
            # as a 1D array of size 1, so we take the first element.
            PJ[iN, ix] = JacobiP(x[ix], alpha[ix], beta[ix], iN)[0]

    # Number of multi-indices for total order NN in dimension nx
    P1 = int(comb(NN + nx, nx))
    XP = np.zeros(P1)
    Mindices = np.array(Mindices)

    # Temp array for 1D values to be multiplied
    QJ = np.zeros(nx)

    for iP in range(P1):
        for ix in range(nx):
            # Get the degree for the current dimension from multi-indices
            order = int(Mindices[iP, ix])
            QJ[ix] = PJ[order, ix]

        # Multivariate polynomial is the product of 1D polynomials
        XP[iP] = np.prod(QJ)

    # Return as a row vector to maintain compatibility
    return XP.reshape(1, -1)
