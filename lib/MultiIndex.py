"""Multi-indices of the NN-dimensional polynomials of total order No."""

from typing import Tuple
import numpy as np
from scipy.special import comb


def MultiIndex(No: int, NN: int) -> Tuple[np.ndarray, int]:
    """
    Compute multi-indices of the NN-dimensional polynomials of total order No.

    Args:
        No: Max total order of the multivariate polynomials.
        NN: Dimension of the problem.

    Returns:
        alpha: Multi-indices matrix of size (P+1, NN).
        P: Number of multi-indices such that P+1 = nchoosek(No+NN, NN).

    Reference:
    O.P. Le Maître & O.M. Knio, "Spectral Methods for Uncertainty Quantification",
    Springer, Dordrecht (2010), pp. 516-517.
    """
    # Total number of multi-indices
    temp = int(comb(NN + No, No))
    alpha = np.zeros((temp, NN))

    if No == 0:
        P = 0
        return alpha, P

    # Initial order 1 indices
    for j in range(NN):
        alpha[j + 1, j] = 1

    if No == 1:
        P = NN
        return alpha, P

    P = NN
    pp = np.zeros((NN, No))
    pp[0:NN, 0] = 1

    # Loop over orders from 1 to No-1
    for k in range(1, No):
        L = P
        for i in range(NN):
            # Sum of number of multi-indices of lower orders
            pp[i, k] = pp[i:NN, k - 1].sum(axis=0)

        for j in range(NN):
            # Generate multi-indices of next order based on previous ones
            for m in range(L - int(pp[j, k]), L):
                P = P + 1
                alpha[P, 0:NN] = alpha[m + 1, 0:NN]
                alpha[P, j] = alpha[P, j] + 1

    return alpha, P
