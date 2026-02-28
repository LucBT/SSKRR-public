"""Tests for Jacobi polynomial library."""

from typing import Tuple
import numpy as np
import pytest
from lib.JacobiP import JacobiP
from lib.MultiIndex import MultiIndex
from lib.PJacn import PJacn
from lib.MPJacn import MPJacn
from lib.JNodeWt import JNodeWt


def test_jacobi_p_scalar() -> None:
    """Test JacobiP with scalar input."""
    # P_0(x) for alpha=0, beta=0 is 1/sqrt(2)
    # gamma0 = 2^(0+0+1)/(0+0+1) * gamma(1)*gamma(1)/gamma(1) = 2
    # P_0 = 1/sqrt(2) = 0.70710678
    val = JacobiP(0.5, 0.0, 0.0, 0)
    assert np.isclose(val[0], 1.0 / np.sqrt(2.0))


def test_jacobi_p_array() -> None:
    """Test JacobiP with array input."""
    x = np.array([0.0, 0.5, 1.0])
    val = JacobiP(x, 0.0, 0.0, 0)
    expected = np.ones(3) / np.sqrt(2.0)
    assert np.allclose(val, expected)


def test_multi_index() -> None:
    """Test MultiIndex generation."""
    # For No=2, NN=3, P+1 = comb(2+3, 2) = 10
    m_indices, p_count = MultiIndex(2, 3)
    assert p_count == 9  # p_count is number of indices - 1 (for 0-th order)
    assert m_indices.shape == (10, 3)
    # Check if first index is all zeros
    assert np.all(m_indices[0, :] == 0)


def test_pjacn() -> None:
    """Test PJacn for filtered polynomials and derivatives."""
    x = np.array([0.1, 0.2])
    nn_order = 2
    alpha, beta = 0.0, 0.0
    filter_user = np.array([1.0, 1.0, 1.0])

    p_vals, q_vals = PJacn(x, nn_order, alpha, beta, filter_user)

    assert p_vals.shape == (3, 2)
    assert q_vals.shape == (3, 2)
    # p_vals[0, :] should be JacobiP(x, 0, 0, 0)
    expected_p0 = JacobiP(x, 0.0, 0.0, 0)
    assert np.allclose(p_vals[0, :], expected_p0)


def test_mpjacn() -> None:
    """Test MPJacn for multivariate polynomials."""
    x_point = np.array([0.5, -0.5, 0.0])  # 3D point
    nn_order = 2
    alpha = np.zeros(3)
    beta = np.zeros(3)
    m_indices, _ = MultiIndex(nn_order, 3)

    results = MPJacn(x_point, nn_order, alpha, beta, m_indices)

    assert results is not None
    assert results.shape == (1, 10)
    # First multivariate polynomial (order 0,0,0) should be P_0(x1)*P_0(x2)*P_0(x3)
    p0_val = 1.0 / np.sqrt(2.0)
    assert np.isclose(results[0, 0], p0_val**3)


def test_jnode_wt() -> None:
    """Test JNodeWt for quadrature points and weights."""
    # Gauss-Lobatto for alpha=0, beta=0, NN=9
    # Should return NN+1 = 10 points
    # Note: JNodeWt might need fix for np.np.pi bug before this passes
    nodes, weights = JNodeWt(9, 2, 0.0, 0.0)
    assert len(nodes) == 10
    assert len(weights) == 10
    # Sum of weights should be mu0
    # mu0 = 2^(0+0+1) * gamma(1)*gamma(1)/gamma(2) = 2
    assert np.isclose(np.sum(weights), 2.0)
