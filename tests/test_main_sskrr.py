"""
Unit tests for the main_sskrr.py script.
"""

from typing import TYPE_CHECKING
import numpy as np
import pytest
import torch

from main_sskrr import (
    setup_matplotlib,
    ishigami_function,
    rosenbrock_function,
    legendre_poly,
    prod_legendre_poly,
    compute_rho_torch_pyro,
    compute_rms_from_data,
    create_pif_pic,
    compute_pce_norm_moment_order,
    load_config,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


def test_setup_matplotlib_latex(mocker: "MockerFixture") -> None:
    """Test setup_matplotlib with use_latex=True."""
    mock_rc = mocker.patch("main_sskrr.rc")
    mock_plt = mocker.patch("main_sskrr.plt")
    
    setup_matplotlib(use_latex=True)
    
    mock_rc.assert_called_once_with("text", usetex=True)
    assert mock_plt.rc.called


def test_setup_matplotlib_no_latex(mocker: "MockerFixture") -> None:
    """Test setup_matplotlib with use_latex=False."""
    mock_plt = mocker.patch("main_sskrr.plt")
    # Using a dictionary-like mock for rcParams
    mock_plt.rcParams = {}
    
    setup_matplotlib(use_latex=False)
    
    assert mock_plt.rcParams["text.usetex"] is False


def test_ishigami_function() -> None:
    """Test the Ishigami function with known values."""
    # Point at origin
    x0 = np.array([[0.0, 0.0, 0.0]])
    res0 = ishigami_function(x0)
    assert np.allclose(res0, [0.0])

    # Point at (pi/2, 0, 0) -> sin(pi/2) = 1
    x1 = np.array([[np.pi / 2, 0.0, 0.0]])
    res1 = ishigami_function(x1)
    assert np.allclose(res1, [1.0])


def test_rosenbrock_function() -> None:
    """Test the Rosenbrock function with known values."""
    # Global minimum at (1, 1, ..., 1) is 0
    x_min = np.ones((1, 3))
    res_min = rosenbrock_function(x_min)
    assert np.allclose(res_min, [0.0])

    # At (0, 0) in 2D
    x0 = np.zeros((1, 2))
    res0 = rosenbrock_function(x0)
    # 100 * (0 - 0)^2 + (1 - 0)^2 = 1
    assert np.allclose(res0, [1.0])


def test_legendre_poly() -> None:
    """Test the legendre_poly function."""
    x = np.array([-1.0, 0.0, 1.0])
    
    # Order 0: eval_legendre(0, x) = 1, norm = sqrt(2/1) -> 1 / sqrt(2)
    res0 = legendre_poly(x, 0)
    expected0 = np.ones_like(x) / np.sqrt(2.0)
    assert np.allclose(res0, expected0)
    
    # Order 1: eval_legendre(1, x) = x, norm = sqrt(2/3) -> x / sqrt(2/3)
    res1 = legendre_poly(x, 1)
    expected1 = x / np.sqrt(2.0 / 3.0)
    assert np.allclose(res1, expected1)


def test_prod_legendre_poly() -> None:
    """Test the prod_legendre_poly function."""
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    indx = np.array([0, 1])
    
    # Poly(dim0, order0) * Poly(dim1, order1)
    # Order 0 gives 1/sqrt(2), Order 1 gives x_dim1 / sqrt(2/3)
    res = prod_legendre_poly(x, indx)
    
    expected0 = (1 / np.sqrt(2.0)) * (0.0 / np.sqrt(2.0 / 3.0))
    expected1 = (1 / np.sqrt(2.0)) * (1.0 / np.sqrt(2.0 / 3.0))
    
    assert res.shape == (2, 1)
    assert np.allclose(res.flatten(), [expected0, expected1])


class MockKernel:
    """Mock for GP kernel."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mock forward returning identity of size x."""
        return torch.eye(x.shape[0])


def test_compute_rho_torch_pyro() -> None:
    """Test the compute_rho_torch_pyro function."""
    xobs = torch.zeros((4, 2))
    f_vals = torch.ones(4)
    pif = np.array([True, True, False, False])
    pic = np.array([True, False, False, False])
    nugget = torch.tensor(1e-4)
    kernel = MockKernel()

    rho = compute_rho_torch_pyro(xobs, f_vals, pif, pic, nugget, kernel)
    
    assert isinstance(rho, torch.Tensor)
    # Both ff and fc are ones. kf and kc are identity * (1 + nugget).
    # num = ff^T (kf^-1 ff) = 2 / (1 + nugget)
    # den = fc^T (kc^-1 fc) = 1 / (1 + nugget)
    # rho = 1 - (1/2) = 0.5
    assert torch.isclose(rho, torch.tensor(0.5))


def test_compute_rms_from_data() -> None:
    """Test the compute_rms_from_data function."""
    x = np.array([0.0, 2.0])
    
    # Distance between 0 and 2 is 4
    # RMS = sqrt(2 / (2 * 1) * 4) = 2
    res = compute_rms_from_data(x)
    assert np.isclose(res, 2.0)


def test_create_pif_pic() -> None:
    """Test the create_pif_pic function."""
    num_obs = 10
    nf = 6
    nc = 3
    pif, pic = create_pif_pic(num_obs, nf, nc, seed=42)
    
    assert pif.shape == (10,)
    assert pic.shape == (10,)
    assert pif.sum() == nf
    assert pic.sum() == nc
    # All true in pic must be true in pif
    assert np.all(pif[pic])


def test_compute_pce_norm_moment_order() -> None:
    """Test the compute_pce_norm_moment_order function."""
    alpha = [0, 0]
    beta = [0, 0]
    dimension = 2
    
    # For dim 0: num = 2^1 * 1 * 1 = 2, den = 1! = 1. -> 2
    # Product over 2 dims -> 4. sqrt(4) = 2.0
    res = compute_pce_norm_moment_order(alpha, beta, dimension)
    assert np.isclose(res, 2.0)


def test_load_config(mocker: "MockerFixture") -> None:
    """Test the load_config function."""
    mock_data = "general:\n  test: 1"
    mocker.patch("builtins.open", mocker.mock_open(read_data=mock_data))
    
    res = load_config("dummy_path.yaml")
    assert "general" in res
    assert res["general"]["test"] == 1
