import numpy as np
import pytest
from bilby.core.prior import DeltaFunction, Normal, PriorDict, Uniform

import gwpopulation
from gwpopulation.models import spin
from gwpopulation.utils import trapezoid, truncnorm

xp = np
N_TEST = 100


def tilt_prior():
    return PriorDict(dict(xi_spin=Uniform(0, 1), sigma_spin=Uniform(0, 4)))


def magnitude_prior():
    return PriorDict(
        dict(amax=Uniform(0.3, 1), alpha_chi=Uniform(1, 4), beta_chi=Uniform(1, 4))
    )


def tilt_test_data(xp):
    costilts = xp.linspace(-1, 1, 2000)
    dataset = dict(
        cos_tilt_1=xp.einsum("i,j->ij", costilts, xp.ones_like(costilts)),
        cos_tilt_2=xp.einsum("i,j->ji", costilts, xp.ones_like(costilts)),
    )
    return costilts, dataset


def magnitude_test_data(xp):
    a_array = xp.linspace(0, 1, 2000)
    dataset = dict(
        a_1=xp.einsum("i,j->ij", a_array, xp.ones_like(a_array)),
        a_2=xp.einsum("i,j->ji", a_array, xp.ones_like(a_array)),
    )
    return a_array, dataset


def test_spin_orientation_normalised(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    norms = list()
    prior = tilt_prior()
    costilts, dataset = tilt_test_data(xp)
    for ii in range(N_TEST):
        parameters = prior.sample()
        temp = spin.iid_spin_orientation_gaussian_isotropic(dataset, **parameters)
    norms.append(float(trapezoid(trapezoid(temp, costilts), costilts)))
    assert float(np.max(np.abs(1 - np.asarray(norms)))) < 1e-5


def test_iid_matches_independent_tilts(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    iid_params = dict(xi_spin=0.5, sigma_spin=0.5)
    ind_params = dict(xi_spin=0.5, sigma_1=0.5, sigma_2=0.5)
    _, dataset = tilt_test_data(xp)
    assert (
        xp.max(
            spin.iid_spin_orientation_gaussian_isotropic(dataset, **iid_params)
            - spin.independent_spin_orientation_gaussian_isotropic(
                dataset, **ind_params
            )
        )
        == 0
    )


def test_spin_magnitude_normalised(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    prior = magnitude_prior()
    a_array, dataset = magnitude_test_data(xp)
    norms = list()
    for ii in range(N_TEST):
        parameters = prior.sample()
        temp = spin.iid_spin_magnitude_beta(dataset, **parameters)
    norms.append(trapezoid(trapezoid(temp, a_array), a_array))
    assert float(xp.max(xp.abs(1 - xp.asarray(norms)))) < 1e-2


def test_returns_zero_alpha_beta_less_zero(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    prior = magnitude_prior()
    a_array, dataset = magnitude_test_data(xp)
    parameters = prior.sample()
    for key in ["alpha_chi", "beta_chi"]:
        parameters[key] = -1
        assert np.all(spin.iid_spin_magnitude_beta(dataset, **parameters) == 0)


def test_iid_matches_independent_magnitudes(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    prior = magnitude_prior()
    a_array, dataset = magnitude_test_data(xp)
    iid_params = prior.sample()
    ind_params = dict()
    ind_params.update({key + "_1": iid_params[key] for key in iid_params})
    ind_params.update({key + "_2": iid_params[key] for key in iid_params})
    assert (
        float(
            xp.max(
                spin.iid_spin_magnitude_beta(dataset, **iid_params)
                - spin.independent_spin_magnitude_beta(dataset, **ind_params)
            )
        )
        < 1e-5
    )


def test_iid_matches_independent(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    test_data = dict()
    costilts, new_data = tilt_test_data(xp)
    test_data.update(new_data)
    a_array, new_data = magnitude_test_data(xp)
    test_data.update(new_data)
    prior = magnitude_prior()
    prior.update(tilt_prior())
    params = prior.sample()
    mag_params = {key: params[key] for key in ["amax", "alpha_chi", "beta_chi"]}
    tilt_params = {key: params[key] for key in ["xi_spin", "sigma_spin"]}
    assert (
        float(
            xp.max(
                spin.iid_spin(test_data, **params)
                - spin.iid_spin_magnitude_beta(test_data, **mag_params)
                * spin.iid_spin_orientation_gaussian_isotropic(test_data, **tilt_params)
            )
        )
        < 1e-5
    )


def test_gaussian_chi_eff(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    assert (
        float(
            xp.max(
                spin.gaussian_chi_eff(
                    dict(chi_eff=xp.linspace(-2, 2, 1001)),
                    mu_chi_eff=0.4,
                    sigma_chi_eff=0.1,
                )
                - truncnorm(xp.linspace(-2, 2, 1001), mu=0.4, sigma=0.1, low=-1, high=1)
            )
        )
        < 1e-5
    )


def test_gaussian_chi_p(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    assert (
        float(
            xp.max(
                spin.gaussian_chi_p(
                    dict(chi_p=xp.linspace(-2, 2, 1001)), mu_chi_p=0.4, sigma_chi_p=0.1
                )
                - truncnorm(xp.linspace(-2, 2, 1001), mu=0.4, sigma=0.1, low=0, high=1)
            )
        )
        == 0
    )


def test_2d_gaussian_normalized(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    model = spin.GaussianChiEffChiP()
    chi_eff, chi_p = xp.meshgrid(xp.linspace(-1, 1, 501), xp.linspace(0, 1, 300))
    parameters = dict(
        mu_chi_eff=0.1,
        mu_chi_p=0.3,
        sigma_chi_eff=0.6,
        sigma_chi_p=0.5,
        spin_covariance=0.9,
    )
    prob = model(dict(chi_eff=chi_eff, chi_p=chi_p), **parameters)
    assert (
        xp.max(
            xp.abs(
                trapezoid(
                    trapezoid(prob, xp.linspace(-1, 1, 501)), xp.linspace(0, 1, 300)
                )
                - 1
            )
        )
        < 1e-3
    )


def test_2d_gaussian_no_covariance_matches_independent(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    model = spin.GaussianChiEffChiP()
    chi_eff, chi_p = xp.meshgrid(xp.linspace(-1, 1, 501), xp.linspace(0, 1, 300))
    data = dict(chi_eff=chi_eff, chi_p=chi_p)
    assert (
        xp.max(
            xp.abs(
                spin.gaussian_chi_eff(data, mu_chi_eff=0.6, sigma_chi_eff=0.2)
                * spin.gaussian_chi_p(data, mu_chi_p=0.4, sigma_chi_p=0.1)
                - model(
                    data,
                    mu_chi_eff=0.6,
                    mu_chi_p=0.4,
                    sigma_chi_eff=0.2,
                    sigma_chi_p=0.1,
                    spin_covariance=0.0,
                )
            )
        )
        < 1e-3
    )


# Helper functions for spline model tests


def spline_magnitude_prior(n_nodes, minimum=0, maximum=1):
    """Create prior for SplineSpinMagnitudeIdentical with n_nodes."""
    prior = PriorDict()
    nodes = np.linspace(minimum, maximum, n_nodes)
    for ii in range(n_nodes):
        prior[f"a{ii}"] = DeltaFunction(nodes[ii])
        prior[f"fa{ii}"] = Normal(0, 1)
    return prior


def spline_tilt_prior(n_nodes, minimum=-1, maximum=1):
    """Create prior for SplineSpinTiltIdentical with n_nodes."""
    prior = PriorDict()
    nodes = np.linspace(minimum, maximum, n_nodes)
    for ii in range(n_nodes):
        prior[f"cos_tilt{ii}"] = DeltaFunction(nodes[ii])
        prior[f"fcos_tilt{ii}"] = Normal(0, 1)
    return prior


# Tests for SplineSpinMagnitudeIdentical


@pytest.mark.parametrize("nodes", [4, 5, 7])
def test_spline_spin_magnitude_normalised(backend, nodes):
    """Test that SplineSpinMagnitudeIdentical is properly normalized."""
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    model = spin.SplineSpinMagnitudeIdentical(nodes=nodes)
    prior = spline_magnitude_prior(nodes)
    a_array, dataset = magnitude_test_data(xp)

    norms = list()
    # Use fewer iterations for computational efficiency
    n_test = 10
    for _ in range(n_test):
        parameters = prior.sample()
        prob = model(dataset, **parameters)
        norm = float(trapezoid(trapezoid(prob, a_array), a_array))
        norms.append(norm)

    # Allow 5% error tolerance for numerical integration
    assert xp.max(xp.abs(1 - xp.asarray(norms))) < 0.05


@pytest.mark.parametrize("kind", ["linear", "cubic"])
def test_spline_spin_magnitude_interpolation_kind(backend, kind):
    """Test that different interpolation kinds work for SplineSpinMagnitudeIdentical."""
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    nodes = 5
    model = spin.SplineSpinMagnitudeIdentical(nodes=nodes, kind=kind)
    prior = spline_magnitude_prior(nodes)
    a_array, dataset = magnitude_test_data(xp)

    parameters = prior.sample()
    prob = model(dataset, **parameters)
    norm = float(trapezoid(trapezoid(prob, a_array), a_array))

    # Should be normalized regardless of interpolation kind
    # Allow 5% error tolerance for numerical integration
    assert abs(norm - 1.0) < 0.05


@pytest.mark.parametrize("regularize", [True, False])
def test_spline_spin_magnitude_regularization(backend, regularize):
    """Test that regularization option works for SplineSpinMagnitudeIdentical."""
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    nodes = 5
    model = spin.SplineSpinMagnitudeIdentical(nodes=nodes, regularize=regularize)
    prior = spline_magnitude_prior(nodes)

    if regularize:
        prior["rmsa"] = Uniform(0.1, 2.0)

    a_array, dataset = magnitude_test_data(xp)
    parameters = prior.sample()

    # Should not raise an error
    prob = model(dataset, **parameters)

    # Check that variable names match expected
    if regularize:
        assert "rmsa" in model.variable_names
    else:
        assert "rmsa" not in model.variable_names


def test_spline_spin_magnitude_zero_outside_range(backend):
    """Test that SplineSpinMagnitudeIdentical returns zero outside the valid range."""
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    nodes = 5
    minimum, maximum = 0.2, 0.8
    model = spin.SplineSpinMagnitudeIdentical(
        minimum=minimum, maximum=maximum, nodes=nodes
    )
    prior = spline_magnitude_prior(nodes, minimum=minimum, maximum=maximum)

    # Create test data that includes values outside the range
    a_array = xp.linspace(0, 1, 2000)
    dataset = dict(
        a_1=xp.einsum("i,j->ij", a_array, xp.ones_like(a_array)),
        a_2=xp.einsum("i,j->ji", a_array, xp.ones_like(a_array)),
    )

    parameters = prior.sample()
    prob = model(dataset, **parameters)

    # Values outside [0.2, 0.8] should be zero
    prob_np = gwpopulation.utils.to_numpy(prob)
    a_array_np = gwpopulation.utils.to_numpy(a_array)

    assert np.max(prob_np[(a_array_np < minimum) | (a_array_np > maximum)]) == 0.0


def test_spline_spin_magnitude_variable_names():
    """Test that variable_names property works for SplineSpinMagnitudeIdentical."""
    nodes = 5
    model = spin.SplineSpinMagnitudeIdentical(nodes=nodes, regularize=False)

    # Should have node positions and values
    expected = [f"a{ii}" for ii in range(nodes)] + [f"fa{ii}" for ii in range(nodes)]
    assert set(model.variable_names) == set(expected)

    # With regularization, should also have rms parameter
    model_reg = spin.SplineSpinMagnitudeIdentical(nodes=nodes, regularize=True)
    expected_reg = expected + ["rmsa"]
    assert set(model_reg.variable_names) == set(expected_reg)


# Tests for SplineSpinTiltIdentical


@pytest.mark.parametrize("nodes", [4, 5, 7])
def test_spline_spin_tilt_normalised(backend, nodes):
    """Test that SplineSpinTiltIdentical is properly normalized."""
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    model = spin.SplineSpinTiltIdentical(nodes=nodes)
    prior = spline_tilt_prior(nodes)
    costilts, dataset = tilt_test_data(xp)

    norms = list()
    # Use fewer iterations for computational efficiency
    n_test = 10
    for _ in range(n_test):
        parameters = prior.sample()
        prob = model(dataset, **parameters)
        norm = float(trapezoid(trapezoid(prob, costilts), costilts))
        norms.append(norm)

    # Allow 5% error tolerance for numerical integration
    assert xp.max(xp.abs(1 - xp.asarray(norms))) < 0.05


@pytest.mark.parametrize("kind", ["linear", "cubic"])
def test_spline_spin_tilt_interpolation_kind(backend, kind):
    """Test that different interpolation kinds work for SplineSpinTiltIdentical."""
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    nodes = 5
    model = spin.SplineSpinTiltIdentical(nodes=nodes, kind=kind)
    prior = spline_tilt_prior(nodes)
    costilts, dataset = tilt_test_data(xp)

    parameters = prior.sample()
    prob = model(dataset, **parameters)
    norm = float(trapezoid(trapezoid(prob, costilts), costilts))

    # Should be normalized regardless of interpolation kind
    # Allow 5% error tolerance for numerical integration
    assert abs(norm - 1.0) < 0.05


@pytest.mark.parametrize("regularize", [True, False])
def test_spline_spin_tilt_regularization(backend, regularize):
    """Test that regularization option works for SplineSpinTiltIdentical."""
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    nodes = 5
    model = spin.SplineSpinTiltIdentical(nodes=nodes, regularize=regularize)
    prior = spline_tilt_prior(nodes)

    if regularize:
        prior["rmscos_tilt"] = Uniform(0.1, 2.0)

    costilts, dataset = tilt_test_data(xp)
    parameters = prior.sample()

    # Should not raise an error
    prob = model(dataset, **parameters)

    # Check that variable names match expected
    if regularize:
        assert "rmscos_tilt" in model.variable_names
    else:
        assert "rmscos_tilt" not in model.variable_names


def test_spline_spin_tilt_zero_outside_range(backend):
    """Test that SplineSpinTiltIdentical returns zero outside the valid range."""
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    nodes = 5
    minimum, maximum = -0.5, 0.5
    model = spin.SplineSpinTiltIdentical(minimum=minimum, maximum=maximum, nodes=nodes)
    prior = spline_tilt_prior(nodes, minimum=minimum, maximum=maximum)

    # Create test data that includes values outside the range
    costilts = xp.linspace(-1, 1, 2000)
    dataset = dict(
        cos_tilt_1=xp.einsum("i,j->ij", costilts, xp.ones_like(costilts)),
        cos_tilt_2=xp.einsum("i,j->ji", costilts, xp.ones_like(costilts)),
    )

    parameters = prior.sample()
    prob = model(dataset, **parameters)

    # Values outside [-0.5, 0.5] should be zero
    prob_np = gwpopulation.utils.to_numpy(prob)
    costilts_np = gwpopulation.utils.to_numpy(costilts)

    assert np.max(prob_np[(costilts_np < minimum) | (costilts_np > maximum)]) == 0.0


def test_spline_spin_tilt_variable_names():
    """Test that variable_names property works for SplineSpinTiltIdentical."""
    nodes = 5
    model = spin.SplineSpinTiltIdentical(nodes=nodes, regularize=False)

    # Should have node positions and values
    expected = [f"cos_tilt{ii}" for ii in range(nodes)] + [
        f"fcos_tilt{ii}" for ii in range(nodes)
    ]
    assert set(model.variable_names) == set(expected)

    # With regularization, should also have rms parameter
    model_reg = spin.SplineSpinTiltIdentical(nodes=nodes, regularize=True)
    expected_reg = expected + ["rmscos_tilt"]
    assert set(model_reg.variable_names) == set(expected_reg)


# Test boundary conditions


def test_spline_magnitude_at_boundaries(backend):
    """Test that SplineSpinMagnitudeIdentical handles boundary values correctly."""
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    nodes = 5
    model = spin.SplineSpinMagnitudeIdentical(minimum=0, maximum=1, nodes=nodes)
    prior = spline_magnitude_prior(nodes)

    # Test with data exactly at boundaries
    dataset = dict(
        a_1=xp.array([[0.0, 0.5, 1.0]]),
        a_2=xp.array([[0.0, 0.5, 1.0]]),
    )

    parameters = prior.sample()
    prob = model(dataset, **parameters)

    # Should get valid probabilities at boundaries
    assert xp.all(prob >= 0)


def test_spline_tilt_at_boundaries(backend):
    """Test that SplineSpinTiltIdentical handles boundary values correctly."""
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    nodes = 5
    model = spin.SplineSpinTiltIdentical(minimum=-1, maximum=1, nodes=nodes)
    prior = spline_tilt_prior(nodes)

    # Test with data exactly at boundaries
    dataset = dict(
        cos_tilt_1=xp.array([[-1.0, 0.0, 1.0]]),
        cos_tilt_2=xp.array([[-1.0, 0.0, 1.0]]),
    )

    parameters = prior.sample()
    prob = model(dataset, **parameters)

    # Should get valid probabilities at boundaries
    assert xp.all(prob >= 0)
