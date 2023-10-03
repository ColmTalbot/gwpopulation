import numpy as np
import pytest
from bilby.core.prior import PriorDict, Uniform

import gwpopulation
from gwpopulation.models import spin
from gwpopulation.utils import truncnorm

from . import TEST_BACKENDS

xp = np
N_TEST = 100


def tilt_prior():
    return PriorDict(dict(xi_spin=Uniform(0, 1), sigma_spin=Uniform(0, 4)))


def magnitude_prior():
    return PriorDict(
        dict(amax=Uniform(0.3, 1), alpha_chi=Uniform(1, 4), beta_chi=Uniform(1, 4))
    )


def tilt_test_data(xp):
    costilts = xp.linspace(-1, 1, 1000)
    dataset = dict(
        cos_tilt_1=xp.einsum("i,j->ij", costilts, xp.ones_like(costilts)),
        cos_tilt_2=xp.einsum("i,j->ji", costilts, xp.ones_like(costilts)),
    )
    return costilts, dataset


def magnitude_test_data(xp):
    a_array = xp.linspace(0, 1, 1000)
    dataset = dict(
        a_1=xp.einsum("i,j->ij", a_array, xp.ones_like(a_array)),
        a_2=xp.einsum("i,j->ji", a_array, xp.ones_like(a_array)),
    )
    return a_array, dataset


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_spin_orientation_normalised(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    norms = list()
    prior = tilt_prior()
    costilts, dataset = tilt_test_data(xp)
    for ii in range(N_TEST):
        parameters = prior.sample()
        temp = spin.iid_spin_orientation_gaussian_isotropic(dataset, **parameters)
        norms.append(float(xp.trapz(xp.trapz(temp, costilts), costilts)))
    assert float(np.max(np.abs(1 - np.asarray(norms)))) < 1e-5


@pytest.mark.parametrize("backend", TEST_BACKENDS)
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


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_spin_magnitude_normalised(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    prior = magnitude_prior()
    a_array, dataset = magnitude_test_data(xp)
    norms = list()
    for ii in range(N_TEST):
        parameters = prior.sample()
        temp = spin.iid_spin_magnitude_beta(dataset, **parameters)
        norms.append(xp.trapz(xp.trapz(temp, a_array), a_array))
    assert float(xp.max(xp.abs(1 - xp.asarray(norms)))) < 1e-2


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_returns_zero_alpha_beta_less_zero(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    prior = magnitude_prior()
    a_array, dataset = magnitude_test_data(xp)
    parameters = prior.sample()
    for key in ["alpha_chi", "beta_chi"]:
        parameters[key] = -1
        assert np.all(spin.iid_spin_magnitude_beta(dataset, **parameters) == 0)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
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


@pytest.mark.parametrize("backend", TEST_BACKENDS)
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


@pytest.mark.parametrize("backend", TEST_BACKENDS)
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


@pytest.mark.parametrize("backend", TEST_BACKENDS)
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


@pytest.mark.parametrize("backend", TEST_BACKENDS)
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
                xp.trapz(
                    xp.trapz(prob, xp.linspace(-1, 1, 501)), xp.linspace(0, 1, 300)
                )
                - 1
            )
        )
        < 1e-3
    )


@pytest.mark.parametrize("backend", TEST_BACKENDS)
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
