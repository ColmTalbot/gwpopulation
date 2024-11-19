import itertools

import numpy as np
import pytest
from bilby.core.prior import DeltaFunction, Normal, PriorDict, Uniform

import gwpopulation
from gwpopulation.models import mass
from gwpopulation.utils import to_numpy

from . import TEST_BACKENDS

xp = np
N_TEST = 10


def double_power_prior():
    power_prior = PriorDict()
    power_prior["alpha_1"] = Uniform(minimum=-4, maximum=12)
    power_prior["alpha_2"] = Uniform(minimum=-4, maximum=12)
    power_prior["beta"] = Uniform(minimum=-4, maximum=12)
    power_prior["mmin"] = Uniform(minimum=3, maximum=10)
    power_prior["mmax"] = Uniform(minimum=40, maximum=100)
    power_prior["break_fraction"] = Uniform(minimum=0, maximum=1)
    return power_prior


def power_prior():
    power_prior = PriorDict()
    power_prior["alpha"] = Uniform(minimum=-4, maximum=12)
    power_prior["beta"] = Uniform(minimum=-4, maximum=12)
    power_prior["mmin"] = Uniform(minimum=3, maximum=10)
    power_prior["mmax"] = Uniform(minimum=40, maximum=100)
    return power_prior


def gauss_prior():
    gauss_prior = PriorDict()
    gauss_prior["lam"] = Uniform(minimum=0, maximum=1)
    gauss_prior["mpp"] = Uniform(minimum=20, maximum=60)
    gauss_prior["sigpp"] = Uniform(minimum=0, maximum=10)
    return gauss_prior


def double_gauss_prior():
    double_gauss_prior = PriorDict()
    double_gauss_prior["lam"] = Uniform(minimum=0, maximum=1)
    double_gauss_prior["lam_1"] = Uniform(minimum=0, maximum=1)
    double_gauss_prior["mpp_1"] = Uniform(minimum=20, maximum=60)
    double_gauss_prior["mpp_2"] = Uniform(minimum=20, maximum=100)
    double_gauss_prior["sigpp_1"] = Uniform(minimum=0, maximum=10)
    double_gauss_prior["sigpp_2"] = Uniform(minimum=0, maximum=10)
    return double_gauss_prior


def smooth_prior():
    smooth_prior = PriorDict()
    smooth_prior["delta_m"] = Uniform(minimum=0, maximum=10)
    return smooth_prior


def interpolated_prior(n_nodes):
    prior = PriorDict()
    nodes = np.logspace(np.log10(2), np.log10(100), n_nodes)
    for ii in range(n_nodes):
        prior[f"mass{ii}"] = DeltaFunction(nodes[ii])
        prior[f"fmass{ii}"] = Normal(0, 1)
    return prior


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_double_power_law_zero_below_mmin(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    to_numpy = gwpopulation.utils.to_numpy
    m1s, _, _ = get_primary_mass_ratio_data(xp)
    prior = double_power_prior()
    for ii in range(N_TEST):
        parameters = prior.sample()
        del parameters["beta"]
        p_m = mass.double_power_law_primary_mass(m1s, **parameters)
        p_m = to_numpy(p_m)
        assert np.max(p_m[to_numpy(m1s) < parameters["mmin"]]) == 0.0


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_power_law_primary_mass_ratio_zero_above_mmax(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    to_numpy = gwpopulation.utils.to_numpy
    m1s, qs, dataset = get_primary_mass_ratio_data(xp)
    prior = double_power_prior()
    m1s = dataset["mass_1"]
    for ii in range(N_TEST):
        parameters = prior.sample()
        p_m = mass.double_power_law_primary_power_law_mass_ratio(dataset, **parameters)
        p_m = to_numpy(p_m)
        assert np.max(p_m[to_numpy(m1s) > parameters["mmax"]]) == 0.0


def get_primary_mass_ratio_data(xp):
    m1s = xp.linspace(3, 100, 1000)
    qs = xp.linspace(0.01, 1, 500)
    m1s_grid, qs_grid = xp.meshgrid(m1s, qs)
    dataset = dict(mass_1=m1s_grid, mass_ratio=qs_grid)
    dataset["mass_2"] = dataset["mass_1"] * dataset["mass_ratio"]
    return m1s, qs, dataset


def get_primary_secondary_data(xp):
    ms = xp.linspace(3, 100, 1000)
    dm = ms[1] - ms[0]
    m1s_grid, m2s_grid = xp.meshgrid(ms, ms)
    dataset = dict(mass_1=m1s_grid, mass_2=m2s_grid)
    return to_numpy(ms), float(dm), dataset


def get_smoothed_data(xp):
    m1s = xp.linspace(2, 100, 1000)
    qs = xp.linspace(0.01, 1, 500)
    m1s_grid, qs_grid = xp.meshgrid(m1s, qs)
    dataset = dict(mass_1=m1s_grid, mass_ratio=qs_grid)
    return m1s, qs, dataset


analytic_models = {
    "single": mass.power_law_primary_mass_ratio,
    "double": mass.double_power_law_primary_power_law_mass_ratio,
    "identical": mass.power_law_primary_secondary_identical,
    "independent": mass.power_law_primary_secondary_independent,
    "two": mass.two_component_primary_mass_ratio,
    "two-identical": mass.two_component_primary_secondary_identical,
    "two-independent": mass.two_component_primary_secondary_independent,
}


def analytic_prior(model):
    if model in ["single", "independent"]:
        return power_prior()
    elif model == "double":
        return double_power_prior()
    elif model == "identical":
        prior = power_prior()
        del prior["beta"]
        return prior
    elif model in ["two", "two-independent"]:
        prior = power_prior()
        prior.update(gauss_prior())
        prior["gaussian_mass_maximum"] = 80
        return prior
    elif model == "two-identical":
        prior = power_prior()
        prior.update(gauss_prior())
        prior["gaussian_mass_maximum"] = 80
        del prior["beta"]
        return prior
    raise ValueError(f"{model} not known")


@pytest.mark.parametrize(
    "backend,model", itertools.product(TEST_BACKENDS, analytic_models.keys())
)
def test_zero_below_mmin(backend, model):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    to_numpy = gwpopulation.utils.to_numpy
    _, _, dataset = get_primary_mass_ratio_data(xp)
    prior = analytic_prior(model)
    model = analytic_models[model]
    m2s = dataset["mass_2"]
    for _ in range(N_TEST):
        parameters = prior.sample()
        p_m = to_numpy(model(dataset, **parameters))
        assert np.max(p_m[to_numpy(m2s) < parameters["mmin"]]) == 0.0


@pytest.mark.parametrize(
    "backend,model", itertools.product(TEST_BACKENDS, analytic_models.keys())
)
def test_zero_above_mmax(backend, model):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    to_numpy = gwpopulation.utils.to_numpy
    _, _, dataset = get_primary_mass_ratio_data(xp)
    prior = analytic_prior(model)
    model = analytic_models[model]
    m1s = dataset["mass_1"]
    for _ in range(N_TEST):
        parameters = prior.sample()
        p_m = to_numpy(model(dataset, **parameters))
        mmax = max(parameters["mmax"], parameters.get("gaussian_mass_maximum", 0))
        assert np.max(p_m[to_numpy(m1s) > mmax]) == 0.0


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_single_peak_delta_m_zero_matches_two_component_primary_mass_ratio(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    _, _, dataset = get_smoothed_data(xp)
    max_diffs = list()
    prior = power_prior()
    prior.update(gauss_prior())
    model = mass.SinglePeakSmoothedMassDistribution()
    for _ in range(N_TEST):
        parameters = prior.sample()
        p_m1 = mass.two_component_primary_mass_ratio(dataset, **parameters)
        parameters["delta_m"] = 0
        p_m2 = model(dataset, **parameters)
        max_diffs.append(_max_abs_difference(p_m1, p_m2, xp=xp))
    assert max(max_diffs) < 1e-5


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_double_peak_delta_m_zero_matches_three_component_primary_mass_ratio(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    _, _, dataset = get_smoothed_data(xp)
    max_diffs = list()
    prior = power_prior()
    prior.update(double_gauss_prior())
    model = mass.MultiPeakSmoothedMassDistribution()
    for _ in range(N_TEST):
        parameters = prior.sample()
        del parameters["beta"]
        p_m1 = mass.three_component_single(mass=dataset["mass_1"], **parameters)
        parameters["delta_m"] = 0
        p_m2 = model.p_m1(dataset, **parameters)
        max_diffs.append(_max_abs_difference(p_m1, p_m2, xp=xp))
    assert max(max_diffs) < 1e-5


def _normalised(model, prior, xp):
    """
    Add jit to jax testing as these functions involve
    caching and so we burn a call and then jit.
    """
    m1s, qs, dataset = get_smoothed_data(xp)
    parameters = prior.sample()
    _ = model(dataset, **parameters)

    if "jax" in xp.__name__:
        import jax

        model = jax.jit(model)

    norms = list()
    for _ in range(N_TEST):
        parameters = prior.sample()
        p_m = model(dataset, **parameters)
        norms.append(float(xp.trapz(xp.trapz(p_m, m1s), qs)))
    assert _max_abs_difference(norms, 1.0, xp=xp) < 0.01


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_single_peak_normalised(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    model = mass.SinglePeakSmoothedMassDistribution()
    prior = power_prior()
    prior.update(gauss_prior())
    prior.update(smooth_prior())
    _normalised(model, prior, xp)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_double_peak_normalised(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    model = mass.MultiPeakSmoothedMassDistribution()
    prior = power_prior()
    prior.update(double_gauss_prior())
    prior.update(smooth_prior())
    _normalised(model, prior, xp)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_broken_power_law_normalised(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    model = mass.BrokenPowerLawSmoothedMassDistribution()
    prior = double_power_prior()
    prior.update(smooth_prior())
    _normalised(model, prior, xp)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_broken_power_law_peak_normalised(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    model = mass.BrokenPowerLawPeakSmoothedMassDistribution()
    prior = double_power_prior()
    prior.update(smooth_prior())
    prior.update(gauss_prior())
    _normalised(model, prior, xp)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_set_minimum_and_maximum(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    model = mass.SinglePeakSmoothedMassDistribution(mmin=5, mmax=150)
    parameters = gauss_prior().sample()
    parameters.update(power_prior().sample())
    parameters.update(smooth_prior().sample())
    parameters["mpp"] = 130
    parameters["sigpp"] = 1
    parameters["lam"] = 0.5
    parameters["mmin"] = 6
    assert (
        model(dict(mass_1=8 * xp.ones(6), mass_ratio=0.5 * xp.ones(6)), **parameters)[0]
        == 0
    )
    assert (
        model(dict(mass_1=130 * xp.ones(5), mass_ratio=0.9 * xp.ones(5)), **parameters)[
            0
        ]
        > 0
    )


def test_mmin_below_global_minimum_raises_error():
    gwpopulation.set_backend("numpy")
    model = mass.SinglePeakSmoothedMassDistribution(mmin=5, mmax=150)
    parameters = gauss_prior().sample()
    parameters.update(power_prior().sample())
    parameters.update(smooth_prior().sample())
    parameters["mmin"] = 2
    with pytest.raises(ValueError):
        model(dict(mass_1=5, mass_ratio=0.9), **parameters)


def test_mmax_above_global_maximum_raises_error():
    gwpopulation.set_backend("numpy")
    model = mass.SinglePeakSmoothedMassDistribution(mmin=5, mmax=150)
    parameters = gauss_prior().sample()
    parameters.update(power_prior().sample())
    parameters.update(smooth_prior().sample())
    parameters["mmax"] = 200
    with pytest.raises(ValueError):
        model(dict(mass_1=5, mass_ratio=0.9), **parameters)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_interpolated_power_law_p_m1_normalised(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    model = mass.InterpolatedPowerlaw()
    prior = interpolated_prior(10)
    prior.update(smooth_prior())
    prior.update(power_prior())
    _normalised(model, prior, xp)


def test_interpolated_power_law_p_m1_variable_names():
    model = mass.InterpolatedPowerlaw()
    prior = interpolated_prior(10)
    prior.update(smooth_prior())
    prior.update(power_prior())
    assert model.variable_names == set(prior.keys())


def _max_abs_difference(array, comparison, xp=np):
    return float(xp.max(xp.abs(comparison - xp.asarray(array))))
