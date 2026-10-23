"""
The log densities of the population models: each against the log of the
density, in single precision against double precision, and :code:`-inf`
outside the support.
"""

import sys

import numpy as np
import pytest

import gwpopulation
from gwpopulation.hyperpe import Model
from gwpopulation.models import spin
from gwpopulation.utils import to_numpy

N_TEST = 10
N_SAMPLES = 2000

# Hyperparameters are drawn uniformly from these ranges with a seeded generator,
# so that the single precision comparisons are reproducible. The ranges include
# narrow peaks and steep spin magnitude distributions, so that the densities
# underflow in single precision.
TILT = dict(xi_spin=(0, 1), sigma_spin=(0.1, 4))
TILTS = dict(
    xi_spin=(0, 1), sigma_1=(0.1, 4), sigma_2=(0.1, 4), mu_1=(-1, 1), mu_2=(-1, 1)
)
MAGNITUDE = dict(amax=(0.3, 1), alpha_chi=(1, 30), beta_chi=(1, 30))
MAGNITUDES = dict(
    amax_1=(0.3, 1),
    amax_2=(0.3, 1),
    alpha_chi_1=(1, 30),
    alpha_chi_2=(1, 30),
    beta_chi_1=(1, 30),
    beta_chi_2=(1, 30),
)
CHI_EFF = dict(mu_chi_eff=(-1, 1), sigma_chi_eff=(0.02, 0.3))
CHI_P = dict(mu_chi_p=(0, 1), sigma_chi_p=(0.02, 0.1))


# points that are outside the support for any of the hyperparameters above
def _outside_magnitude(data, parameters):
    return (data["a_1"] > parameters["amax"]) | (data["a_2"] > parameters["amax"])


def _outside_magnitudes(data, parameters):
    return (data["a_1"] > parameters["amax_1"]) | (data["a_2"] > parameters["amax_2"])


def _outside_chi_eff(data, parameters):
    return np.abs(data["chi_eff"]) > 1


def _outside_chi_p(data, parameters):
    return (data["chi_p"] < 0) | (data["chi_p"] > 1)


def _outside_chi_eff_chi_p(data, parameters):
    return _outside_chi_eff(data, parameters) | _outside_chi_p(data, parameters)


def _nowhere(data, parameters):
    return np.zeros(data["mass_1"].shape, dtype=bool)


def _model(component):
    return lambda: Model([component()], cache=False)


# name: (builder of a Model, ranges, outside the support)
MODELS = dict(
    iid_spin=(_model(lambda: spin.iid_spin), TILT | MAGNITUDE, _outside_magnitude),
    iid_spin_magnitude_beta=(
        _model(lambda: spin.iid_spin_magnitude_beta),
        MAGNITUDE,
        _outside_magnitude,
    ),
    independent_spin_magnitude_beta=(
        _model(lambda: spin.independent_spin_magnitude_beta),
        MAGNITUDES,
        _outside_magnitudes,
    ),
    iid_spin_orientation_gaussian_isotropic=(
        _model(lambda: spin.iid_spin_orientation_gaussian_isotropic),
        TILT | dict(mu_spin=(-1, 1)),
        _nowhere,
    ),
    independent_spin_orientation_gaussian_isotropic=(
        _model(lambda: spin.independent_spin_orientation_gaussian_isotropic),
        TILTS,
        _nowhere,
    ),
    # without the isotropic component, the density can underflow
    aligned_spin_orientation=(
        _model(lambda: spin.iid_spin_orientation_gaussian_isotropic),
        dict(xi_spin=1.0, sigma_spin=(0.05, 0.3), mu_spin=(0, 1)),
        _nowhere,
    ),
    gaussian_chi_eff=(
        _model(lambda: spin.gaussian_chi_eff),
        CHI_EFF,
        _outside_chi_eff,
    ),
    gaussian_chi_p=(_model(lambda: spin.gaussian_chi_p), CHI_P, _outside_chi_p),
    GaussianChiEffChiP=(
        _model(spin.GaussianChiEffChiP),
        CHI_EFF | CHI_P | dict(spin_covariance=(-0.75, 0.75)),
        _outside_chi_eff_chi_p,
    ),
)

# the linear densities of these models do not underflow or overflow in single
# precision for the ranges above
LINEAR_SINGLE_PRECISION = [
    "iid_spin_orientation_gaussian_isotropic",
    "independent_spin_orientation_gaussian_isotropic",
]


def _never(data, parameters, ln_expected):
    return np.zeros(ln_expected.shape, dtype=bool)


# Single precision tolerances, where points may be lost (lost), and the log
# densities above which points are compared (compare_above).
SINGLE_PRECISION = {
    name: dict(rtol=2e-5, atol=2e-5, lost=_never, compare_above=-np.inf)
    for name in MODELS
}


def _dataset():
    rng = np.random.default_rng(10)
    mass_1 = np.exp(rng.uniform(np.log(2), np.log(150), N_SAMPLES))
    mass_ratio = rng.uniform(0.01, 1.1, N_SAMPLES)
    return dict(
        mass_1=mass_1,
        mass_ratio=mass_ratio,
        mass_2=mass_1 * mass_ratio,
        a_1=rng.uniform(0, 1, N_SAMPLES),
        a_2=rng.uniform(0, 1, N_SAMPLES),
        cos_tilt_1=rng.uniform(-1, 1, N_SAMPLES),
        cos_tilt_2=rng.uniform(-1, 1, N_SAMPLES),
        chi_eff=rng.uniform(-1.2, 1.2, N_SAMPLES),
        chi_p=rng.uniform(-0.2, 1.2, N_SAMPLES),
        redshift=rng.uniform(0, 2.5, N_SAMPLES),
    )


def _draws(name):
    rng = np.random.default_rng(sum(map(ord, name)))
    ranges = MODELS[name][1]
    return [
        {
            key: float(rng.uniform(*value)) if isinstance(value, tuple) else value
            for key, value in ranges.items()
        }
        for _ in range(N_TEST)
    ]


def _single(values):
    """Round to single precision, keeping double precision types"""
    if isinstance(values, dict):
        return {key: _single(value) for key, value in values.items()}
    elif isinstance(values, np.ndarray):
        return values.astype(np.float32).astype(np.float64)
    return float(np.float32(values))


@pytest.mark.parametrize("name", MODELS)
def test_ln_function_is_used(name):
    model = MODELS[name][0]()
    component = model.models[0]
    ln_function = model.ln_function(component)
    if hasattr(component, "ln_prob"):
        assert ln_function == component.ln_prob
    else:
        module = sys.modules[component.__module__]
        assert ln_function is getattr(module, f"ln_{component.__name__}")


@pytest.mark.parametrize("name", MODELS)
def test_ln_prob_matches_log_prob(backend, name):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    builder, _, outside = MODELS[name]
    model = builder()
    data = _dataset()
    dataset = {key: xp.asarray(value) for key, value in data.items()}
    for parameters in _draws(name):
        prob = to_numpy(model.prob(dataset, **parameters))
        ln_prob = to_numpy(model.ln_prob(dataset, **parameters))
        assert not np.any(np.isnan(ln_prob))
        positive = prob > 1e-300
        assert np.any(positive)
        np.testing.assert_allclose(
            ln_prob[positive], np.log(prob[positive]), rtol=1e-10, atol=1e-12
        )
        # where the density underflows, the log density is finite
        assert np.all(ln_prob[prob == 0] < np.log(1e-300))
        unsupported = outside(data, parameters)
        assert np.all(prob[unsupported] == 0)
        assert np.all(np.isneginf(ln_prob[unsupported]))
    gwpopulation.set_backend("numpy")


@pytest.mark.parametrize("name", MODELS)
def test_ln_prob_single_precision(name):
    jax = pytest.importorskip("jax")
    builder, _, outside = MODELS[name]
    # round the inputs so that only the arithmetic differs
    data = _single(_dataset())
    draws = [_single(parameters) for parameters in _draws(name)]
    gwpopulation.set_backend("numpy")
    model = builder()
    expected = [to_numpy(model.ln_prob(data, **parameters)) for parameters in draws]

    gwpopulation.set_backend("jax", enable_x64=False)
    try:
        xp = gwpopulation.utils.xp
        dataset = {
            key: xp.asarray(value, dtype=xp.float32) for key, value in data.items()
        }
        model = builder()
        jitted = jax.jit(lambda parameters: model.ln_prob(dataset, **parameters))
        settings = SINGLE_PRECISION[name]
        n_linear_lost = 0
        for parameters, ln_expected in zip(draws, expected):
            ln_prob = jitted(parameters)
            assert ln_prob.dtype == xp.float32
            ln_prob = to_numpy(ln_prob)
            assert not np.any(np.isnan(ln_prob))
            assert np.all(np.isneginf(ln_prob[outside(data, parameters)]))
            finite = np.isfinite(ln_expected)
            assert not np.any(np.isfinite(ln_prob) & ~finite)
            lost = finite & np.isneginf(ln_prob)
            assert np.all(settings["lost"](data, parameters, ln_expected)[lost])
            compare = finite & ~lost & (ln_expected > settings["compare_above"])
            np.testing.assert_allclose(
                ln_prob[compare],
                ln_expected[compare],
                rtol=settings["rtol"],
                atol=settings["atol"],
            )
            prob = to_numpy(model.prob(dataset, **parameters))
            n_linear_lost += np.sum(finite & ~(prob > 0))
        # the log density is needed: the density underflows or overflows
        assert (n_linear_lost > 0) != (name in LINEAR_SINGLE_PRECISION)
    finally:
        gwpopulation.set_backend("jax", enable_x64=True)
        gwpopulation.set_backend("numpy")
