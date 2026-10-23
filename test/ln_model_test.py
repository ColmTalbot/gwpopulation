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
from gwpopulation.models import mass, redshift, spin
from gwpopulation.utils import to_numpy

N_TEST = 10
# gradients are slower to evaluate, so they are tested at fewer draws
N_GRADIENT_TEST = 3
N_SAMPLES = 2000

# Hyperparameters are drawn uniformly from these ranges with a seeded generator,
# so that the single precision comparisons are reproducible. The ranges include
# steep power laws, narrow peaks, and steep spin magnitude distributions, so
# that the densities underflow in single precision.
LIMITS = dict(mmin=(3, 10), mmax=(40, 100))
STEEP = dict(alpha=(-30, 30), beta=(-30, 30)) | LIMITS
BROKEN_STEEP = dict(alpha_1=(-30, 30), alpha_2=(-30, 30), beta=(-30, 30)) | LIMITS
BROKEN_STEEP["break_fraction"] = (0.05, 0.95)
# the numerical normalization of the smoothed models needs shallower slopes
POWER = dict(alpha=(-4, 12), beta=(-4, 12)) | LIMITS
BROKEN = dict(alpha_1=(-4, 12), alpha_2=(-4, 12), beta=(-4, 12)) | LIMITS
BROKEN["break_fraction"] = (0.05, 0.95)
PEAK = dict(lam=(0, 1), mpp=(20, 60), sigpp=(1, 10), gaussian_mass_maximum=100)
TWO_PEAKS = dict(
    lam=(0, 1),
    lam_1=(0, 1),
    mpp_1=(20, 60),
    mpp_2=(20, 100),
    sigpp_1=(1, 10),
    sigpp_2=(1, 10),
)
SMOOTH = dict(delta_m=(0, 10))
MASS_NODES = {
    f"mass{ii}": value for ii, value in enumerate(np.geomspace(2, 100, 10))
} | {f"fmass{ii}": (-2, 2) for ii in range(10)}
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


def _spline_nodes(base, low, high, n_nodes=5):
    nodes = {
        f"{base}{ii}": value for ii, value in enumerate(np.linspace(low, high, n_nodes))
    }
    return nodes | {f"f{base}{ii}": (-2, 2) for ii in range(n_nodes)}


def _without(ranges, *keys):
    return {key: value for key, value in ranges.items() if key not in keys}


# points that are outside the support for any of the hyperparameters above
def _outside_mass_ratio(data, parameters):
    return (
        (data["mass_1"] < parameters["mmin"])
        | (data["mass_1"] > 100)
        | (data["mass_ratio"] > 1)
        | (data["mass_2"] < parameters["mmin"])
    )


def _outside_secondary(data, parameters):
    return (
        (data["mass_2"] > data["mass_1"])
        | (data["mass_1"] < parameters["mmin"])
        | (data["mass_1"] > 100)
        | (data["mass_2"] < parameters["mmin"])
    )


def _outside_mass(data, parameters):
    return (data["mass_1"] < parameters["mmin"]) | (data["mass_1"] > 100)


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


def _outside_redshift(data, parameters):
    return data["redshift"] > 2.3


def _nowhere(data, parameters):
    return np.zeros(data["mass_1"].shape, dtype=bool)


class _MassFunction:
    """A function of the primary mass alone and its log, e.g., ``power_law_mass``"""

    def __init__(self, name):
        self.function = getattr(mass, name)
        self.ln_function = getattr(mass, f"ln_{name}")

    def prob(self, data, **parameters):
        return self.function(data["mass_1"], **parameters)

    def ln_prob(self, data, **parameters):
        return self.ln_function(data["mass_1"], **parameters)


def _mass_function(name):
    return lambda: _MassFunction(name)


def _model(component):
    return lambda: Model([component()], cache=False)


# name: (builder of an object with prob and ln_prob, ranges, outside the support)
MODELS = dict(
    power_law_mass=(
        _mass_function("power_law_mass"),
        _without(STEEP, "beta"),
        _outside_mass,
    ),
    two_component_single=(
        _mass_function("two_component_single"),
        _without(STEEP, "beta") | PEAK,
        _outside_mass,
    ),
    three_component_single=(
        _mass_function("three_component_single"),
        _without(STEEP, "beta") | TWO_PEAKS | dict(gaussian_mass_maximum=100),
        _outside_mass,
    ),
    double_power_law_primary_mass=(
        _mass_function("double_power_law_primary_mass"),
        _without(BROKEN_STEEP, "beta"),
        _outside_mass,
    ),
    double_power_law_peak_primary_mass=(
        _mass_function("double_power_law_peak_primary_mass"),
        _without(BROKEN_STEEP, "beta") | PEAK,
        _outside_mass,
    ),
    power_law_primary_mass_ratio=(
        _model(lambda: mass.power_law_primary_mass_ratio),
        STEEP,
        _outside_mass_ratio,
    ),
    power_law_primary_secondary_independent=(
        _model(lambda: mass.power_law_primary_secondary_independent),
        STEEP,
        _outside_secondary,
    ),
    power_law_primary_secondary_identical=(
        _model(lambda: mass.power_law_primary_secondary_identical),
        _without(STEEP, "beta"),
        _outside_secondary,
    ),
    two_component_primary_mass_ratio=(
        _model(lambda: mass.two_component_primary_mass_ratio),
        STEEP | PEAK,
        _outside_mass_ratio,
    ),
    two_component_primary_secondary_independent=(
        _model(lambda: mass.two_component_primary_secondary_independent),
        STEEP | PEAK,
        _outside_secondary,
    ),
    two_component_primary_secondary_identical=(
        _model(lambda: mass.two_component_primary_secondary_identical),
        _without(STEEP, "beta") | PEAK,
        _outside_secondary,
    ),
    double_power_law_primary_power_law_mass_ratio=(
        _model(lambda: mass.double_power_law_primary_power_law_mass_ratio),
        BROKEN_STEEP,
        _outside_mass_ratio,
    ),
    SinglePeakSmoothedMassDistribution=(
        _model(mass.SinglePeakSmoothedMassDistribution),
        POWER | PEAK | SMOOTH,
        _outside_mass_ratio,
    ),
    # narrow peaks below mmax, so that the density underflows above them
    MultiPeakSmoothedMassDistribution=(
        _model(mass.MultiPeakSmoothedMassDistribution),
        POWER
        | TWO_PEAKS
        | SMOOTH
        | dict(mmax=(40, 60), mpp_2=(30, 50), sigpp_1=(1, 3), sigpp_2=(1, 3)),
        _outside_mass_ratio,
    ),
    BrokenPowerLawSmoothedMassDistribution=(
        _model(mass.BrokenPowerLawSmoothedMassDistribution),
        BROKEN | SMOOTH,
        _outside_mass_ratio,
    ),
    BrokenPowerLawPeakSmoothedMassDistribution=(
        _model(mass.BrokenPowerLawPeakSmoothedMassDistribution),
        BROKEN | PEAK | SMOOTH,
        _outside_mass_ratio,
    ),
    InterpolatedPowerlaw=(
        _model(mass.InterpolatedPowerlaw),
        POWER | SMOOTH | MASS_NODES,
        _outside_mass_ratio,
    ),
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
    SplineSpinMagnitudeIdentical=(
        _model(spin.SplineSpinMagnitudeIdentical),
        _spline_nodes("a", 0, 1),
        _nowhere,
    ),
    SplineSpinTiltIdentical=(
        _model(spin.SplineSpinTiltIdentical),
        _spline_nodes("cos_tilt", -1, 1),
        _nowhere,
    ),
    PowerLawRedshift=(
        _model(redshift.PowerLawRedshift),
        dict(lamb=(-15, 15)),
        _outside_redshift,
    ),
    MadauDickinsonRedshift=(
        _model(redshift.MadauDickinsonRedshift),
        dict(gamma=(-15, 15), kappa=(-15, 15), z_peak=(0, 5)),
        _outside_redshift,
    ),
)

# the linear densities of these models do not underflow or overflow in single
# precision for the ranges above
LINEAR_SINGLE_PRECISION = [
    "iid_spin_orientation_gaussian_isotropic",
    "independent_spin_orientation_gaussian_isotropic",
    "SplineSpinMagnitudeIdentical",
    "SplineSpinTiltIdentical",
    "PowerLawRedshift",
    "MadauDickinsonRedshift",
]


def _never(data, parameters, ln_expected):
    return np.zeros(ln_expected.shape, dtype=bool)


def _close_to_mmin(data, parameters, ln_expected):
    """
    The mass ratio normalization of the smoothed models is computed without
    logs, and underflows in single precision for primary masses within
    ~delta_m / 87 of mmin, where the log density is below ~-87.
    """
    return data["mass_1"] < parameters["mmin"] + parameters["delta_m"] / 50


# Single precision tolerances, where points may be lost (lost), and the log
# densities above which points are compared (compare_above). For the smoothed
# models, the smoothing is evaluated at m - mmin, so the error close to mmin
# grows like the square of the log density.
SINGLE_PRECISION = {
    name: dict(rtol=2e-5, atol=2e-5, lost=_never, compare_above=-np.inf)
    for name in MODELS
}
for name in MODELS:
    if "Smoothed" in name or name == "InterpolatedPowerlaw":
        SINGLE_PRECISION[name] = dict(
            rtol=2e-4, atol=2e-4, compare_above=-1000.0, lost=_close_to_mmin
        )


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
    if not isinstance(model, Model):
        return
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
        # the first call caches the interpolants of the spline and smoothed models
        model.ln_prob(dataset, **draws[0])
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


@pytest.mark.parametrize("enable_x64", [True, False])
@pytest.mark.parametrize("name", MODELS)
def test_ln_prob_gradient(name, enable_x64):
    """
    The gradient of the log of the summed density with respect to each
    hyperparameter is finite, although some samples are outside the support,
    e.g., for mixtures, where the gradient of :code:`logaddexp(-inf, -inf)`
    is :code:`nan`. In double precision, it matches the gradient computed
    from the density wherever that is finite.
    """
    jax = pytest.importorskip("jax")
    from gwpopulation.utils import logsumexp

    builder, ranges, _ = MODELS[name]
    fixed = {
        key: value for key, value in ranges.items() if not isinstance(value, tuple)
    }
    gwpopulation.set_backend("jax", enable_x64=enable_x64)
    try:
        xp = gwpopulation.utils.xp
        dtype = xp.float64 if enable_x64 else xp.float32
        dataset = {
            key: xp.asarray(value, dtype=dtype) for key, value in _dataset().items()
        }
        model = builder()
        draws = [
            {key: value for key, value in parameters.items() if key not in fixed}
            for parameters in _draws(name)[:N_GRADIENT_TEST]
        ]
        # the first call caches the interpolants of the spline and smoothed models
        model.ln_prob(dataset, **draws[0], **fixed)

        def ln_total(parameters):
            return logsumexp(model.ln_prob(dataset, **parameters, **fixed))

        def total(parameters):
            return xp.log(xp.sum(model.prob(dataset, **parameters, **fixed)))

        ln_gradient = jax.jit(jax.grad(ln_total))
        gradient = jax.jit(jax.grad(total))
        for parameters in draws:
            ln_result = {
                key: float(value) for key, value in ln_gradient(parameters).items()
            }
            for key, value in ln_result.items():
                assert np.isfinite(value), f"d/d{key} = {value} at {parameters}"
            if not enable_x64:
                continue
            result = {key: float(value) for key, value in gradient(parameters).items()}
            for key, value in result.items():
                if np.isfinite(value):
                    np.testing.assert_allclose(
                        ln_result[key], value, rtol=1e-8, atol=1e-10, err_msg=key
                    )
    finally:
        gwpopulation.set_backend("jax", enable_x64=True)
        gwpopulation.set_backend("numpy")
