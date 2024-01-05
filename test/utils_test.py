import numpy as np
import pandas as pd
import pytest
from scipy.stats import vonmises

import gwpopulation
from gwpopulation import utils

from . import TEST_BACKENDS

N_TEST = 100


def test_beta_dist_zero_below_zero():
    equal_zero = True
    for _ in range(N_TEST):
        vals = utils.beta_dist(-1, *np.random.uniform(0, 10, 3))
        equal_zero = equal_zero & (vals == 0.0)
    assert equal_zero


def test_beta_dist_zero_above_scale():
    equal_zero = True
    for _ in range(N_TEST):
        vals = utils.beta_dist(20, *np.random.uniform(0, 10, 3))
        equal_zero = equal_zero & (vals == 0.0)
    assert equal_zero


def test_beta_dist_alpha_below_zero_raises_value_error():
    gwpopulation.set_backend("numpy")
    with pytest.raises(ValueError):
        utils.beta_dist(xx=0.5, alpha=-1, beta=1, scale=1)


def test_beta_dist_beta_below_zero_raises_value_error():
    gwpopulation.set_backend("numpy")
    with pytest.raises(ValueError):
        utils.beta_dist(xx=0.5, alpha=1, beta=-1, scale=1)


def test_powerlaw_zero_below_low():
    equal_zero = True
    for ii in range(N_TEST):
        vals = utils.powerlaw(
            xx=1,
            alpha=np.random.uniform(-10, 10),
            low=np.random.uniform(5, 15),
            high=np.random.uniform(20, 30),
        )
        equal_zero = equal_zero & (vals == 0.0)
    assert equal_zero


def test_powerlaw_zero_above_high():
    equal_zero = True
    for ii in range(N_TEST):
        vals = utils.powerlaw(
            xx=40,
            alpha=np.random.uniform(-10, 10),
            low=np.random.uniform(5, 15),
            high=np.random.uniform(20, 30),
        )
        equal_zero = equal_zero & (vals == 0.0)
    assert equal_zero


def test_powerlaw_low_below_zero_raises_value_error():
    with pytest.raises(ValueError):
        utils.powerlaw(xx=0, alpha=3, high=10, low=-4)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_powerlaw_alpha_equal_zero(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    assert utils.powerlaw(xx=1.0, alpha=-1, low=0.5, high=2) == 1 / xp.log(4)
    gwpopulation.set_backend("numpy")


def test_truncnorm_zero_below_low():
    equal_zero = True
    for _ in range(N_TEST):
        vals = utils.truncnorm(
            -40,
            mu=np.random.uniform(-10, 10),
            sigma=np.random.uniform(0, 10),
            low=np.random.uniform(-30, -20),
            high=np.random.uniform(20, 30),
        )
        equal_zero = equal_zero & (vals == 0.0)
    assert equal_zero


def test_truncnorm_zero_above_high():
    equal_zero = True
    for _ in range(N_TEST):
        vals = utils.truncnorm(
            40,
            mu=np.random.uniform(-10, 10),
            sigma=np.random.uniform(0, 10),
            low=np.random.uniform(-30, -20),
            high=np.random.uniform(20, 30),
        )
        equal_zero = equal_zero & (vals == 0.0)
    assert equal_zero


def test_truncnorm_sigma_below_zero_raises_value_error():
    with pytest.raises(ValueError):
        utils.truncnorm(xx=0, mu=0, sigma=-1, high=10, low=-10)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_matches_scipy(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    xx = xp.linspace(0, 2 * np.pi, 1000)
    for ii in range(N_TEST):
        mu = np.random.uniform(-np.pi, np.pi)
        kappa = np.random.uniform(0, 15)
        gwpop_vals = utils.to_numpy(utils.von_mises(xx, mu, kappa))
        scipy_vals = vonmises(kappa=kappa, loc=mu).pdf(utils.to_numpy(xx))
        assert max(abs(gwpop_vals - scipy_vals)) < 1e-3


def test_to_numpy_leaves_pandas_changed():
    test = pd.Series([1, 2, 3])
    assert type(test) == type(utils.to_numpy(test))


def test_to_numpy_non_array_type_raises_error():
    with pytest.raises(TypeError):
        utils.to_numpy([1, 2, 3])


def test_get_version():
    assert gwpopulation.__version__ == utils.get_version_information()


@gwpopulation.utils.apply_conditions(dict(a=lambda a: a > 0))
def _condition_func(a):
    return a


def test_callable_condition_not_satisfied():
    gwpopulation.set_backend("numpy")
    with pytest.raises(ValueError):
        _condition_func(a=-1)


def test_callable_condition_satisfied():
    gwpopulation.set_backend("numpy")
    assert _condition_func(a=1) == 1


def test_non_callable_op_raises_error():
    gwpopulation.set_backend("numpy")

    @gwpopulation.utils.apply_conditions(dict(a=("ge", 0)))
    def _condition_func(a):
        return

    with pytest.raises(ValueError):
        _condition_func(a=1)
