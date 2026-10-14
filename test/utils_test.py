import numpy as np
import pandas as pd
import pytest
from scipy.stats import vonmises

import gwpopulation
from gwpopulation import utils

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


def test_truncnorm_matches_scipy(backend):
    from scipy.stats import truncnorm

    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    xx = xp.linspace(-2, 2, 1000)
    for ii in range(N_TEST):
        mu = np.random.uniform(-10, 10)
        sigma = np.random.uniform(0, 5)
        low, high = np.sort(np.random.uniform(-10, 10, 2))
        gwpop_vals = utils.to_numpy(
            utils.truncnorm(xx, mu=mu, sigma=sigma, low=low, high=high)
        )
        scipy_vals = truncnorm(
            loc=mu, scale=sigma, a=(low - mu) / sigma, b=(high - mu) / sigma
        ).pdf(utils.to_numpy(xx))
        assert max(abs(gwpop_vals - scipy_vals)) < 1e-3


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


def test_trapezoid_matches_scipy(backend):
    from scipy.integrate import trapezoid

    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    x = xp.linspace(0, 10, 1000)
    y = xp.sin(x) + 1.1
    gwpop_int = utils.trapezoid(y, x)
    scipy_int = float(trapezoid(y, x))
    assert abs(gwpop_int - scipy_int) < 1e-10


def _log_matches(ln_vals, vals, xp):
    """ln_vals == log(vals) where vals > 0; where vals underflows to zero
    ln_vals is -inf (outside the support) or below the underflow limit."""
    ln_vals, vals = utils.to_numpy(ln_vals), utils.to_numpy(vals)
    positive = vals > 1e-300
    np.testing.assert_allclose(ln_vals[positive], np.log(vals[positive]), rtol=1e-8)
    assert np.all(ln_vals[vals == 0] < np.log(1e-300))


def test_ln_powerlaw_matches_log_powerlaw(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    xx = xp.linspace(0.1, 120, 1000)
    for alpha in [-1, 0] + list(np.random.uniform(-10, 10, N_TEST)):
        low, high = np.random.uniform(1, 10), np.random.uniform(50, 100)
        _log_matches(
            utils.ln_powerlaw(xx, alpha=alpha, high=high, low=low),
            utils.powerlaw(xx, alpha=alpha, high=high, low=low),
            xp,
        )
    gwpopulation.set_backend("numpy")


def test_ln_powerlaw_array_low(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    xx = xp.linspace(0.01, 1, 100)
    low = xp.linspace(0.005, 0.5, 100)
    _log_matches(
        utils.ln_powerlaw(xx, alpha=2.5, high=1, low=low),
        utils.powerlaw(xx, alpha=2.5, high=1, low=low),
        xp,
    )
    gwpopulation.set_backend("numpy")


def test_ln_powerlaw_zero_low():
    gwpopulation.set_backend("numpy")
    xx = np.linspace(0, 1, 11)
    assert np.allclose(utils.ln_powerlaw(xx, alpha=0, high=1, low=0), 0)
    assert np.allclose(
        utils.ln_powerlaw(xx[1:], alpha=2, high=1, low=0), np.log(3 * xx[1:] ** 2)
    )


def test_ln_truncnorm_matches_log_truncnorm(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    xx = xp.linspace(-20, 20, 1000)
    for _ in range(N_TEST):
        mu = np.random.uniform(-10, 10)
        sigma = np.random.uniform(0.1, 5)
        low, high = np.sort(np.random.uniform(-15, 15, 2))
        _log_matches(
            utils.ln_truncnorm(xx, mu=mu, sigma=sigma, high=high, low=low),
            utils.truncnorm(xx, mu=mu, sigma=sigma, high=high, low=low),
            xp,
        )
    gwpopulation.set_backend("numpy")


def test_ln_functions_single_precision_jax():
    pytest.importorskip("jax")
    gwpopulation.set_backend("jax", enable_x64=False)
    xp = gwpopulation.utils.xp
    try:
        xx = xp.linspace(2, 200, 1000, dtype=xp.float32)
        ln_pow = utils.ln_powerlaw(xx, alpha=-12.0, high=150.0, low=3.0)
        # far tail of a narrow peak, where the linear density underflows
        ln_norm = utils.ln_truncnorm(xx, mu=20.0, sigma=0.1, high=150.0, low=3.0)
        for ln_vals in (ln_pow, ln_norm):
            assert ln_vals.dtype == xp.float32
            assert not xp.any(xp.isnan(ln_vals))
        inside = (xx >= 3) & (xx <= 150)
        assert xp.all(xp.isfinite(ln_pow[inside]))
        assert xp.all(xp.isfinite(ln_norm[inside]))
        assert xp.all(xp.isneginf(ln_norm[~inside]))
        expected = -((np.asarray(xx, dtype=float) - 20) ** 2) / 2 / 0.01
        expected -= np.log(0.1) + np.log(2 * np.pi) / 2
        np.testing.assert_allclose(
            np.asarray(ln_norm)[np.asarray(inside)],
            expected[np.asarray(inside)],
            rtol=1e-5,
        )
    finally:
        gwpopulation.set_backend("jax", enable_x64=True)
        gwpopulation.set_backend("numpy")
