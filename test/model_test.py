import numpy as np
import pytest
from bilby.core.prior import PriorDict, Uniform

import gwpopulation
from gwpopulation.hyperpe import Model
from gwpopulation.models import mass, spin
from gwpopulation.utils import to_numpy

N_TEST = 5


def _prior():
    prior = PriorDict()
    prior["alpha"] = Uniform(minimum=-4, maximum=12)
    prior["beta"] = Uniform(minimum=-4, maximum=12)
    prior["mmin"] = Uniform(minimum=3, maximum=10)
    prior["mmax"] = Uniform(minimum=40, maximum=100)
    prior["lam"] = Uniform(minimum=0, maximum=1)
    prior["mpp"] = Uniform(minimum=20, maximum=60)
    prior["sigpp"] = Uniform(minimum=1, maximum=10)
    prior["xi_spin"] = Uniform(minimum=0, maximum=1)
    prior["sigma_spin"] = Uniform(minimum=0.1, maximum=4)
    prior["gaussian_mass_maximum"] = 80
    return prior


def _dataset(xp, dtype=None):
    rng = np.random.default_rng(42)
    dataset = dict(
        mass_1=rng.uniform(2, 120, 1000),
        mass_ratio=rng.uniform(0, 1, 1000),
        cos_tilt_1=rng.uniform(-1, 1, 1000),
        cos_tilt_2=rng.uniform(-1, 1, 1000),
    )
    return {key: xp.asarray(value, dtype=dtype) for key, value in dataset.items()}


def _linear_only(dataset, xi_spin, sigma_spin):
    return spin.iid_spin_orientation_gaussian_isotropic(
        dataset, xi_spin=xi_spin, sigma_spin=sigma_spin
    )


class _ClassModel:
    variable_names = ["xi_spin", "sigma_spin"]

    def __init__(self):
        self.n_ln_calls = 0

    def __call__(self, dataset, **kwargs):
        return _linear_only(dataset, **kwargs)

    def ln_prob(self, dataset, **kwargs):
        self.n_ln_calls += 1
        return gwpopulation.utils.xp.log(_linear_only(dataset, **kwargs))


def test_ln_function_resolution():
    model = Model([mass.two_component_primary_mass_ratio, _linear_only])
    assert (
        model.ln_function(mass.two_component_primary_mass_ratio)
        is mass.ln_two_component_primary_mass_ratio
    )
    assert model.ln_function(_linear_only).__name__ == "ln_function"
    instance = _ClassModel()
    assert Model([instance]).ln_function(instance) == instance.ln_prob


def test_ln_function_attribute():
    def linear(dataset, xi_spin):
        return xi_spin * gwpopulation.utils.xp.ones(dataset["mass_1"].shape)

    def ln_linear(dataset, xi_spin):
        return gwpopulation.utils.xp.log(xi_spin) * gwpopulation.utils.xp.ones(
            dataset["mass_1"].shape
        )

    linear.ln_prob = ln_linear
    model = Model([linear])
    assert model.ln_function(linear) is ln_linear
    np.testing.assert_allclose(
        model.ln_prob(_dataset(np), xi_spin=0.5), np.log(0.5) * np.ones(1000)
    )


def test_ln_prob_matches_log_prob(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    dataset = _dataset(xp)
    prior = _prior()
    model = Model(
        [mass.two_component_primary_mass_ratio, _linear_only, _ClassModel()],
        cache=False,
    )
    for _ in range(N_TEST):
        parameters = prior.sample()
        prob = to_numpy(model.prob(dataset, **parameters))
        ln_prob = to_numpy(model.ln_prob(dataset, **parameters))
        positive = prob > 1e-300
        assert np.any(positive)
        np.testing.assert_allclose(ln_prob[positive], np.log(prob[positive]), rtol=1e-8)
        assert np.all(ln_prob[prob == 0] < np.log(1e-300))
        assert not np.any(np.isnan(ln_prob))
    gwpopulation.set_backend("numpy")


def test_prob_matches_bilby_model():
    from bilby.hyper.model import Model as BilbyModel

    dataset = _dataset(np)
    parameters = _prior().sample()
    functions = [mass.two_component_primary_mass_ratio, _linear_only]
    np.testing.assert_array_equal(
        Model(functions).prob(dataset, **parameters),
        BilbyModel(functions).prob(dataset, **parameters),
    )


def test_as_ln_model():
    from bilby.hyper.model import Model as BilbyModel

    from gwpopulation.hyperpe import as_ln_model

    functions = [mass.two_component_primary_mass_ratio, _linear_only]
    bilby_model = BilbyModel(functions, cache=False)
    converted = as_ln_model(bilby_model)
    assert type(converted) is Model
    assert converted.models == functions
    assert not converted.cache
    assert converted.parameters is bilby_model.parameters

    class Subclass(BilbyModel):
        pass

    for unchanged in [Model(functions), Subclass(functions), _linear_only]:
        assert as_ln_model(unchanged) is unchanged


def test_ln_prob_caching_separate_from_prob():
    dataset = _dataset(np)
    component = _ClassModel()
    model = Model([mass.two_component_primary_mass_ratio, component])
    parameters = _prior().sample()

    prob = model.prob(dataset, **parameters)
    assert component.n_ln_calls == 0
    assert all(value is None for value in model._cached_ln_probability.values())

    ln_prob = model.ln_prob(dataset, **parameters)
    assert component.n_ln_calls == 1
    np.testing.assert_allclose(ln_prob[prob > 0], np.log(prob[prob > 0]))

    model.ln_prob(dataset, **parameters)
    assert component.n_ln_calls == 1
    np.testing.assert_array_equal(model.prob(dataset, **parameters), prob)

    model.ln_prob(dataset, **_prior().sample())
    assert component.n_ln_calls == 2
    for function in model.models:
        assert model._cached_ln_probability[function] is not (
            model._cached_probability[function]
        )


def test_ln_prob_no_cache():
    dataset = _dataset(np)
    component = _ClassModel()
    model = Model([component], cache=False)
    parameters = _prior().sample()
    model.ln_prob(dataset, **parameters)
    model.ln_prob(dataset, **parameters)
    assert component.n_ln_calls == 2
    assert model._cached_ln_probability[component] is None


def test_ln_prob_jit():
    jax = pytest.importorskip("jax")
    gwpopulation.set_backend("jax")
    try:
        xp = gwpopulation.utils.xp
        dataset = _dataset(xp)
        model = Model(
            [mass.two_component_primary_mass_ratio, _linear_only], cache=False
        )
        parameters = {key: float(value) for key, value in _prior().sample().items()}
        jitted = jax.jit(lambda parameters: model.ln_prob(dataset, **parameters))
        np.testing.assert_allclose(
            to_numpy(jitted(parameters)),
            to_numpy(model.ln_prob(dataset, **parameters)),
            rtol=1e-12,
        )
    finally:
        gwpopulation.set_backend("numpy")


def test_ln_prob_single_precision_jax():
    pytest.importorskip("jax")
    import jax

    prior = _prior()
    samples = [prior.sample() for _ in range(N_TEST)]
    functions = [mass.two_component_primary_mass_ratio, _linear_only]
    expected = [
        Model(functions).ln_prob(_dataset(np), **parameters) for parameters in samples
    ]
    gwpopulation.set_backend("jax", enable_x64=False)
    try:
        xp = gwpopulation.utils.xp
        dataset = _dataset(xp, dtype=xp.float32)
        model = Model(functions, cache=False)
        jitted = jax.jit(lambda parameters: model.ln_prob(dataset, **parameters))
        for parameters, ln_expected in zip(samples, expected):
            parameters = {key: float(value) for key, value in parameters.items()}
            ln_prob = jitted(parameters)
            assert ln_prob.dtype == xp.float32
            ln_prob = to_numpy(ln_prob)
            assert not np.any(np.isnan(ln_prob))
            finite = np.isfinite(ln_expected)
            assert np.all(np.isfinite(ln_prob) == finite)
            np.testing.assert_allclose(
                ln_prob[finite], ln_expected[finite], rtol=1e-4, atol=1e-3
            )
    finally:
        gwpopulation.set_backend("jax", enable_x64=True)
        gwpopulation.set_backend("numpy")
