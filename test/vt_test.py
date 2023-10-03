import numpy as np
import pytest
from bilby.hyper.model import Model

import gwpopulation
from gwpopulation import vt
from gwpopulation.models.redshift import PowerLawRedshift, total_four_volume

from . import TEST_BACKENDS

xp = np


def dummy_function(dataset, alpha):
    return 1


def test_initialize_with_function():
    evaluator = vt._BaseVT(model=dummy_function, data=dict())
    assert evaluator.model.models == [dummy_function]


def test_initialize_with_list_of_functions():
    evaluator = vt._BaseVT(model=[dummy_function, dummy_function], data=dict())
    assert evaluator.model.models == [dummy_function, dummy_function]


def test_initialize_with_bilby_model():
    model = Model([dummy_function, dummy_function])
    evaluator = vt._BaseVT(model=model, data=dict())
    assert evaluator.model.models == [dummy_function, dummy_function]


def test_base_cannot_be_called():
    model = Model([dummy_function, dummy_function])
    evaluator = vt._BaseVT(model=model, data=dict())
    with pytest.raises(NotImplementedError):
        evaluator()


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_grid_vt_correct(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    model = lambda dataset: xp.ones_like(dataset["a"])
    data = dict(a=xp.linspace(0, 1, 1000), vt=2)
    assert float(vt.GridVT(model, data)(dict())) == 2


def resampling_data(xp):
    return dict(a=xp.linspace(0, 1, 1000), prior=xp.ones(1000), vt=2)


def get_vt(xp):
    data = resampling_data(xp)
    model = lambda dataset: (
        xp.exp(-((dataset["a"] - 0.5) ** 2) / 2) / (2 * xp.pi) ** 0.5
    )
    evaluator = vt.ResamplingVT(data=data, model=model, n_events=0)
    return evaluator


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_marginalized_vt_correct(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    assert abs(float(get_vt(xp).vt_factor(dict())) - 0.38289403358409585) < 1e-6


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_resampling_vt_correct(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    assert abs(float(get_vt(xp)(dict())[0]) - 0.38289325179141254) < 1e-6


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_returns_inf_when_n_effective_too_small(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    evaluator = get_vt(xp)
    evaluator.marginalize_uncertainty = True
    evaluator.n_events = xp.inf
    assert evaluator(dict()) == xp.inf


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_observed_volume_with_no_redshift_model(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    evaluator = get_vt(xp)
    assert evaluator.surveyed_hypervolume(dict()) == total_four_volume(
        lamb=0, analysis_time=1
    )


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_observed_volume_with_redshift_model(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    data = resampling_data(xp)
    model = PowerLawRedshift()
    evaluator = vt.ResamplingVT(data=data, model=model, n_events=0)
    assert (
        abs(
            evaluator.surveyed_hypervolume(dict(lamb=4))
            - total_four_volume(lamb=4, analysis_time=1)
        )
        < 1e-4
    )
