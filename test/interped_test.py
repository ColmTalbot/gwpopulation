import functools
import importlib

import numpy as np
import pytest

from gwpopulation.backend import _np_module, set_backend
from gwpopulation.models.interped import (
    InterpolatedNoBaseModelIdentical,
    _setup_interpolant,
)

from . import TEST_BACKENDS


@pytest.mark.parametrize("regularize", [True, False])
def test_regularization_works(regularize):
    model = InterpolatedNoBaseModelIdentical(
        parameters=["a_1"], minimum=0, maximum=1, regularize=regularize
    )
    parameters = {key: value for key, value in zip(model.xkeys, np.linspace(0, 1, 10))}
    parameters.update({key: 1.0 for key in model.fkeys})
    if regularize:
        parameters["rmsa"] = 0.1
    values = model.extract_spline_points(parameters)[0]
    if regularize:
        expected = 0.1
    else:
        expected = 1
    assert abs((values**2).mean() ** 0.5 - expected) < 1e-10


def test_regularize_option_adds_rms_variable():
    for regularize in [True, False]:
        model = InterpolatedNoBaseModelIdentical(
            parameters=["a_1"], minimum=0, maximum=1, regularize=regularize
        )
        assert ("rmsa" in model.variable_names) is regularize


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_setup_interpolant_obeys_backend(backend):
    xp = importlib.import_module(_np_module[backend])
    interpolant = _setup_interpolant(
        nodes=np.linspace(0, 1, 5),
        values=np.linspace(0, 1, 1000),
        backend=xp,
    )
    assert isinstance(interpolant.func.conversion, xp.ndarray)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_setup_interpolant_obeys_backend_implicit(backend):
    xp = importlib.import_module(_np_module[backend])
    set_backend(backend)
    interpolant = _setup_interpolant(
        nodes=np.linspace(0, 1, 5),
        values=np.linspace(0, 1, 1000),
    )
    assert isinstance(interpolant.func.conversion, xp.ndarray)
