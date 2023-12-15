import importlib

import numpy
import pytest

import gwpopulation

from . import TEST_BACKENDS


def test_unsupported_backend_raises_value_error():
    with pytest.raises(ValueError):
        gwpopulation.set_backend("fail")


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_set_backend(backend):
    gwpopulation.set_backend(backend)
    from gwpopulation.utils import xp

    print(xp)
    print(gwpopulation.backend._np_module[backend])
    assert xp == importlib.import_module(gwpopulation.backend._np_module[backend])


def test_enable_cupy_deprecated():
    with pytest.deprecated_call():
        try:
            gwpopulation.enable_cupy()
        except ImportError:
            pass


def test_disable_cupy_deprecated():
    with pytest.deprecated_call():
        gwpopulation.disable_cupy()
