import numpy
import pytest

import gwpopulation


def test_unsupported_backend_raises_value_error():
    with pytest.raises(ValueError):
        gwpopulation.set_backend("fail")


def test_set_backend_numpy():
    gwpopulation.set_backend("numpy")
    from gwpopulation.utils import xp

    assert xp == numpy


def test_enable_cupy_deprecated():
    with pytest.deprecated_call():
        try:
            gwpopulation.enable_cupy()
        except ModuleNotFoundError:
            pass


def test_disable_cupy_deprecated():
    with pytest.deprecated_call():
        gwpopulation.disable_cupy()
