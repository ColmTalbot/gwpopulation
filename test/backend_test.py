import importlib

import pytest

import gwpopulation


def test_unsupported_backend_raises_value_error():
    with pytest.raises(ValueError):
        gwpopulation.set_backend("fail")


def test_set_backend(backend):
    gwpopulation.set_backend(backend)
    from gwpopulation.utils import xp

    assert xp == importlib.import_module(gwpopulation.backend._np_module[backend])


def test_import_error_caught_for_mangled_install():
    """
    Replace importlib.import_module with a dummy function raise
    the required error.

    Two calls are needed to avoid caching of the backend.

    FIXME: figure out how to replace this with mock
    """

    def _import(module):
        raise ImportError

    gwpopulation.backend.import_module = _import
    with pytest.raises(ImportError):
        gwpopulation.set_backend("numpy")
        gwpopulation.set_backend("jax")

    gwpopulation.backend.import_module = importlib.import_module


def test_loading_arbitrary():
    """
    Test loading arbitrary functions works as we don't have any native
    entry points for them.
    """
    pytest.importorskip("jax")

    from jax.scipy.linalg import toeplitz

    func = gwpopulation.backend._load_arbitrary(
        func="scipy.linalg.toeplitz", backend="jax"
    )
    assert func == toeplitz


def test_set_backend_enable_x64():
    pytest.importorskip("jax")
    import jax

    try:
        gwpopulation.set_backend("jax", enable_x64=False)
        assert not jax.config.jax_enable_x64
        # the default leaves an explicit choice alone, including after
        # switching backend and back
        gwpopulation.set_backend("jax")
        assert not jax.config.jax_enable_x64
        gwpopulation.set_backend("numpy")
        gwpopulation.set_backend("jax")
        assert not jax.config.jax_enable_x64
        gwpopulation.set_backend("jax", enable_x64=True)
        assert jax.config.jax_enable_x64
    finally:
        gwpopulation.set_backend("jax", enable_x64=True)
        gwpopulation.set_backend("numpy")
