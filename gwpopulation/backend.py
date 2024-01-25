from importlib import import_module

__backend__ = ""
SUPPORTED_BACKENDS = ["numpy", "cupy", "jax"]
_np_module = dict(numpy="numpy", cupy="cupy", jax="jax.numpy")
_scipy_module = dict(numpy="scipy", cupy="cupyx.scipy", jax="jax.scipy")


__doc__ = f"""
:code:`GWPopulation` provides a unified interface to a number of :code:`numpy/scipy` like APIs.

The backend can be set using :code:`gwpopulation.set_backend(backend)`, where
:code:`backend` is one of :code:`{', '.join(SUPPORTED_BACKENDS)}`.

Downstream packages can automatically track the active backend using :code:`entry_points`.
With this set up, packages can use :code:`xp` and :code:`scs` in specified modules.

..note::
    Each module that wants to use the :code:`GWPopulation` backend must be specified independently
    for the automatic propagation to work.

If there is a backend that you would like to use that is not currently supported, please open an issue.
"""


def modules_to_update():
    import sys

    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points
    all_with_xp = [
        ".hyperpe",
        ".models.interped",
        ".models.mass",
        ".models.redshift",
        ".models.spin",
        ".utils",
        ".vt",
    ]
    all_with_xp.extend(
        [module.value for module in entry_points(group="gwpopulation.xp")]
    )
    all_with_scs = [".models.mass", ".utils"]
    all_with_scs.extend(
        [module.value for module in entry_points(group="gwpopulation.scs")]
    )
    return all_with_xp, all_with_scs


def disable_cupy():
    from warnings import warn

    warn(
        f"Function enable_cupy is deprecated, use set_backed('cupy') instead",
        DeprecationWarning,
    )
    set_backend(backend="numpy")


def enable_cupy():
    from warnings import warn

    warn(
        f"Function enable_cupy is deprecated, use set_backed('cupy') instead",
        DeprecationWarning,
    )
    set_backend(backend="cupy")


def _configure_jax(xp):
    from jax import config
    from jax.scipy.integrate import trapezoid

    config.update("jax_enable_x64", True)
    xp.trapz = trapezoid


def _load_numpy_and_scipy(backend):
    try:
        xp = import_module(_np_module[backend])
        scs = import_module(_scipy_module[backend]).special
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"{backend} not installed for gwpopulation")
    except ImportError:
        raise ImportError(f"{backend} installed but not importable for gwpopulation")

    if backend == "jax":
        _configure_jax(xp)

    return xp, scs


def _set_in_module(module, name, value):
    if module.startswith("."):
        package = "gwpopulation"
    else:
        package = None
    setattr(import_module(module, package=package), name, value)


def set_backend(backend="numpy"):
    global __backend__
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Backend {backend} not supported, should be in {', '.join(SUPPORTED_BACKENDS)}"
        )
    elif backend == __backend__:
        return

    xp, scs = _load_numpy_and_scipy(backend)

    __backend__ = backend
    all_with_xp, all_with_scs = modules_to_update()
    for module in all_with_xp:
        _set_in_module(module, "xp", xp)
    for module in all_with_scs:
        _set_in_module(module, "scs", scs)
