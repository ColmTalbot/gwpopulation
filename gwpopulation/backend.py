__all_with_xp = [
    "hyperpe",
    "models.interped",
    "models.mass",
    "models.redshift",
    "models.spin",
    "utils",
    "vt",
]
__all_with_scs = ["models.mass", "utils"]
__backend__ = ""
SUPPORTED_BACKENDS = ["numpy", "cupy"]
_scipy_module = dict(numpy="scipy", cupy="cupyx.scipy")


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


def set_backend(backend="numpy"):
    global __backend__
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Backend {backend} not supported, should be in {', '.join(SUPPORTED_BACKENDS)}"
        )
    elif backend == __backend__:
        return

    from importlib import import_module

    try:
        xp = import_module(backend)
        scs = import_module(_scipy_module[backend]).special
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"{backend} not installed")
    for module in __all_with_xp:
        __backend__ = backend
        import_module(f".{module}", package="gwpopulation").xp = xp
    for module in __all_with_scs:
        import_module(f".{module}", package="gwpopulation").scs = scs
