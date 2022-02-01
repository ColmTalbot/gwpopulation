"""
GWPopulation
============

A collection of code for doing population inference.

All of this code will run on either CPUs or GPUs using cupy for GPU
acceleration.

This includes:
  - commonly used likelihood functions in the Bilby framework.
  - population models for gravitational-wave sources.
  - selection functions for gravitational-wave sources.

The code is hosted at `<www.github.com/ColmTalbot/gwpopulation>`_.
"""
from .hyperpe import RateLikelihood
from . import conversions, cupy_utils, hyperpe, models, utils, vt

__version__ = utils.get_version_information()

__all_with_xp = [
    models.mass,
    models.redshift,
    models.spin,
    cupy_utils,
    hyperpe,
    utils,
    vt,
]


def disable_cupy():
    import numpy as np

    for module in __all_with_xp:
        module.xp = np


def enable_cupy():
    try:
        import cupy as cp
    except ImportError:
        import numpy as cp

        print("Cannot import cupy, falling back to numpy.")
    for module in __all_with_xp:
        module.xp = cp
