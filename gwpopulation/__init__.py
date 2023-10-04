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
from . import conversions, hyperpe, models, utils, vt
from .backend import SUPPORTED_BACKENDS, disable_cupy, enable_cupy, set_backend
from .hyperpe import RateLikelihood

try:
    from ._version import __version__
except ModuleNotFoundError:  # development mode
    __version__ = "unknown"

try:
    set_backend("cupy")
except ImportError:
    set_backend("numpy")
