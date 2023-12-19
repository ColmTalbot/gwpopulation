"""
GWPopulation
============

A collection of code for doing population inference with the
gravitational-wave transient catalogue.

This includes:
  - commonly used likelihood functions in the Bilby framework.
  - population models for gravitational-wave sources.
  - selection functions for gravitational-wave sources.

:code:`GWPopulation` supports multiple numpy-like backends, including
:code:`numpy`, :code:`jax`, and :code:`cupy`. The :code:`jax` and
:code:`cupy` backends allow for GPU acceleration.

The code is hosted at `<www.github.com/ColmTalbot/gwpopulation>`_ and
available via :code:`conda-forge` and :code:`pypi`.
"""
from . import conversions, hyperpe, models, utils, vt
from ._version import __version__
from .backend import SUPPORTED_BACKENDS, set_backend
