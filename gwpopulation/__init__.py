"""
:code:`GWPopulation` is a Python package for doing population inference
with the gravitational-wave transient catalogue supporting a range of 
numpy-like backends.

This includes:

- commonly used likelihood functions in the Bilby framework.
- population models for gravitational-wave sources.
- selection functions for gravitational-wave sources.
- spectral siren cosmological inference with flat wCDM models.
"""

from . import conversions, hyperpe, models, utils, vt
from ._version import __version__
from .backend import SUPPORTED_BACKENDS, set_backend
