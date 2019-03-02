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

The code is hosted as github.com/ColmTalbot/gwpopulation.
"""


from .hyperpe import RateLikelihood
from . import conversions, cupy_utils, hyperpe, models, utils, vt

__version__ = utils.get_version_information()
