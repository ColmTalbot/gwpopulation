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
import sys

if sys.version_info < (3,):  # noqa
    raise ImportError(
        """You are running GWPopulation 0.5.0 or higher on Python 2

GWPopulation 0.5.0 and above are no longer compatible with Python 2, and you
still ended up with this version installed. That's unfortunate; sorry about
that. It should not have happened. Make sure you have pip >= 9.0 to avoid this
kind of issue, as well as setuptools >= 24.2:

 $ pip install pip setuptools --upgrade

Your choices:

- Upgrade to Python 3.

- Install an older version of GWPopulation:

 $ pip install 'gwpopulation<0.5.0'

It would be great if you can figure out how this version ended up being
installed, and try to check how to prevent that for future users.

"""
    )

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
