try:
    import cupy as xp
    from .cupy_utils import trapz
    CUPY_LOADED = True
except ImportError:
    import numpy as xp
    from numpy import trapz
    CUPY_LOADED = False

from .hyperpe import RateLikelihood
from . import conversions, hyperpe, models, vt
