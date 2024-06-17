"""
Helper functions for probability distributions and backend switching.
"""

from numbers import Number
from operator import ge, gt, ne

import numpy as np
from scipy import special as scs

xp = np


def apply_conditions(conditions):
    """
    A decorator to apply conditions to inputs of a function.

    Parameters
    ==========
    func: callable
        The function to decorate.
    conditions: dict
        A dictionary of conditions to apply to the function. The keys are the
        argument names and the values are the conditions. The conditions can be
        should be in the form (op, value) where op is a comparison operator and
        value is the value to compare to. The conditions can also be a callable
        which takes the value as an argument and returns a boolean. The variable
        must be specified as a keyword argument for the test to be applied.
    """
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            if "jax" in xp.__name__:
                return func(*args, **kwargs)
            for key, condition in conditions.items():
                if key in kwargs:
                    value = kwargs[key]
                else:
                    continue
                if callable(condition):
                    if not condition(value):
                        raise ValueError(f"Condition {condition} not met")
                else:
                    op, val = condition
                    if "cupy" in xp.__name__:
                        value = xp.asarray(value)
                    if callable(op):
                        if not xp.all(op(value, val)):
                            raise ValueError(
                                f"{key}: {value} does not satisfy {op.__name__}"
                            )
                    else:
                        raise ValueError(f"Operator {op} not supported")
            return func(*args, **kwargs)

        return wrapped_function

    return decorator


@apply_conditions(dict(alpha=(gt, 0), beta=(gt, 0), scale=(gt, 0)))
def beta_dist(xx, alpha, beta, scale=1):
    r"""
    Beta distribution probability

    .. math::
        p(x) = \frac{x^{\alpha - 1} (x_\max - x)^{\beta - 1}}{B(\alpha, \beta) x_\max^{\alpha + \beta + 1}}

    Parameters
    ----------
    xx: float, array-like
        The abscissa values (:math:`x`)
    alpha: float
        The Beta alpha parameter (:math:`\alpha`)
    beta: float
        The Beta beta parameter (:math:`\beta`)
    scale: float, array-like
        A scale factor for the distribution of the distribution (:math:`x_\max`)

    Returns
    -------
    prob: float, array-like
        The distribution evaluated at `xx`

    """
    ln_beta = (alpha - 1) * xp.log(xx) + (beta - 1) * xp.log(scale - xx)
    ln_beta -= scs.betaln(alpha, beta)
    ln_beta -= (alpha + beta - 1) * xp.log(scale)
    prob = xp.exp(ln_beta)
    prob = xp.nan_to_num(prob)
    prob *= (xx >= 0) * (xx <= scale)
    return prob


@apply_conditions(dict(low=(ge, 0), alpha=(ne, 1)))
def powerlaw(xx, alpha, high, low):
    r"""
    Power-law probability

    .. math::
        p(x) = \frac{1 + \alpha}{x_\max^{1 + \alpha} - x_\min^{1 + \alpha}} x^\alpha

    Parameters
    ----------
    xx: float, array-like
        The abscissa values (:math:`x`)
    alpha: float, array-like
        The spectral index of the distribution (:math:`\alpha`)
    high: float, array-like
        The maximum of the distribution (:math:`x_\min`)
    low: float, array-like
        The minimum of the distribution (:math:`x_\max`)

    Returns
    -------
    prob: float, array-like
        The distribution evaluated at `xx`

    """
    norm = xp.where(
        xp.array(alpha) == -1,
        1 / xp.log(high / low),
        (1 + alpha) / xp.array(high ** (1 + alpha) - low ** (1 + alpha)),
    )
    prob = xp.power(xx, alpha)
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


@apply_conditions(dict(sigma=(gt, 0)))
def truncnorm(xx, mu, sigma, high, low):
    r"""
    Truncated normal probability

    .. math::

        p(x) =
        \sqrt{\frac{2}{\pi\sigma^2}}\frac{\exp\left(-\frac{(\mu - x)^2}{2 \sigma^2}\right)}
        {\text{erf}\left(\frac{x_\max - \mu}{\sqrt{2}}\right) + \text{erf}\left(\frac{\mu - x_\min}{\sqrt{2}}\right)}

    Parameters
    ----------
    xx: float, array-like
        The abscissa values (:math:`x`)
    mu: float, array-like
        The mean of the normal distribution (:math:`\mu`)
    sigma: float
        The standard deviation of the distribution (:math:`\sigma`)
    high: float, array-like
        The maximum of the distribution (:math:`x_\min`)
    low: float, array-like
        The minimum of the distribution (:math:`x_\max`)

    Returns
    -------
    prob: float, array-like
        The distribution evaluated at `xx`

    """
    norm = 2**0.5 / xp.pi**0.5 / sigma
    norm /= scs.erf((high - mu) / 2**0.5 / sigma) + scs.erf(
        (mu - low) / 2**0.5 / sigma
    )
    prob = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma**2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


def unnormalized_2d_gaussian(xx, yy, mu_x, mu_y, sigma_x, sigma_y, covariance):
    r"""
    Compute the probability distribution for a correlated 2-dimensional Gaussian
    neglecting normalization terms.

    .. math::
        \ln p(x) &= (x - \mu_{x} y - \mu_{y}) \Sigma^{-1} (x - \mu_x y - \mu_{y})^{T} \\
        \Sigma &= \begin{bmatrix}
                \sigma^{2}_{x} & \rho \sigma_{x} \sigma_{y} \\
                \rho \sigma_{x} \sigma_{y} & \sigma^{2}_{y}
            \end{bmatrix}

    Parameters
    ----------
    xx: array-like
        Input data in first dimension (:math:`x`)
    yy: array-like
        Input data in second dimension (:math:`y`)
    mu_x: float
        Mean in the first dimension (:math:`\mu_{x}`)
    sigma_x: float
        Standard deviation in the first dimension (:math:`\sigma_{x}`)
    mu_y: float
        Mean in the second dimension (:math:`\mu_{y}`)
    sigma_y: float
        Standard deviation in the second dimension (:math:`\sigma_{y}`)
    covariance: float
        The normalized covariance (bounded in [0, 1]) (:math:`\rho`)

    Returns
    -------
    prob: array-like
        The unnormalized probability distribution (:math:`p(x)`)
    """
    determinant = sigma_x**2 * sigma_y**2 * (1 - covariance)
    residual_x = (mu_x - xx) * sigma_y
    residual_y = (mu_y - yy) * sigma_x
    prob = xp.exp(
        -(residual_x**2 + residual_y**2 - 2 * residual_x * residual_y * covariance)
        / 2
        / determinant
    )
    return prob


def von_mises(xx, mu, kappa):
    r"""
    PDF of the von Mises distribution defined on the standard interval.

    .. math::
        p(x) =
        \frac{\exp\left( \kappa \cos(x - \mu) \right)}{2 \pi \mathcal{I}_{0}(\kappa)}

    Parameters
    ----------
    xx: array-like
        Input points at which to evaluate the distribution
    mu: float
        The mean of the distribution
    kappa: float
        The scale parameter

    Returns
    -------
    array-like
        The probability

    Notes
    -----
    For numerical stability, the factor of :math:`\exp(\kappa)` from using
    :func:`scipy.special.i0e` is accounted for in the numerator.
    """
    return xp.exp(kappa * (xp.cos(xx - mu) - 1)) / (2 * xp.pi * scs.i0e(kappa))


def get_version_information():
    """
    .. deprecated:: 1.2.0

    Get the version of :code:`gwpopulation`.
    Use :code:`importlib.metadata.version("gwpopulation")` instead.

    """
    from gwpopulation import __version__

    return __version__


def get_name(input):
    """
    Attempt to find the name of the the input. This either returns
    :code:`input.__name__` or :code:`input.__class__.__name__`

    Parameters
    ==========
    input: any
        The input to find the name for.

    Returns
    =======
    str: The name of the input.
    """
    if hasattr(input, "__name__"):
        return input.__name__
    else:
        return input.__class__.__name__


def to_number(value, func):
    """
    Convert a zero-dimensional array to a number.

    Parameters
    ==========
    value: array-like
        The zero-dimensional array to convert.
    func: callable
        The function to convert the array with,
        e.g., :code:`int,float,complex`.
    """
    if "jax" in xp.__name__:
        return value.astype(func)
    else:
        return func(value)


def to_numpy(array):
    """
    Convert an array to a numpy array.
    Numeric types and pandas objects are returned unchanged.

    Parameters
    ==========
    array: array-like
        The array to convert.
    """
    if isinstance(array, (Number, np.ndarray)):
        return array
    elif "cupy" in array.__class__.__module__:
        return xp.asnumpy(array)
    elif "pandas" in array.__class__.__module__:
        return array
    elif "jax" in array.__class__.__module__:
        return np.asarray(array)
    else:
        raise TypeError(f"Cannot convert {type(array)} to numpy array")
