"""
Helper functions for probability distributions.
"""

import os

from .cupy_utils import erf, betaln, xp


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
    if alpha < 0:
        raise ValueError(f"Parameter alpha must be greater or equal zero, low={alpha}.")
    if beta < 0:
        raise ValueError(f"Parameter beta must be greater or equal zero, low={beta}.")
    ln_beta = (alpha - 1) * xp.log(xx) + (beta - 1) * xp.log(scale - xx)
    ln_beta -= betaln(alpha, beta)
    ln_beta -= (alpha + beta - 1) * xp.log(scale)
    prob = xp.exp(ln_beta)
    prob = xp.nan_to_num(prob)
    prob *= (xx >= 0) * (xx <= scale)
    return prob


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
    if xp.any(xp.asarray(low) < 0):
        raise ValueError(f"Parameter low must be greater or equal zero, low={low}.")
    if alpha == -1:
        norm = 1 / xp.log(high / low)
    else:
        norm = (1 + alpha) / (high ** (1 + alpha) - low ** (1 + alpha))
    prob = xp.power(xx, alpha)
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


def truncnorm(xx, mu, sigma, high, low):
    r"""
    Truncated normal probability

    .. math::
        p(x) =
        \sqrt{\frac{2}{\pi\sigma^2}}
        \left[\text{erf}\left(\frac{x_\max - \mu}{\sqrt{2}}\right) + \text{erf}\left(\frac{\mu - x_\min}{\sqrt{2}}\right)\right]^{-1}
        \exp\left(-\frac{(\mu - x)^2}{2 \sigma^2}\right)

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
    if sigma <= 0:
        raise ValueError(f"Sigma must be greater than 0, sigma={sigma}")
    norm = 2**0.5 / xp.pi**0.5 / sigma
    norm /= erf((high - mu) / 2**0.5 / sigma) + erf((mu - low) / 2**0.5 / sigma)
    prob = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma**2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


def unnormalized_2d_gaussian(xx, yy, mu_x, mu_y, sigma_x, sigma_y, covariance):
    determinant = sigma_x**2 * sigma_y**2 * (1 - covariance)
    residual_x = (mu_x - xx) * sigma_x
    residual_y = (mu_y - yy) * sigma_y
    prob = xp.exp(
        -(residual_x**2 + residual_y**2 - 2 * residual_x * residual_y * covariance)
        / 2
        / determinant
    )
    return prob


def get_version_information():
    version_file = os.path.join(os.path.dirname(__file__), ".version")
    try:
        with open(version_file, "r") as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")
