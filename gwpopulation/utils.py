import os

from .cupy_utils import erf, gammaln, xp


def beta_dist(xx, alpha, beta, scale=1):
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


def betaln(alpha, beta):
    ln_beta = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    return ln_beta


def powerlaw(xx, alpha, high, low):
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
    if sigma <= 0:
        raise ValueError(f"Sigma must be greater than 0, sigma={sigma}")
    norm = 2 ** 0.5 / xp.pi ** 0.5 / sigma
    norm /= erf((high - mu) / 2 ** 0.5 / sigma) + erf((mu - low) / 2 ** 0.5 / sigma)
    prob = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma ** 2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


def get_version_information():
    version_file = os.path.join(os.path.dirname(__file__), ".version")
    try:
        with open(version_file, "r") as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")
