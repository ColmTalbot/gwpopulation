import os

from .cupy_utils import erf, gammaln, xp


def beta_dist(xx, alpha, beta, scale=1):
    ln_beta = (alpha - 1) * xp.log(xx) + (beta - 1) * xp.log(scale - xx)
    ln_beta -= betaln(alpha, beta)
    ln_beta -= (alpha + beta - 1) * xp.log(scale)
    prob = xp.exp(ln_beta)
    prob = xp.nan_to_num(prob)
    prob *= (xx <= scale)
    return prob


def betaln(alpha, beta):
    ln_beta = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    return ln_beta


def powerlaw(xx, alpha, high, low):
    norm = (1 + alpha) / (high**(1 + alpha) - low**(1 + alpha))
    prob = xp.power(xx, alpha)
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


def truncnorm(xx, mu, sigma, high, low):
    norm = 2**0.5 / xp.pi**0.5 / sigma
    norm /= erf((high - mu) / 2**0.5 / sigma) + erf((mu - low) / 2**0.5 / sigma)
    prob = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma**2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


def get_version_information():
    version_file = os.path.join(os.path.dirname(__file__), '.version')
    try:
        with open(version_file, 'r') as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")
