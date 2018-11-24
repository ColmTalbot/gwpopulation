try:
    import cupy as xp
    from cupyx.scipy.special import erf, gammaln
except ImportError:
    import numpy as xp
    from scipy.special import erf, gammaln


def beta_dist(xx, alpha, beta, scale=1):
    ln_beta = (alpha - 1) * xx + (beta - 1) * (scale - xx)
    ln_beta -= betaln(alpha, beta)
    ln_beta -= (alpha + beta - 1) * xp.log(scale)
    prob = xp.exp(ln_beta)
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
