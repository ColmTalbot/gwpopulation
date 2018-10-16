from __future__ import division

import numpy as np
from scipy.special import erf
from scipy.stats import beta as beta_dist
from scipy.stats import truncnorm
from scipy.interpolate import interp1d


def iid_spin(dataset, xi, sigma_spin, amax, alpha_chi, beta_chi):
    """
    Independently and identically distributed spins.
    """
    prior = iid_spin_orientation(dataset, xi, sigma_spin) *\
        iid_spin_magnitude(dataset, amax, alpha_chi, beta_chi)
    return prior


def iid_spin_orientation(dataset, xi, sigma_spin):
    """
    Independently and identically distributed spin orientations.
    """
    return spin_orientation_likelihood(dataset, xi, sigma_spin, sigma_spin)


def iid_spin_magnitude(dataset, amax=1, alpha_chi=1, beta_chi=1):
    """
    Independently and identically distributed spin magnitudes.
    """
    return spin_magnitude_beta_likelihood(
        dataset, alpha_chi, alpha_chi, beta_chi, beta_chi, amax, amax)


def spin_orientation_likelihood(dataset, xi, sigma_1, sigma_2):
    """A mixture model of spin orientations with isotropic and normally
    distributed components.

    https://arxiv.org/abs/1704.08370 Eq. (4)

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'costilt1' and 'costilt2'.
    xi: float
        Fraction of black holes in preferentially aligned component.
    sigma_1: float
        Width of preferentially aligned component for the more
        massive black hole.
    sigma_2: float
        Width of preferentially aligned component for the less
        massive black hole.
    """
    # prior = (1 - xi) / 4\
    #     + xi * 2 / np.pi / sigma_1 / sigma_2\
    #     * truncnorm_wrapper(dataset['costilt1'], 1, sigma_1, -1, 1)\
    #     * truncnorm_wrapper(dataset['costilt2'], 1, sigma_2, -1, 1)
    prior = (1 - xi) / 4\
        + xi * 2 / np.pi / sigma_1 / sigma_2\
        * np.exp(-(dataset['costilt1'] - 1)**2 / (2 * sigma_1**2))\
        / erf(2**0.5 / sigma_1)\
        * np.exp(-(dataset['costilt2'] - 1)**2 / (2 * sigma_2**2))\
        / erf(2**0.5 / sigma_2)
    prior *= (abs(dataset['costilt1']) <= 1) & (abs(dataset['costilt2']) <= 1)
    return prior


def truncnorm_wrapper(xx, mean, sigma, min_, max_):
    """Wrapper to scipy truncnorm"""
    aa = (min_ - mean) / sigma
    bb = (max_ - mean) / sigma
    return truncnorm.pdf(xx, aa, bb, loc=mean, scale=sigma)


def spin_magnitude_beta_likelihood(dataset, alpha_1, alpha_2, beta_1, beta_2,
                                   amax_1, amax_2):
    """ Independent beta distributions for both spin magnitudes.

    https://arxiv.org/abs/1805.06442 Eq. (10)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays containing 'a1' and 'a2'.
    alpha_1, beta_1: float
        Parameters of Beta distribution for more massive black hole.
    alpha_2, beta_2: float
        Parameters of Beta distribution for less massive black hole.
    amax_1, amax_2: float
        Maximum spin of the more/less massive black hole.
    """
    if alpha_1 < 0 or beta_1 < 0 or alpha_2 < 0 or beta_2 < 0:
        return 0
    prior = beta_dist.pdf(dataset['a1'], alpha_1, beta_1, scale=amax_1) *\
        beta_dist.pdf(dataset['a2'], alpha_2, beta_2, scale=amax_2)
    return prior


def mass_distribution(dataset, alpha, mmin, mmax, lam, mpp, sigpp, beta,
                      delta_m):
    """ Powerlaw + peak model for two-dimensional mass distribution adjusted
    for sensitive volume.

    https://arxiv.org/abs/1801.02699 Eq. (11) (T&T18)

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'm1_source' and 'm2_source', also
        'arg_m1s'.
    alpha: float
        Powerlaw exponent for more massive black hole.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum mass in the powerlaw distributed component.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation fo the Gaussian component.
    beta: float
        Power law exponent of the mass ratio distribution.
    delta_m: float
        Rise length of the low end of the mass distribution.
    """
    parameters = [alpha, mmin, mmax, lam, mpp, sigpp, beta, delta_m]
    probability = mass_distribution_no_vt(dataset, *parameters)
    vt_fac = norm_vt(parameters)
    probability /= vt_fac
    return probability


def mass_distribution_no_vt(dataset, alpha, mmin, mmax, lam, mpp, sigpp, beta,
                            delta_m):
    """ Powerlaw + peak model for two-dimensional mass distribution not adjusted
    for sensitive volume.

    https://arxiv.org/abs/1801.02699 Eq. (11) (T&T18)

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'm1_source' and 'm2_source', also
        'arg_m1s'.
    alpha: float
        Powerlaw exponent for more massive black hole.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum mass in the powerlaw distributed component.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation fo the Gaussian component.
    beta: float
        Power law exponent of the mass ratio distribution.
    delta_m: float
        Rise length of the low end of the mass distribution.

    Notes
    -----
    The interpolation of the p(q) normalisation has a fill value of
    the normalisation factor for m_1 = 100.
    """
    parameters = [alpha, mmin, mmax, lam, mpp, sigpp, beta, delta_m]
    pow_norm, pp_norm, qnorms_ = norms(parameters)
    qnorms = interp1d(m1s, qnorms_, bounds_error=False,
                      fill_value=qnorms_[-1])(dataset['m1_source'])
    probability = pmodel2d(dataset['m1_source'], dataset['q'], parameters,
                           pow_norm, pp_norm, qnorms)
    return probability


def iid_mass(dataset, alpha, mmin, mmax, lam, mpp, sigpp, delta_m):
    """
    Identically and independently masses following p(m1) in T&T 2018

    Parameters
    ----------
    dataset: dict
        Dictionary containing NxM arrays, m1_source and m2_source
    alpha, mmin, mmax, lam, mpp, sigpp, delta_m: see mass_distribution

    Returns
    -------
    probability: array-like
        Probability of m1, m2 in dataset, shape=(NxM)

    Notes
    -----
    The factor of 2 comes from requiring m1>m2
    """
    parameters = dict(
        alpha=alpha, mmin=mmin, mmax=mmax, lam=lam, mpp=mpp,
        sigpp=sigpp, delta_m=delta_m, beta=0)
    pow_norm = norm_ppow(parameters)
    pp_norm = norm_pnorm(parameters)
    probability = pmodel1d(dataset['m1_source'], parameters, pow_norm, pp_norm)
    probability *= pmodel1d(dataset['m2_source'], parameters, pow_norm, pp_norm)
    probability[dataset['m1_source'] < dataset['m2_source']] = 0
    probability *= 2
    return probability


def norms(parameters):
    """
    Calculate normalisation factors for the model in T&T 2018.

    Since our model doesn't have an anlaytic integral we must normalise
    numerically. Every value of m_1 has a unique normalisation for q.

    Parameters
    ----------
    parameters: array-like
        Rescaled sample from the prior distribution.

    Return
    ------
    pow_norm: float
        Normalisation factor for the smoothed power law distribution.
    pp_norm: float
        Normalisation factor for the smoothed Gaussian distribution.
    qnorms: array-like
        Normalisation factor for each value of m1 in norm_array
    """
    pow_norm = norm_ppow(parameters)
    pp_norm = norm_pnorm(parameters)
    qnorms = norm_pq(parameters)
    return [pow_norm, pp_norm, qnorms]


def pmodel2d(ms, qs, parameters, pow_norm, pp_norm, qnorms, vt_fac=1.):
    """
    2d mass model from T&T 2018

    Notes
    -----
    nan_to_num captures case when qnorms=0.
    """
    p_norm_no_vt = pmodel1d(ms, parameters, pow_norm, pp_norm) *\
        pq(qs, ms, parameters) / qnorms
    if not vt_fac == 1:
        print('Providing vt_fac to pmodel2d is being deprecated.')
    p_norm = p_norm_no_vt / vt_fac
    return np.nan_to_num(p_norm)


def pmodel1d(ms, parameters, pow_norm, pp_norm):
    """normalised m1 pdf from T&T 2018"""
    al, mn, mx, lam, mp, sp, bt, delta_m = extract_mass_parameters(parameters)
    p_pow = ppow(ms, parameters) / pow_norm
    p_norm = pnorm(ms, parameters) / pp_norm
    return (1 - lam) * p_pow + lam * p_norm


def ppow(ms, parameters):
    """1d unnormalised powerlaw mass probability with smoothed low-mass end"""
    al, mn, mx, lam, mp, sp, bt, delta_m = extract_mass_parameters(parameters)
    return ms**(-al) * window(ms, mn, mx, delta_m)


def norm_ppow(parameters):
    """normalise ppow, requires m1s, an array of m values, and dm, the spacing of
    that array"""
    return np.trapz(ppow(m1s, parameters), m1s)


def pnorm(ms, parameters):
    """1d unnormalised normal distribution with low-mass smoothing"""
    al, mn, mx, lam, mp, sp, bt, delta_m = extract_mass_parameters(parameters)
    return np.exp(-(ms - mp)**2 / (2 * sp**2)) * window(ms, mn, 100., delta_m)


def norm_pnorm(parameters):
    """normalise pnorm, requires m1s, an array of m values, and dm, the spacing of
    that array"""
    return np.trapz(pnorm(m1s, parameters), m1s)


def pq(qs, ms, parameters):
    """unnormalised pdf for q, powerlaw + smoothing"""
    al, mn, mx, lam, mp, sp, bt, delta_m = extract_mass_parameters(parameters)
    return qs**(bt) * window(qs * ms, mn, 100., delta_m)


def norm_vt(parameters):
    """Calculate the total observed volume for a given set of parameters.

    This is equivalent to Eq. 6 of https://arxiv.org/abs/1805.06442

    """
    pow_norm = norm_ppow(parameters)
    pp_norm = norm_pnorm(parameters)
    qnorms = np.einsum('i,j->ji', norm_pq(parameters), np.ones_like(qs))
    qnorms[qnorms == 0] = 1.
    p_norm_no_vt = pmodel1d(norm_array['m1'], parameters, pow_norm, pp_norm)\
        * pq(norm_array['q'], norm_array['m1'], parameters) / qnorms
    vt_fac = np.trapz(np.trapz(p_norm_no_vt * norm_array['vt'], m1s), qs)
    return vt_fac


def norm_pq(parameters):
    """normalise pq, requires m1s, an array of m values, and dm, the spacing of
    that array"""
    norm = np.trapz(pq(norm_array['q'], norm_array['m1'], parameters),
                    qs, axis=0)
    return norm


def window(ms, mn, mx, delta_m=0.):
    """Apply a one sided window between mmin and mmin+dm to the mass pdf.

    The upper cut off is a step function,
    the lower cutoff is a logistic rise over delta_m solar masses.

    See T&T18 Eq

    """
    dM = mx - mn
    delta_m /= dM
    # some versions of numpy can't deal with pandas columns indexing an array
    ms_arr = np.array(ms)
    sel_p = (ms_arr >= mn) & (ms_arr <= (mn + delta_m * dM))
    ms_p = ms_arr[sel_p] - mn
    Zp = np.nan_to_num(2 * delta_m * (1 / (2 * ms_p / dM) +
                       1 / (2 * ms_p / dM - 2 * delta_m)))
    window = np.ones_like(ms)
    window[(ms_arr < mn) | (ms_arr > mx)] = 0
    window[sel_p] = 1 / (np.exp(Zp) + 1)
    return window


def extract_mass_parameters(parameters):
    """extract the parameters of the mass distribution hyperparameters used in
    T&T18 from either a list or dictionary."""
    if isinstance(parameters, list):
        return parameters
    elif isinstance(parameters, dict):
        keys = ['alpha', 'mmin', 'mmax', 'lam', 'mpp',
                'sigpp', 'beta', 'delta_m']
        return [parameters[key] for key in keys]


def set_vt(vt_array):
    """
    Set up normalisation arrays, including VT(m)

    Parameters
    ----------
    vt_array: dict
        Dictionary containing arrays in m1, q and VT to use for normalisation
    """
    global dm, dq, m1s, qs, norm_array
    norm_array = vt_array
    m1s = np.unique(norm_array['m1'])
    qs = np.unique(norm_array['q'])
    dm = m1s[1] - m1s[0]
    dq = qs[1] - qs[0]


# set up arrays for numerical normalisation
# this doesn't include VT(m)
m1s = np.linspace(3, 100, 1000)
qs = np.linspace(0.1, 1, 500)
dm = m1s[1] - m1s[0]
dq = qs[1] - qs[0]

norm_array = dict()
norm_array['m1'] = np.einsum('i,j->ji', m1s, np.ones_like(qs))
norm_array['q'] = np.einsum('i,j->ji', np.ones_like(m1s), qs)
