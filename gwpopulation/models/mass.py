from ..cupy_utils import trapz, xp
from ..utils import powerlaw, truncnorm


def power_law_primary_mass_ratio(dataset, alpha, beta, mmin, mmax):
    """
    Power law model for two-dimensional mass distribution, modelling primary
    mass and conditional mass ratio distribution.

    p(m1, q) = p(m1) * p(q | m1)

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha: float
        Negative power law exponent for more massive black hole.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum black hole mass.
    beta: float
        Power law exponent of the mass ratio distribution.
    """
    return two_component_primary_mass_ratio(
        dataset, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax, lam=0, mpp=35,
        sigpp=1)


def _primary_secondary_general(dataset, p_m1, p_m2):
    return p_m1 * p_m2 * (dataset['mass_1'] >= dataset['mass_2']) * 2


def power_law_primary_secondary_independent(dataset, alpha, beta, mmin, mmax):
    """
    Power law model for two-dimensional mass distribution, modelling the
    primary and secondary masses as following independent distributions.

    p(m1, m2) = p(m1) * p(m2) : m1 >= m2

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_2'.
    alpha: float
        Negative power law exponent for more massive black hole.
    beta: float
        Negative power law exponent of the secondary mass distribution.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum black hole mass.
    """
    p_m1 = powerlaw(dataset['mass_1'], -alpha, mmax, mmin)
    p_m2 = powerlaw(dataset['mass_2'], -beta, mmax, mmin)
    prob = _primary_secondary_general(dataset, p_m1, p_m2)
    return prob


def power_law_primary_secondary_identical(dataset, alpha, mmin, mmax):
    """
    Power law model for two-dimensional mass distribution, modelling the
    primary and secondary masses as following independent distributions.

    p(m1, m2) = p(m1) * p(m2) : m1 >= m2

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_2'.
    alpha: float
        Negative power law exponent for more massive black hole.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum black hole mass.
    """
    return power_law_primary_secondary_independent(
        dataset=dataset, alpha=alpha, beta=alpha, mmin=mmin, mmax=mmax)


def two_component_single(mass, alpha, mmin, mmax, lam, mpp, sigpp):
    """
    Power law model for one-dimensional mass distribution.

    Parameters
    ----------
    mass: array-like
        Array of mass values.
    alpha: float
        Negative power law exponent for the black hole distribution.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum black hole mass.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation fo the Gaussian component.
    """
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=100, low=mmin)
    prob = (1 - lam) * p_pow + lam * p_norm
    return prob


def two_component_primary_mass_ratio(
        dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp):
    """
    Power law model for two-dimensional mass distribution, modelling primary
    mass and conditional mass ratio distribution.

    p(m1, q) = p(m1) * p(q | m1)

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha: float
        Negative power law exponent for more massive black hole.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum black hole mass.
    beta: float
        Power law exponent of the mass ratio distribution.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation fo the Gaussian component.
    """
    params = dict(mmin=mmin, mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp)
    p_m1 = two_component_single(dataset['mass_1'], alpha=alpha, **params)
    p_q = powerlaw(dataset['mass_ratio'], beta, 1, mmin / dataset['mass_1'])
    prob = p_m1 * p_q
    return prob


def two_component_primary_secondary_independent(
        dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp):
    """
    Power law model for two-dimensional mass distribution, modelling the
    primary and secondary masses as following independent distributions.

    p(m1, m2) = p(m1) * p(m2) : m1 >= m2

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_2'.
    alpha: float
        Negative power law exponent for more massive black hole.
    beta: float
        Negative power law exponent of the secondary mass distribution.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum black hole mass.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation fo the Gaussian component.
    """
    params = dict(mmin=mmin, mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp)
    p_m1 = two_component_single(dataset['mass_1'], alpha=alpha, **params)
    p_m2 = two_component_single(dataset['mass_2'], alpha=beta, **params)

    prob = _primary_secondary_general(dataset, p_m1, p_m2)
    return prob


def two_component_primary_secondary_identical(
        dataset, alpha, mmin, mmax, lam, mpp, sigpp):
    """
    Power law model for two-dimensional mass distribution, modelling the
    primary and secondary masses as following independent distributions.

    p(m1, m2) = p(m1) * p(m2) : m1 >= m2

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_2'.
    alpha: float
        Negative power law exponent for more massive black hole.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum black hole mass.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation fo the Gaussian component.
    """
    return two_component_primary_secondary_independent(
        dataset=dataset, alpha=alpha, beta=alpha, mmin=mmin, mmax=mmax,
        lam=lam, mpp=mpp, sigpp=sigpp)


class SmoothedMassDistribution(object):

    def __init__(self):
        self.m1s = xp.linspace(3, 100, 1000)
        self.qs = xp.linspace(0.001, 1, 500)
        self.dm = self.m1s[1] - self.m1s[0]
        self.dq = self.qs[1] - self.qs[0]
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)

    def __call__(
            self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m):
        """
        Powerlaw + peak model for two-dimensional mass distribution with low
        mass smoothing.

        https://arxiv.org/abs/1801.02699 Eq. (11) (T&T18)

        Parameters
        ----------
        dataset: dict
            Dictionary of numpy arrays for 'mass_1' and 'mass_2', also
            'arg_m1s'.
        alpha: float
            Powerlaw exponent for more massive black hole.
        beta: float
            Power law exponent of the mass ratio distribution.
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
        delta_m: float
            Rise length of the low end of the mass distribution.

        Notes
        -----
        The interpolation of the p(q) normalisation has a fill value of
        the normalisation factor for m_1 = 100.
        """
        p_m1 = self.p_m1(dataset, alpha=alpha, mmin=mmin, mmax=mmax, lam=lam,
                         mpp=mpp, sigpp=sigpp, delta_m=delta_m)
        p_q = self.p_q(dataset, beta=beta, mmin=mmin, delta_m=delta_m)
        prob = p_m1 * p_q
        return prob

    def p_m1(self, dataset, alpha, mmin, mmax, lam, mpp, sigpp, delta_m):
        p_m = two_component_single(dataset['mass_1'], alpha=alpha, mmin=mmin,
                                   mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp)
        p_m *= self.smoothing(
            dataset['mass_1'], mmin=mmin, mmax=100, delta_m=delta_m)
        norm = self.norm_p_m1(alpha=alpha, mmin=mmin, mmax=mmax, lam=lam,
                              mpp=mpp, sigpp=sigpp, delta_m=delta_m)
        return p_m / norm

    def p_q(self, dataset, beta, mmin, delta_m):
        p_q = powerlaw(dataset['mass_ratio'], beta, 1,
                       mmin / dataset['mass_1'])
        p_q *= self.smoothing(
            dataset['mass_1'] * dataset['mass_ratio'], mmin=mmin,
            mmax=dataset['mass_1'], delta_m=delta_m)
        try:
            p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)
        except (AttributeError, TypeError, ValueError):
            self._cache_q_norms(dataset['mass_1'])
            p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)

        return xp.nan_to_num(p_q)

    def norm_p_m1(self, alpha, mmin, mmax, lam, mpp, sigpp, delta_m):
        """Calculate the normalisation factor for the primary mass"""
        if delta_m == 0.0:
            return 1
        p_m = two_component_single(self.m1s, alpha=alpha, mmin=mmin,
                                   mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp)
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=100, delta_m=delta_m)

        norm = trapz(p_m, self.m1s)
        return norm

    def norm_p_q(self, beta, mmin, delta_m):
        """Calculate the mass ratio normalisation by linear interpolation"""
        if delta_m == 0.0:
            return 1
        p_q = powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid)
        p_q *= self.smoothing(self.m1s_grid * self.qs_grid, mmin=mmin,
                              mmax=self.m1s_grid, delta_m=delta_m)
        norms = trapz(p_q, self.qs, axis=0)

        all_norms = (norms[self.n_below] * (1 - self.step) +
                     norms[self.n_above] * self.step)

        return all_norms

    def _cache_q_norms(self, masses):
        """
        Cache the information necessary for linear interpolation of the mass
        ratio normalisation
        """
        self.n_below = xp.zeros_like(masses, dtype=xp.int) - 1
        m_below = xp.zeros_like(masses)
        for mm in self.m1s:
            self.n_below += masses > mm
            m_below[masses > mm] = mm
        self.n_above = self.n_below + 1
        max_idx = len(self.m1s)
        self.n_below[self.n_below < 0] = 0
        self.n_above[self.n_above == max_idx] = max_idx - 1
        self.step = xp.minimum((masses - m_below) / self.dm, 1)

    @staticmethod
    def smoothing(masses, mmin, mmax, delta_m):
        """
        Apply a one sided window between mmin and mmin+dm to the mass pdf.

        The upper cut off is a step function,
        the lower cutoff is a logistic rise over delta_m solar masses.

        See T&T18 Eq
        """
        window = xp.ones_like(masses)
        if delta_m > 0.0:
            mass_range = mmax - mmin
            delta_m /= mass_range
            sel_p = ((masses >= mmin) &
                     (masses <= (mmin + delta_m * mass_range)))
            ms_p = masses - mmin
            z_p = xp.nan_to_num(2 * delta_m * (1 / (2 * ms_p / mass_range) +
                                1 / (2 * ms_p / mass_range - 2 * delta_m)))
            window[sel_p] = 1 / (xp.exp(z_p[sel_p]) + 1)
        window[(masses < mmin) | (masses > mmax)] = 0
        return window


smoothed_two_component_primary_mass_ratio = SmoothedMassDistribution()
