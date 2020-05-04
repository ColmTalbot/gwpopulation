from warnings import warn

from ..cupy_utils import trapz, xp
from ..utils import powerlaw, truncnorm


def double_power_law_primary_mass(mass, alpha_1, alpha_2, mmin, mmax, break_fraction):
    """
    Parameters
    ----------
    alpha_1: float
        Powerlaw exponent for more massive black hole below break.
    alpha_2: float
        Powerlaw exponent for more massive black hole above break.
    break_fraction:float
        The fraction between mmin and mmax primary mass distribution breaks.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum mass in the powerlaw distributed component.
    """

    prob = xp.zeros_like(mass)
    m_break = mmin + break_fraction * (mmax - mmin)
    correction = powerlaw(m_break, alpha=-alpha_2, low=m_break, high=mmax) / powerlaw(
        m_break, alpha=-alpha_1, low=mmin, high=m_break
    )
    low_part = powerlaw(mass[mass < m_break], alpha=-alpha_1, low=mmin, high=m_break)
    prob[mass < m_break] = low_part * correction
    high_part = powerlaw(mass[mass >= m_break], alpha=-alpha_2, low=m_break, high=mmax)
    prob[mass >= m_break] = high_part
    return prob / (1 + correction)


def double_power_law_peak_primary_mass(
        mass, alpha_1, alpha_2, mmin, mmax, break_fraction, lam, mpp, sigpp
):
    """
    Parameters
    ----------
    alpha_1: float
        Powerlaw exponent for more massive black hole below break.
    alpha_2: float
        Powerlaw exponent for more massive black hole above break.
    break_fraction:float
        The fraction between mmin and mmax primary mass distribution breaks.
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
    """

    p_pow = double_power_law_primary_mass(
        mass=mass,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        mmin=mmin,
        mmax=mmax,
        break_fraction=break_fraction,
    )
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=100, low=mmin)
    prob = (1 - lam) * p_pow + lam * p_norm
    return prob


def double_power_law_primary_power_law_mass_ratio(
    dataset, alpha_1, alpha_2, beta, mmin, mmax, break_fraction
):
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
    break_fraction: float
        Break point of the primary mass distribution.
        This is specified as a fraction of the way between mmin and mmax.
        E.g., mmin=5, mmax=45, break_fraction=0.5 would have a break at 25
    beta: float
        Power law exponent of the mass ratio distribution.
    """
    params = dict(mmin=mmin, mmax=mmax, break_fraction=break_fraction)
    p_m1 = double_power_law_primary_mass(
        dataset["mass_1"], alpha_1=alpha_1, alpha_2=alpha_2, **params
    )
    p_q = powerlaw(dataset["mass_ratio"], beta, 1, mmin / dataset["mass_1"])
    prob = p_m1 * p_q
    return prob


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
        dataset, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax, lam=0, mpp=35, sigpp=1
    )


def _primary_secondary_general(dataset, p_m1, p_m2):
    return p_m1 * p_m2 * (dataset["mass_1"] >= dataset["mass_2"]) * 2


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
    p_m1 = powerlaw(dataset["mass_1"], -alpha, mmax, mmin)
    p_m2 = powerlaw(dataset["mass_2"], -beta, mmax, mmin)
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
        dataset=dataset, alpha=alpha, beta=alpha, mmin=mmin, mmax=mmax
    )


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
        Standard deviation of the Gaussian component.
    """
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=100, low=mmin)
    prob = (1 - lam) * p_pow + lam * p_norm
    return prob


def three_component_single(
    mass, alpha, mmin, mmax, lam, lam_1, mpp_1, sigpp_1, mpp_2, sigpp_2
):
    """
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
        Fraction of black holes in the Gaussian components.
    lam_1: float
        Fraction of black holes in the lower mass Gaussian component.
    mpp_1: float
        Mean of the lower mass Gaussian component.
    mpp_2: float
        Mean of the upper mass Gaussian component.
    sigpp_1: float
        Standard deviation of the lower mass Gaussian component.
    sigpp_2: float
        Standard deviation of the upper mass Gaussian component.
    """
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm1 = truncnorm(mass, mu=mpp_1, sigma=sigpp_1, high=100, low=mmin)
    p_norm2 = truncnorm(mass, mu=mpp_2, sigma=sigpp_2, high=100, low=mmin)
    prob = (1 - lam) * p_pow + lam * lam_1 * p_norm1 + lam * (1 - lam_1) * p_norm2
    return prob


def two_component_primary_mass_ratio(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp):
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
    p_m1 = two_component_single(dataset["mass_1"], alpha=alpha, **params)
    p_q = powerlaw(dataset["mass_ratio"], beta, 1, mmin / dataset["mass_1"])
    prob = p_m1 * p_q
    return prob


def two_component_primary_secondary_independent(
    dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp
):
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
    p_m1 = two_component_single(dataset["mass_1"], alpha=alpha, **params)
    p_m2 = two_component_single(dataset["mass_2"], alpha=beta, **params)

    prob = _primary_secondary_general(dataset, p_m1, p_m2)
    return prob


def two_component_primary_secondary_identical(
    dataset, alpha, mmin, mmax, lam, mpp, sigpp
):
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
        dataset=dataset,
        alpha=alpha,
        beta=alpha,
        mmin=mmin,
        mmax=mmax,
        lam=lam,
        mpp=mpp,
        sigpp=sigpp,
    )


class _SmoothedMassDistribution(object):
    def __init__(self):
        self.m1s = xp.linspace(2, 100, 1000)
        self.qs = xp.linspace(0.001, 1, 500)
        self.dm = self.m1s[1] - self.m1s[0]
        self.dq = self.qs[1] - self.qs[0]
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)

    def p_q(self, dataset, beta, mmin, delta_m):
        p_q = powerlaw(dataset["mass_ratio"], beta, 1, mmin / dataset["mass_1"])
        p_q *= self.smoothing(
            dataset["mass_1"] * dataset["mass_ratio"],
            mmin=mmin,
            mmax=dataset["mass_1"],
            delta_m=delta_m,
        )
        try:
            p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)
        except (AttributeError, TypeError, ValueError):
            self._cache_q_norms(dataset["mass_1"])
            p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)

        return xp.nan_to_num(p_q)

    def norm_p_q(self, beta, mmin, delta_m):
        """Calculate the mass ratio normalisation by linear interpolation"""
        if delta_m == 0.0:
            return 1
        p_q = powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid)
        p_q *= self.smoothing(
            self.m1s_grid * self.qs_grid, mmin=mmin, mmax=self.m1s_grid, delta_m=delta_m
        )
        norms = trapz(p_q, self.qs, axis=0)

        all_norms = (
            norms[self.n_below] * (1 - self.step) + norms[self.n_above] * self.step
        )

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
        Apply a one sided window between mmin and mmin + delta_m to the
        mass pdf.

        The upper cut off is a step function,
        the lower cutoff is a logistic rise over delta_m solar masses.

        See T&T18 Eqs 7-8
        Note that there is a sign error in that paper.

        S = (f(m - mmin, delta_m) + 1)^{-1}
        f(m') = delta_m / m' + delta_m / (m' - delta_m)

        See also, https://en.wikipedia.org/wiki/Window_function#Planck-taper_window
        """
        window = xp.ones_like(masses)
        if delta_m > 0.0:
            smoothing_region = (masses > mmin) & (masses < (mmin + delta_m))
            shifted_mass = masses[smoothing_region] - mmin
            if shifted_mass.size:
                exponent = xp.nan_to_num(
                    delta_m / shifted_mass + delta_m / (shifted_mass - delta_m)
                )
                window[smoothing_region] = 1 / (xp.exp(exponent) + 1)
        window[(masses < mmin) | (masses > mmax)] = 0
        return window


class SinglePeakSmoothedMassDistribution(_SmoothedMassDistribution):
    def __call__(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m):
        """
        Powerlaw + peak model for two-dimensional mass distribution with low
        mass smoothing.

        https://arxiv.org/abs/1801.02699 Eq. (11) (T&T18)

        Parameters
        ----------
        dataset: dict
            Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
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
        p_m1 = self.p_m1(
            dataset,
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
            delta_m=delta_m,
        )
        p_q = self.p_q(dataset, beta=beta, mmin=mmin, delta_m=delta_m)
        prob = p_m1 * p_q
        return prob

    def p_m1(self, dataset, alpha, mmin, mmax, lam, mpp, sigpp, delta_m):
        p_m = two_component_single(
            dataset["mass_1"],
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        p_m *= self.smoothing(dataset["mass_1"], mmin=mmin, mmax=100, delta_m=delta_m)
        norm = self.norm_p_m1(
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
            delta_m=delta_m,
        )
        return p_m / norm

    def norm_p_m1(self, alpha, mmin, mmax, lam, mpp, sigpp, delta_m):
        """Calculate the normalisation factor for the primary mass"""
        if delta_m == 0.0:
            return 1
        p_m = two_component_single(
            self.m1s, alpha=alpha, mmin=mmin, mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp
        )
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=100, delta_m=delta_m)

        norm = trapz(p_m, self.m1s)
        return norm


class SmoothedMassDistribution(SinglePeakSmoothedMassDistribution):
    def __init__(self):
        warn(
            "SmoothedMassDistribution has been deprecated and will be removed in a "
            "future release, use SinglePeakSmoothedMassDistribution instead.",
            DeprecationWarning,
        )
        super(SmoothedMassDistribution, self).__init__()


class MultiPeakSmoothedMassDistribution(_SmoothedMassDistribution):
    def __call__(
        self,
        dataset,
        alpha,
        beta,
        mmin,
        mmax,
        lam,
        lam_1,
        mpp_1,
        sigpp_1,
        mpp_2,
        sigpp_2,
        delta_m,
    ):
        """
        Powerlaw + two peak model for two-dimensional mass distribution with
        low mass smoothing.

        Parameters
        ----------
        dataset: dict
            Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
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
        lam_1: float
            Fraction of black holes in the lower mass Gaussian component.
        mpp_1: float
            Mean of the lower mass Gaussian component.
        mpp_2: float
            Mean of the upper mass Gaussian component.
        sigpp_1: float
            Standard deviation of the lower mass Gaussian component.
        sigpp_2: float
            Standard deviation of the upper mass Gaussian component.
        delta_m: float
            Rise length of the low end of the mass distribution.
        """
        p_m1 = self.p_m1(
            dataset,
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            lam_1=lam_1,
            mpp_1=mpp_1,
            mpp_2=mpp_2,
            sigpp_1=sigpp_1,
            sigpp_2=sigpp_2,
            delta_m=delta_m,
        )
        p_q = self.p_q(dataset, beta=beta, mmin=mmin, delta_m=delta_m)
        prob = p_m1 * p_q
        return prob

    def p_m1(
        self,
        dataset,
        alpha,
        mmin,
        mmax,
        lam,
        lam_1,
        mpp_1,
        sigpp_1,
        mpp_2,
        sigpp_2,
        delta_m,
    ):
        p_m = three_component_single(
            dataset["mass_1"],
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            lam_1=lam_1,
            mpp_1=mpp_1,
            mpp_2=mpp_2,
            sigpp_1=sigpp_1,
            sigpp_2=sigpp_2,
        )
        p_m *= self.smoothing(dataset["mass_1"], mmin=mmin, mmax=100, delta_m=delta_m)
        norm = self.norm_p_m1(
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            lam_1=lam_1,
            mpp_1=mpp_1,
            mpp_2=mpp_2,
            sigpp_1=sigpp_1,
            sigpp_2=sigpp_2,
            delta_m=delta_m,
        )
        return p_m / norm

    def norm_p_m1(
        self, alpha, mmin, mmax, lam, lam_1, mpp_1, sigpp_1, mpp_2, sigpp_2, delta_m
    ):
        if delta_m == 0.0:
            return 1
        p_m = three_component_single(
            self.m1s,
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            lam_1=lam_1,
            mpp_1=mpp_1,
            mpp_2=mpp_2,
            sigpp_1=sigpp_1,
            sigpp_2=sigpp_2,
        )
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=100, delta_m=delta_m)
        norm = trapz(p_m, self.m1s)
        return norm


class BrokenPowerLawSmoothedMassDistribution(_SmoothedMassDistribution):
    def __call__(
        self, dataset, alpha_1, alpha_2, beta, mmin, mmax, delta_m, break_fraction,
    ):
        """
        Broken power law for two-dimensional mass distribution with low
        mass smoothing.

        Parameters
        ----------
        dataset: dict
            Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
        alpha_1: float
            Powerlaw exponent for more massive black hole below break.
        alpha_2: float
            Powerlaw exponent for more massive black hole above break.
        beta: float
            Power law exponent of the mass ratio distribution.
        break_fraction: float
            Fraction between mmin and mmax primary mass distribution breaks at.
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
        """

        p_m1 = self.p_m1(
            dataset,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            delta_m=delta_m,
            break_fraction=break_fraction,
        )
        p_q = self.p_q(dataset, beta=beta, mmin=mmin, delta_m=delta_m)
        prob = p_m1 * p_q
        return prob

    def p_m1(self, dataset, alpha_1, alpha_2, mmin, mmax, delta_m, break_fraction):
        p_m = double_power_law_primary_mass(
            dataset["mass_1"],
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            break_fraction=break_fraction,
        )
        p_m *= self.smoothing(dataset["mass_1"], mmin=mmin, mmax=100, delta_m=delta_m)
        norm = self.norm_p_m1(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            delta_m=delta_m,
            break_fraction=break_fraction,
        )
        return p_m / norm

    def norm_p_m1(self, alpha_1, alpha_2, mmin, mmax, delta_m, break_fraction):
        if delta_m == 0.0:
            return 1
        p_m = double_power_law_primary_mass(
            self.m1s,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            break_fraction=break_fraction,
        )
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=100, delta_m=delta_m)
        norm = trapz(p_m, self.m1s)
        return norm


class BrokenPowerLawPeakSmoothedMassDistribution(_SmoothedMassDistribution):
    def __call__(
        self,
        dataset,
        alpha_1,
        alpha_2,
        beta,
        mmin,
        mmax,
        delta_m,
        break_fraction,
        lam,
        mpp,
        sigpp,
    ):
        """
        Broken power law for two-dimensional mass distribution with low
        mass smoothing.

        Parameters
        ----------
        dataset: dict
            Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
        alpha_1: float
            Powerlaw exponent for more massive black hole below break.
        alpha_2: float
            Powerlaw exponent for more massive black hole above break.
        beta: float
            Power law exponent of the mass ratio distribution.
        break_fraction: float
            Fraction between mmin and mmax primary mass distribution breaks at.
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
        """

        p_m1 = self.p_m1(
            dataset,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            delta_m=delta_m,
            break_fraction=break_fraction,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        p_q = self.p_q(dataset, beta=beta, mmin=mmin, delta_m=delta_m)
        prob = p_m1 * p_q
        return prob

    def p_m1(self, dataset, alpha_1, alpha_2, mmin, mmax, delta_m, break_fraction, lam, mpp, sigpp):
        p_m = double_power_law_peak_primary_mass(
            dataset["mass_1"],
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            break_fraction=break_fraction,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        p_m *= self.smoothing(dataset["mass_1"], mmin=mmin, mmax=100, delta_m=delta_m)
        norm = self.norm_p_m1(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            delta_m=delta_m,
            break_fraction=break_fraction,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        return p_m / norm

    def norm_p_m1(
        self, alpha_1, alpha_2, mmin, mmax, delta_m, break_fraction, lam, mpp, sigpp
    ):
        if delta_m == 0.0:
            return 1
        p_m = double_power_law_peak_primary_mass(
            self.m1s,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            break_fraction=break_fraction,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=100, delta_m=delta_m)
        norm = trapz(p_m, self.m1s)
        return norm
