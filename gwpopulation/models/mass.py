"""
Implemented mass models
"""

import inspect

import numpy as np
import scipy.special as scs

from ..utils import powerlaw, truncnorm
from .interped import InterpolatedNoBaseModelIdentical

xp = np

__all__ = [
    "BaseSmoothedMassDistribution",
    "SinglePeakSmoothedMassDistribution",
    "MultiPeakSmoothedMassDistribution",
    "BrokenPowerLawSmoothedMassDistribution",
    "BrokenPowerLawPeakSmoothedMassDistribution",
    "InterpolatedPowerLaw",
    "double_power_law_primary_mass",
    "double_power_law_peak_primary_mass",
    "double_power_law_primary_power_law_mass_ratio",
    "power_law_primary_mass_ratio",
    "_primary_secondary_general",
    "power_law_primary_secondary_independent",
    "power_law_primary_secondary_identical",
    "power_law_mass",
    "two_component_single",
    "three_component_single",
    "two_component_primary_mass_ratio",
    "two_component_primary_secondary_independent",
    "two_component_primary_secondary_identical",
]


def double_power_law_primary_mass(mass, alpha_1, alpha_2, mmin, mmax, break_fraction):
    r"""
    Broken power-law mass distribution

    .. math::
        p(m | \alpha_1, m_\min, m_\max, \delta) &\propto \begin{cases}
            m^{-\alpha_1} : m_\min \leq m < m_\min + \delta (m_\max - m_\min)\\
            m^{-\alpha_2} : m_\min + \delta (m_\max - m_\min) \leq m < m_\max
        \end{cases}

    Parameters
    ----------
    mass: array-like
        Mass to evaluate probability at (:math:`m`).
    alpha_1: float
        Powerlaw exponent for more massive black hole below break (:math:`\alpha_1`).
    alpha_2: float
        Powerlaw exponent for more massive black hole above break (:math:`\alpha_2`).
    break_fraction: float
        The fraction between mmin and mmax primary mass distribution breaks (:math:`\delta`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum mass in the powerlaw distributed component (:math:`m_\max`).
    """

    m_break = mmin + break_fraction * (mmax - mmin)
    correction = powerlaw(m_break, alpha=-alpha_2, low=m_break, high=mmax) / powerlaw(
        m_break, alpha=-alpha_1, low=mmin, high=m_break
    )
    low_part = powerlaw(mass, alpha=-alpha_1, low=mmin, high=m_break)
    high_part = powerlaw(mass, alpha=-alpha_2, low=m_break, high=mmax)
    prob = low_part * (mass < m_break) * correction + high_part * (mass >= m_break)
    return prob / (1 + correction)


def double_power_law_peak_primary_mass(
    mass,
    alpha_1,
    alpha_2,
    mmin,
    mmax,
    break_fraction,
    lam,
    mpp,
    sigpp,
    gaussian_mass_maximum=100,
):
    r"""
    Broken power-law with a Gaussian component.

    .. math::
        p(m | \alpha_1, \alpha_2, m_\min, m_\max, \delta, \lambda_m, \mu_m, \sigma_m) =
        (1 - \lambda_m) p_{\text{bpl}}(m | \alpha_1, \alpha_2, m_\min, m_\max, \delta)
        + \lambda_m p_{\text{norm}}(m | \mu_m, \sigma_m)

    .. math::
        p_{\text{bpl}}(m | \alpha_1, m_\min, m_\max, \delta) &\propto \begin{cases}
            m^{-\alpha_1} : m_\min \leq m < m_\min + \delta (m_\max - m_\min)\\
            m^{-\alpha_2} : m_\min + \delta (m_\max - m_\min) \leq m < m_\max
        \end{cases}

    .. math::
        p_{\text{norm}}(m | \mu_m, \sigma_m) \propto \exp\left(-\frac{(m - \mu_{m})^2}{2\sigma^2_m}\right)

    Parameters
    ----------
    mass: array-like
        Mass to evaluate probability at (:math:`m`).
    alpha_1: float
        Powerlaw exponent for more massive black hole below break (:math:`\alpha_1`).
    alpha_2: float
        Powerlaw exponent for more massive black hole above break (:math:`\alpha_2`).
    break_fraction:float
        The fraction between mmin and mmax primary mass distribution breaks (:math:`\delta`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum mass in the powerlaw distributed component (:math:`m_\max`).
    lam: float
        Fraction of black holes in the Gaussian component (:math:`\lambda_m`).
    mpp: float
        Mean of the Gaussian component (:math:`\mu_m`).
    sigpp: float
        Standard deviation of the Gaussian component (:math:`\sigma_m`).
    gaussian_mass_maximum: float, optional
        Upper truncation limit of the Gaussian component. (default: 100)
    """

    p_pow = double_power_law_primary_mass(
        mass=mass,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        mmin=mmin,
        mmax=mmax,
        break_fraction=break_fraction,
    )
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=gaussian_mass_maximum, low=mmin)
    prob = (1 - lam) * p_pow + lam * p_norm
    return prob


def double_power_law_primary_power_law_mass_ratio(
    dataset, alpha_1, alpha_2, beta, mmin, mmax, break_fraction
):
    r"""
    Power law model for two-dimensional mass distribution, modelling primary
    mass and conditional mass ratio distribution.

    .. math::
        p(m_1, q) = p_{\text{bpl}}(m_1) p(q | m_1)

    .. math::
        p_{\text{bpl}}(m | \alpha_1, m_\min, m_\max, \delta) &\propto \begin{cases}
            m^{-\alpha_1} : m_\min \leq m < m_\min + \delta (m_\max - m_\min)\\
            m^{-\alpha_2} : m_\min + \delta (m_\max - m_\min) \leq m < m_\max
        \end{cases}

    .. math::
        p(q | m_1) \propto q^\beta : \frac{m_1}{m_\min} \leq q \leq 1

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for `mass_1` (:math:`m_1`) and `mass_ratio` (:math:`q`).
    alpha_1: float
        Negative power law exponent for more massive black hole before break (:math:`\alpha_1`).
    alpha_2: float
        Negative power law exponent for more massive black hole after break (:math:`\alpha_2`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
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
    r"""
    Power law model for two-dimensional mass distribution, modelling primary
    mass and conditional mass ratio distribution.

    .. math::
        p(m_1, q) &= p_{\text{pow}}(m_1) p(q | m_1)

        p_{\text{pow}}(m) &\propto m^{-\alpha} : m_\min \leq m < m_\max

        p(q | m_1) &\propto q^\beta : \frac{m_1}{m_\min} \leq q \leq 1

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' (:math:`m_1`) and 'mass_ratio' (:math:`q`).
    alpha: float
        Negative power law exponent for more massive black hole (:math:`\alpha`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
    beta: float
        Power law exponent of the mass ratio distribution (:math:`\beta`).
    """
    return two_component_primary_mass_ratio(
        dataset, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax, lam=0, mpp=35, sigpp=1
    )


def _primary_secondary_general(dataset, p_m1, p_m2):
    return p_m1 * p_m2 * (dataset["mass_1"] >= dataset["mass_2"]) * 2


def power_law_primary_secondary_independent(dataset, alpha, beta, mmin, mmax):
    r"""
    Power law model for two-dimensional mass distribution, modelling the
    primary and secondary masses as following independent distributions.

    .. math::
        p(m1, m2) &= p_{\text{pow}}(m1) p_{\text{pow}}(m2) : m1 \geq m2

        p_{\text{pow}}(m) &\propto m^{-\alpha} : m_\min \leq m < m_\max

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' (:math:`m_1`) and 'mass_2' (:math:`m_2`).
    alpha: float
        Negative power law exponent for more massive black hole (:math:`\alpha`).
    beta: float
        Negative power law exponent of the secondary mass distribution (:math:`\beta`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
    """
    p_m1 = powerlaw(dataset["mass_1"], -alpha, mmax, mmin)
    p_m2 = powerlaw(dataset["mass_2"], -beta, mmax, mmin)
    prob = _primary_secondary_general(dataset, p_m1, p_m2)
    return prob


def power_law_primary_secondary_identical(dataset, alpha, mmin, mmax):
    r"""
    Power law model for two-dimensional mass distribution, modelling the
    primary and secondary masses as following independent distributions.

    .. math::
        p(m_1, m_2 | \alpha, m_\min, m_\max) &= p_{\text{pow}}(m_1 | \alpha) p_{\text{pow}}(m_2 | \alpha) : m_1 \geq m_2

        p_{\text{pow}}(m | \alpha) &\propto m^{-\alpha} : m_\min \leq m < m_\max

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' (:math:`m_1`) and 'mass_2' (:math:`m_2`).
    alpha: float
        Negative power law exponent for both black holes (:math:`\alpha`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
    """
    return power_law_primary_secondary_independent(
        dataset=dataset, alpha=alpha, beta=alpha, mmin=mmin, mmax=mmax
    )


def power_law_mass(mass, alpha, mmin, mmax):
    r"""
    Power law model for one-dimensional mass distribution.

    .. math::
        p(m) \propto m^{-\alpha} : m_\min \leq m < m_\max

    Parameters
    ----------
    mass: array-like
        Array of mass values (:math:`m`).
    alpha: float
        Negative power law exponent for the black hole distribution (:math:`\alpha`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
    """
    return powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)


def two_component_single(
    mass, alpha, mmin, mmax, lam, mpp, sigpp, gaussian_mass_maximum=100
):
    r"""
    Power law model for one-dimensional mass distribution with a Gaussian component.

    .. math::
        p(m) &= (1 - \lambda_m) p_{\text{pow}} + \lambda_m p_{\text{norm}}

        p_{\text{pow}}(m) &\propto m^{-\alpha} : m_\min \leq m < m_\max

        p_{\text{norm}}(m) &\propto \exp\left(-\frac{(m - \mu_{m})^2}{2\sigma^2_m}\right)

    Parameters
    ----------
    mass: array-like
        Array of mass values (:math:`m`).
    alpha: float
        Negative power law exponent for the black hole distribution (:math:`\alpha`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
    lam: float
        Fraction of black holes in the Gaussian component (:math:`\lambda_m`).
    mpp: float
        Mean of the Gaussian component (:math:`\mu_m`).
    sigpp: float
        Standard deviation of the Gaussian component (:math:`\sigma_m`).
    gaussian_mass_maximum: float, optional
        Upper truncation limit of the Gaussian component. (default: 100)
    """
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=gaussian_mass_maximum, low=mmin)
    prob = (1 - lam) * p_pow + lam * p_norm
    return prob


def three_component_single(
    mass,
    alpha,
    mmin,
    mmax,
    lam,
    lam_1,
    mpp_1,
    sigpp_1,
    mpp_2,
    sigpp_2,
    gaussian_mass_maximum=100,
):
    r"""
    Power law model for one-dimensional mass distribution with two Gaussian components.

    .. math::
        p(m) &= (1 - \lambda_m) p_{\text{pow}}(m) + \lambda_m p_{\text{norm}}(m)

        p_{\text{pow}}(m) &\propto m^{-\alpha} : m_\min \leq m < m_\max

        p_{\text{norm}}(m) &\propto \exp\left(-\frac{(m - \mu_{m})^2}{2\sigma^2_m}\right)

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
    gaussian_mass_maximum: float, optional
        Upper truncation limit of the Gaussian component. (default: 100)
        Note that this applies the same value to both.
    """
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm1 = truncnorm(
        mass, mu=mpp_1, sigma=sigpp_1, high=gaussian_mass_maximum, low=mmin
    )
    p_norm2 = truncnorm(
        mass, mu=mpp_2, sigma=sigpp_2, high=gaussian_mass_maximum, low=mmin
    )
    prob = (1 - lam) * p_pow + lam * lam_1 * p_norm1 + lam * (1 - lam_1) * p_norm2
    return prob


def two_component_primary_mass_ratio(
    dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, gaussian_mass_maximum=100
):
    r"""
    Power law model for two-dimensional mass distribution, modelling primary
    mass and conditional mass ratio distribution.

    .. math::
        p(m_1, q) = p(m1) p(q | m_1)

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
        Standard deviation of the Gaussian component.
    gaussian_mass_maximum: float, optional
        Upper truncation limit of the Gaussian component. (default: 100)
    """
    params = dict(
        mmin=mmin,
        mmax=mmax,
        lam=lam,
        mpp=mpp,
        sigpp=sigpp,
        gaussian_mass_maximum=gaussian_mass_maximum,
    )
    p_m1 = two_component_single(dataset["mass_1"], alpha=alpha, **params)
    p_q = powerlaw(dataset["mass_ratio"], beta, 1, mmin / dataset["mass_1"])
    prob = p_m1 * p_q
    return prob


def two_component_primary_secondary_independent(
    dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, gaussian_mass_maximum=100
):
    r"""
    Power law model for two-dimensional mass distribution, modelling the
    primary and secondary masses as following independent distributions.

    .. math::
        p(m_1, m_2) = p_{\text{pow}}(m_1) p_{\text{pow}}(m_2) : m1 \geq m_2

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
        Standard deviation of the Gaussian component.
    gaussian_mass_maximum: float, optional
        Upper truncation limit of the Gaussian component. (default: 100)
    """
    params = dict(
        mmin=mmin,
        mmax=mmax,
        lam=lam,
        mpp=mpp,
        sigpp=sigpp,
        gaussian_mass_maximum=gaussian_mass_maximum,
    )
    p_m1 = two_component_single(dataset["mass_1"], alpha=alpha, **params)
    p_m2 = two_component_single(dataset["mass_2"], alpha=beta, **params)

    prob = _primary_secondary_general(dataset, p_m1, p_m2)
    return prob


def two_component_primary_secondary_identical(
    dataset, alpha, mmin, mmax, lam, mpp, sigpp, gaussian_mass_maximum=100
):
    r"""
    Power law model for two-dimensional mass distribution, modelling the
    primary and secondary masses as following independent distributions.

    .. math::
        p(m_1, m_2) = p(m_1) p(m_2) : m_1 \geq m_2

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
        Standard deviation of the Gaussian component.
    gaussian_mass_maximum: float, optional
        Upper truncation limit of the Gaussian component. (default: 100)
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
        gaussian_mass_maximum=gaussian_mass_maximum,
    )


class BaseSmoothedMassDistribution:
    """
    Generic smoothed mass distribution base class.

    Implements the low-mass smoothing and power-law mass ratio
    distribution. Requires p_m1 to be implemented.

    Parameters
    ==========
    mmin: float
        The minimum mass considered for numerical normalization
    mmax: float
        The maximum mass considered for numerical normalization
    """

    primary_model = None

    @property
    def variable_names(self):
        vars = getattr(
            self.primary_model,
            "variable_names",
            inspect.getfullargspec(self.primary_model).args[1:],
        )
        vars += ["beta", "delta_m"]
        vars = set(vars).difference(self.kwargs.keys())
        return vars

    @property
    def kwargs(self):
        return dict()

    def __init__(self, mmin=2, mmax=100, normalization_shape=(1000, 500), cache=True):
        self.mmin = mmin
        self.mmax = mmax
        self.m1s = xp.linspace(mmin, mmax, normalization_shape[0])
        self.qs = xp.linspace(0.001, 1, normalization_shape[1])
        self.dm = self.m1s[1] - self.m1s[0]
        self.dq = self.qs[1] - self.qs[0]
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.cache = cache

    def __call__(self, dataset, *args, **kwargs):
        beta = kwargs.pop("beta")
        mmin = kwargs.get("mmin", self.mmin)
        mmax = kwargs.get("mmax", self.mmax)
        if "jax" not in xp.__name__:
            if mmin < self.mmin:
                raise ValueError(
                    "{self.__class__}: mmin ({mmin}) < self.mmin ({self.mmin})"
                )
            if mmax > self.mmax:
                raise ValueError(
                    "{self.__class__}: mmax ({mmax}) > self.mmax ({self.mmax})"
                )
        delta_m = kwargs.get("delta_m", 0)
        p_m1 = self.p_m1(dataset, **kwargs, **self.kwargs)
        p_q = self.p_q(dataset, beta=beta, mmin=mmin, delta_m=delta_m)
        prob = p_m1 * p_q
        return prob

    def p_m1(self, dataset, **kwargs):
        mmin = kwargs.get("mmin", self.mmin)
        delta_m = kwargs.pop("delta_m", 0)
        p_m = self.__class__.primary_model(dataset["mass_1"], **kwargs)
        p_m *= self.smoothing(
            dataset["mass_1"], mmin=mmin, mmax=self.mmax, delta_m=delta_m
        )
        norm = self.norm_p_m1(delta_m=delta_m, **kwargs)
        return p_m / norm

    def norm_p_m1(self, delta_m, **kwargs):
        """Calculate the normalisation factor for the primary mass"""
        mmin = kwargs.get("mmin", self.mmin)
        if "jax" not in xp.__name__ and delta_m == 0:
            return 1
        p_m = self.__class__.primary_model(self.m1s, **kwargs)
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=self.mmax, delta_m=delta_m)

        norm = xp.nan_to_num(xp.trapz(p_m, self.m1s)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )
        return norm

    def p_q(self, dataset, beta, mmin, delta_m):
        p_q = powerlaw(dataset["mass_ratio"], beta, 1, mmin / dataset["mass_1"])
        p_q *= self.smoothing(
            dataset["mass_1"] * dataset["mass_ratio"],
            mmin=mmin,
            mmax=dataset["mass_1"],
            delta_m=delta_m,
        )

        try:
            if self.cache:
                p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)
            else:
                self._cache_q_norms(dataset["mass_1"])
                p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)
        except (AttributeError, TypeError, ValueError):
            self._cache_q_norms(dataset["mass_1"])
            p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)

        return xp.nan_to_num(p_q)

    def norm_p_q(self, beta, mmin, delta_m):
        """Calculate the mass ratio normalisation by linear interpolation"""
        p_q = powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid)
        p_q *= self.smoothing(
            self.m1s_grid * self.qs_grid, mmin=mmin, mmax=self.m1s_grid, delta_m=delta_m
        )

        norms = xp.nan_to_num(xp.trapz(p_q, self.qs, axis=0)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )

        return self._q_interpolant(norms)

    def _cache_q_norms(self, masses):
        """
        Cache the information necessary for linear interpolation of the mass
        ratio normalisation
        """
        from .interped import _setup_interpolant

        self._q_interpolant = _setup_interpolant(
            self.m1s, masses, kind="linear", backend=xp
        )

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
        if "jax" in xp.__name__ or delta_m > 0.0:
            shifted_mass = xp.nan_to_num((masses - mmin) / delta_m, nan=0)
            shifted_mass = xp.clip(shifted_mass, 1e-6, 1 - 1e-6)
            exponent = 1 / shifted_mass - 1 / (1 - shifted_mass)
            window = scs.expit(-exponent)
            window *= (masses >= mmin) * (masses <= mmax)
            return window
        else:
            return xp.ones(masses.shape)


class SinglePeakSmoothedMassDistribution(BaseSmoothedMassDistribution):
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
        Standard deviation of the Gaussian component.
    delta_m: float
        Rise length of the low end of the mass distribution.

    Notes
    -----
    The Gaussian component is bounded between [`mmin`, `self.mmax`].
    This means that the `mmax` parameter is _not_ the global maximum.
    """

    primary_model = two_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)


class MultiPeakSmoothedMassDistribution(BaseSmoothedMassDistribution):
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

    Notes
    -----
    The Gaussian components are bounded between [`mmin`, `self.mmax`].
    This means that the `mmax` parameter is _not_ the global maximum.
    """

    primary_model = three_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)


class BrokenPowerLawSmoothedMassDistribution(BaseSmoothedMassDistribution):
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
    delta_m: float
        Rise length of the low end of the mass distribution.
    """

    primary_model = double_power_law_primary_mass


class BrokenPowerLawPeakSmoothedMassDistribution(BaseSmoothedMassDistribution):
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
        Standard deviation of the Gaussian component.
    delta_m: float
        Rise length of the low end of the mass distribution.

    Notes
    -----
    The Gaussian component is bounded between [`mmin`, `self.mmax`].
    This means that the `mmax` parameter is _not_ the global maximum.
    """

    primary_model = double_power_law_peak_primary_mass

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)


class InterpolatedPowerlaw(
    BaseSmoothedMassDistribution, InterpolatedNoBaseModelIdentical
):
    """
    Interpolated powerlaw primary mass distribution with powerlaw mass ratio distribution.

    See https://arxiv.org/abs/2109.06137 for details.

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
    delta_m: float
        Rise length of the low end of the mass distribution.
    mass{ii}: float
        The locations of the spline nodes for the primary mass distribution.
    fmass{ii}: float
        The values of the spline nodes for the primary mass distribution.
    """

    primary_model = power_law_mass

    def __init__(
        self,
        nodes=10,
        kind="cubic",
        mmin=2,
        mmax=100,
        normalization_shape=(1000, 500),
        regularize=False,
    ):
        """
        Parameters
        ==========
        nodes: int
            Number of spline nodes to use for interpolation, default=10.
        kind: str
            Order of the spline to use for interpolation, default="cubic".
        mmin: float
            The minimum mass considered for numerical normalization, default=2.
        mmax: float
            The maximum mass considered for numerical normalization, default=100.
        normalization_shape: tuple
            Shape of the grid used for numerical normalization, default=(1000, 500).
        regularize: bool
            Whether to regularize the spline node values to have root-mean-square value
            :code:`rms{name}`, default=False
        """
        BaseSmoothedMassDistribution.__init__(
            self,
            mmin=mmin,
            mmax=mmax,
            normalization_shape=normalization_shape,
        )
        InterpolatedNoBaseModelIdentical.__init__(
            self,
            minimum=mmin,
            maximum=mmax,
            parameters=["mass_1"],
            nodes=nodes,
            kind=kind,
            log_nodes=True,
            regularize=regularize,
        )
        self._xs = self.m1s

    @property
    def variable_names(self):
        variable_names = super().variable_names.union(
            InterpolatedNoBaseModelIdentical.variable_names.fget(self)
        )
        return variable_names

    def p_m1(self, dataset, **kwargs):

        f_splines, m_splines = self.extract_spline_points(kwargs)

        mmin = kwargs.get("mmin", self.mmin)
        delta_m = kwargs.pop("delta_m", 0)
        p_m = self.__class__.primary_model(
            dataset["mass_1"], **{key: kwargs[key] for key in ["alpha", "mmin", "mmax"]}
        )
        p_m *= self.smoothing(
            dataset["mass_1"], mmin=mmin, mmax=self.mmax, delta_m=delta_m
        )
        p_m *= self.p_x_unnormed(dataset, "mass_1", m_splines, f_splines, **kwargs)

        norm = self.norm_p_m1(delta_m=delta_m, f_splines=f_splines, **kwargs)
        return p_m / norm

    def norm_p_m1(self, delta_m, f_splines=None, **kwargs):
        mmin = kwargs.get("mmin", self.mmin)
        p_m = self.__class__.primary_model(
            self.m1s, **{key: kwargs[key] for key in ["alpha", "mmin", "mmax"]}
        )
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=self.mmax, delta_m=delta_m) ** (
            delta_m > 0
        )
        p_m *= xp.exp(self._norm_spline(y=f_splines))
        norm = xp.trapz(p_m, self.m1s)
        return norm
