"""
Implemented spin models
"""

import numpy as xp

from ..utils import beta_dist, truncnorm, unnormalized_2d_gaussian
from .interped import InterpolatedNoBaseModelIdentical

__all__ = [
    "GaussianChiEffChiP",
    "SplineSpinMagnitudeIdentical",
    "SplineSpinTiltIdentical",
    "iid_spin",
    "iid_spin_magnitude_beta",
    "independent_spin_magnitude_beta",
    "iid_spin_orientation_gaussian_isotropic",
    "independent_spin_orientation_gaussian_isotropic",
    "gaussian_chi_eff",
    "gaussian_chi_p",
]


def iid_spin(dataset, xi_spin, sigma_spin, amax, alpha_chi, beta_chi):
    r"""
    Independently and identically distributed spins.
    The magnitudes are assumed to follow a Beta distribution and the
    orientations are assumed to follow an isotropic + truncated half
    Gaussian mixture model.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays containing 'a_1' and 'a_2'.
    xi_spin: float
        Fraction of black holes in preferentially aligned component.
    sigma_spin: float
        Width of preferentially aligned component.
    alpha_chi, beta_chi: float
        Parameters of Beta distribution for both black holes.
    amax: float
        Maximum black hole spin.
    """
    prior = iid_spin_orientation_gaussian_isotropic(
        dataset, xi_spin, sigma_spin
    ) * iid_spin_magnitude_beta(dataset, amax, alpha_chi, beta_chi)
    return prior


def iid_spin_magnitude_beta(dataset, amax=1, alpha_chi=1, beta_chi=1):
    """
    Independent and identically distributed beta distributions for both spin magnitudes.

    See `Wysocki+ <https://arxiv.org/abs/1805.06442>` Eq. (10)

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays containing 'a_1' and 'a_2'.
    alpha_chi, beta_chi: float
        Parameters of Beta distribution for both black holes.
    amax: float
        Maximum black hole spin.
    """
    return independent_spin_magnitude_beta(
        dataset, alpha_chi, alpha_chi, beta_chi, beta_chi, amax, amax
    )


def independent_spin_magnitude_beta(
    dataset, alpha_chi_1, alpha_chi_2, beta_chi_1, beta_chi_2, amax_1, amax_2
):
    """
    Independent beta distributions for both spin magnitudes.

    See `Wysocki+ <https://arxiv.org/abs/1805.06442>` Eq. (10)

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays containing 'a_1' and 'a_2'.
    alpha_chi_1, beta_chi_1: float
        Parameters of Beta distribution for more massive black hole.
    alpha_chi_2, beta_chi_2: float
        Parameters of Beta distribution for less massive black hole.
    amax_1, amax_2: float
        Maximum spin of the more/less massive black hole.
    """
    prior = beta_dist(
        dataset["a_1"], alpha_chi_1, beta_chi_1, scale=amax_1
    ) * beta_dist(dataset["a_2"], alpha_chi_2, beta_chi_2, scale=amax_2)
    return prior


def iid_spin_orientation_gaussian_isotropic(dataset, xi_spin, sigma_spin):
    r"""
    A mixture model of spin orientations with isotropic and normally
    distributed components. The distribution of primary and secondary spin
    orientations are expected to be identical and independent.

    See `Talbot and Thrane <https://arxiv.org/abs/1704.08370>`_ Eq. (4)

    .. math::

        p(z_1, z_2 | \xi, \sigma) =
        \frac{(1 - \xi)^2}{4}
        + \xi \prod_{i\in\{1, 2\}} \mathcal{N}_{[-1, 1]}(z_i; \mu=1, \sigma=\sigma)

    Where :math:`\mathcal{N}_{[a, b]}` is the truncated normal distribution over :math:`[a, b]`.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'cos_tilt_1' and 'cos_tilt_2'.
    xi_spin: float
        Fraction of black holes in preferentially aligned component (:math:`\xi`).
    sigma_spin: float
        Width of preferentially aligned component.
    """
    return independent_spin_orientation_gaussian_isotropic(
        dataset, xi_spin, sigma_spin, sigma_spin
    )


def independent_spin_orientation_gaussian_isotropic(dataset, xi_spin, sigma_1, sigma_2):
    r"""
    A mixture model of spin orientations with isotropic and normally
    distributed components.

    See `Talbot and Thrane <https://arxiv.org/abs/1704.08370>`_ Eq. (4)

    .. math::

        p(z_1, z_2 | \xi, \sigma) =
        \frac{(1 - \xi)^2}{4}
        + \xi \prod_{i\in\{1, 2\}} \mathcal{N}_{[-1, 1]}(z_i; \mu=1, \sigma=\sigma_{i})

    Where :math:`\mathcal{N}_{[a, b]}` is the truncated normal distribution over :math:`[a, b]`.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'cos_tilt_1' and 'cos_tilt_2'.
    xi_spin: float
        Fraction of black holes in preferentially aligned component (:math:`\xi`).
    sigma_1: float
        Width of preferentially aligned component for the more
        massive black hole (:math:`\sigma_1`).
    sigma_2: float
        Width of preferentially aligned component for the less
        massive black hole (:math:`\sigma_2`).
    """
    prior = (1 - xi_spin) / 4 + xi_spin * truncnorm(
        dataset["cos_tilt_1"], 1, sigma_1, 1, -1
    ) * truncnorm(dataset["cos_tilt_2"], 1, sigma_2, 1, -1)
    return prior


def gaussian_chi_eff(dataset, mu_chi_eff, sigma_chi_eff):
    r"""
    A Gaussian in chi effective distribution


    .. math::
        p(\chi_{\text{eff}} | \mu_\chi, \sigma_\chi) =
        \mathcal{N}_{[-1, 1]}(\chi_{\text{eff}}; \mu=\mu_\chi, \sigma=\sigma_\chi)

    Where :math:`\mathcal{N}_{[a, b]}` is the truncated normal distribution over :math:`[a, b]`.

    See `Miller+ <https://arxiv.org/abs/2001.06051>`_ and `Callister+ <https://arxiv.org/abs/2010.14533>`_.

    Parameters
    ----------
    dataset: dict
        Input data, must contain `chi_eff` (:math:`\chi_{\text{eff}}`)
    mu_chi_eff: float
        Mean of the distribution (:math:`\mu_\chi`)
    sigma_chi_eff: float
        Standard deviation of the distribution (:math:`\sigma_\chi`)

    Returns
    -------
    array-like: The probability
    """
    return truncnorm(
        dataset["chi_eff"], mu=mu_chi_eff, sigma=sigma_chi_eff, low=-1, high=1
    )


def gaussian_chi_p(dataset, mu_chi_p, sigma_chi_p):
    r"""
    A Gaussian distribution in precessing effective spin (chi p)

    .. math::
        p(\chi_p) = \mathcal{N}_{[0, 1]}(\chi_p; \mu=\mu_\chi, \sigma=\sigma_\chi)

    Where :math:`\mathcal{N}_{[a, b]}` is the truncated normal distribution over :math:`[a, b]`.

    See `Miller+ <https://arxiv.org/abs/2001.06051>`_ and `Callister+ <https://arxiv.org/abs/2010.14533>`_.

    Parameters
    ----------
    dataset: dict
        Input data, must contain `chi_eff` (:math:`\chi_p`)
    mu_chi_p: float
        Mean of the distribution (:math:`\mu_\chi`)
    sigma_chi_p: float
        Standard deviation of the distribution (:math:`\sigma_\chi`)

    Returns
    -------
    array-like: The probability
    """
    return truncnorm(dataset["chi_p"], mu=mu_chi_p, sigma=sigma_chi_p, low=0, high=1)


class GaussianChiEffChiP(object):
    r"""
    A covariant Gaussian in effective aligned and precessing spins.

    .. math::

        p(\chi_{\rm eff}, \chi_p | \Lambda) = \mathcal{N}_{[-1, 1], [0, 1]}(\chi_{\rm eff}, \chi_p; [\mu_{\rm eff}, \mu_{p}], \Sigma)

    Where :math:`\mathcal{N}_{[a, b], [c, d]}` is the two-dimensional truncated normal distribution
    over :math:`[a, b]` and :math:`[c, d]`.

    The covariance matrix is given by:

    .. math::
        \Sigma = \begin{bmatrix}
            \sigma^2_{\text{eff}} & \rho \sigma_{\text{eff}} \sigma_{p} \\
            \rho \sigma_{\text{eff}} \sigma_{p} & \sigma^2_{p}
        \end{bmatrix}

    See `Miller+ <https://arxiv.org/abs/2001.06051>`_ and `Callister+ <https://arxiv.org/abs/2010.14533>`_.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for :code:`chi_eff` and :code:`chi_p`.
    mu_chi_eff: float
        Mean of the chi effective distribution (:math:`\mu_{\text{eff}}`)
    mu_chi_p: float
        Mean of the chi p distribution (:math:`\mu_{p}`)
    sigma_chi_eff: float
        Standard deviation of the chi effective distribution (:math:`\sigma_{\text{eff}}`)
    sigma_chi_p: float
        Standard deviation of the chi p distribution (:math:`\sigma_{p}`)
    spin_covariance: float
        Covariance between the two parameters (:math:`\rho`)
    """

    def __init__(self):
        self.chi_eff = xp.linspace(-1, 1, 500)
        self.chi_p = xp.linspace(0, 1, 250)
        self.chi_eff_grid, self.chi_p_grid = xp.meshgrid(self.chi_eff, self.chi_p)

    def __call__(
        self, dataset, mu_chi_eff, sigma_chi_eff, mu_chi_p, sigma_chi_p, spin_covariance
    ):
        if spin_covariance == 0:
            prob = gaussian_chi_eff(
                dataset=dataset,
                mu_chi_eff=mu_chi_eff,
                sigma_chi_eff=sigma_chi_eff,
            )
            prob *= gaussian_chi_p(
                dataset=dataset, mu_chi_p=mu_chi_p, sigma_chi_p=sigma_chi_p
            )
        else:
            prob = unnormalized_2d_gaussian(
                dataset["chi_eff"],
                dataset["chi_p"],
                mu_chi_eff,
                mu_chi_p,
                sigma_chi_eff,
                sigma_chi_p,
                spin_covariance,
            )
            normalization = self._normalization(
                mu_chi_eff=mu_chi_eff,
                sigma_chi_eff=sigma_chi_eff,
                mu_chi_p=mu_chi_p,
                sigma_chi_p=sigma_chi_p,
                spin_covariance=spin_covariance,
            )
            prob /= normalization
            prob *= xp.abs(dataset["chi_eff"]) <= 1
            prob *= (dataset["chi_p"] <= 1) * (dataset["chi_p"] >= 0)
        return prob

    def _normalization(
        self, mu_chi_eff, sigma_chi_eff, mu_chi_p, sigma_chi_p, spin_covariance
    ):
        r"""
        Numerically calculate the normalization over a two-dimensional grid with
        trapezoidal integration

        Parameters
        ----------
        mu_chi_eff: float
            Mean of the chi effective distribution (:math:`\mu_{\text{eff}}`)
        mu_chi_p: float
            Mean of the chi p distribution (:math:`\mu_{p}`)
        sigma_chi_eff: float
            Standard deviation of the chi effective distribution (:math:`\sigma_{\text{eff}}`)
        sigma_chi_p: float
            Standard deviation of the chi p distribution (:math:`\sigma_{p}`)
        spin_covariance: float
            Covariance between the two parameters (:math:`\rho`)

        Returns
        -------
        float
            The normalizing constant
        """
        prob = unnormalized_2d_gaussian(
            self.chi_eff_grid,
            self.chi_p_grid,
            mu_chi_eff,
            mu_chi_p,
            sigma_chi_eff,
            sigma_chi_p,
            spin_covariance,
        )
        return xp.trapz(
            y=xp.trapz(y=prob, axis=-1, x=self.chi_eff), axis=-1, x=self.chi_p
        )


class SplineSpinMagnitudeIdentical(InterpolatedNoBaseModelIdentical):
    """
    Interpolated spline model for spin magnitudes.

    See `Golomb and Talbot <https://arxiv.org/abs/2210.12287>`_

    Parameters
    ----------
    minimum: float
        Minimum value to normalize the spline over, default=0.
    maximum: float
        Maximum value to normalize the spline over, default=1.
    nodes: int
        Number of nodes to use in the spline, default=5.
    kind: str
        The interpolation order of the spline, default=”cubic”.
    regularize: bool
        Whether to regularize the spline node values to have root-mean-square
        value :code:`rms{name}`, default=False.
    """

    def __init__(self, minimum=0, maximum=1, nodes=5, kind="cubic", regularize=False):

        super(SplineSpinMagnitudeIdentical, self).__init__(
            parameters=["a_1", "a_2"],
            minimum=minimum,
            maximum=maximum,
            nodes=nodes,
            kind=kind,
            regularize=regularize,
        )


class SplineSpinTiltIdentical(InterpolatedNoBaseModelIdentical):
    """
    Interpolated spline model for spin orientations.

    See `Golomb and Talbot <https://arxiv.org/abs/2210.12287>`_

    Parameters
    ----------
    minimum: float
        Minimum value to normalize the spline over, default=-1.
    maximum: float
        Maximum value to normalize the spline over, default=1.
    nodes: int
        Number of nodes to use in the spline, default=5.
    kind: str
        The interpolation order of the spline, default=”cubic”.
    regularize: bool
        Whether to regularize the spline node values to have root-mean-square
        value :code:`rms{name}`, default=False.
    """

    def __init__(self, minimum=-1, maximum=1, nodes=5, kind="cubic", regularize=False):

        super(SplineSpinTiltIdentical, self).__init__(
            parameters=["cos_tilt_1", "cos_tilt_2"],
            minimum=minimum,
            maximum=maximum,
            nodes=nodes,
            kind=kind,
            regularize=regularize,
        )
