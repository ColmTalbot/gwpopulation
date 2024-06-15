"""
Implemented redshift models
"""

import numpy as xp

from ..experimental.cosmo_models import CosmoMixin

__all__ = [
    "_Redshift",
    "PowerLawRedshift",
    "MadauDickinsonRedshift",
    "total_four_volume",
]


class _Redshift(CosmoMixin):
    r"""
    Base redshift model class.

    This assumes the model is defined as

    .. math::

        p(z | \Lambda) = \frac{1}{(1 + z)} \frac{dVc}{dz} \psi(z | \Lambda).

    Subclasses define :math:`\psi(z | \Lambda)` as :func:`_Redshift.psi_of_z`.

    Attributes
    ----------
    base_variable_names: list
        :math:`\Lambda` - list of astrophysical rate-evolution parameters
        for the model.
    """

    base_variable_names = None

    @property
    def variable_names(self):
        """
        Variable names for the model

        Returns
        -------
        vars: list
            Variable names including astrophysical rate-evolution parameters
            and cosmological parameters.
        """
        vars = self.cosmology_names.copy()
        if self.base_variable_names is not None:
            vars += self.base_variable_names
        return vars

    def __init__(self, z_max=2.3, cosmo_model="Planck15"):
        super().__init__(cosmo_model=cosmo_model)
        self.z_max = z_max
        self.zs = xp.linspace(1e-6, z_max, 2500)

    def __call__(self, dataset, **kwargs):
        return self.probability(dataset=dataset, **kwargs)

    def normalisation(self, parameters):
        r"""
        Compute the normalization of the rate-weighted spacetime volume.

        .. math::

            \mathcal{V} = \int dz \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters

        Returns
        -------
        norm: float | array-like:
            Total rate-weighted comoving spacetime volume
        """
        normalisation_data = self.differential_spacetime_volume(
            dict(redshift=self.zs), bounds=True, **parameters
        )
        norm = xp.trapz(normalisation_data, self.zs)
        return norm

    def probability(self, dataset, **parameters):
        normalisation = self.normalisation(parameters=parameters)
        differential_volume = self.differential_spacetime_volume(
            dataset=dataset, bounds=True, **parameters
        )
        return differential_volume / normalisation

    def psi_of_z(self, redshift, **parameters):
        raise NotImplementedError

    def dvc_dz(self, redshift, **parameters):
        return (
            4
            * xp.pi
            * self.cosmology(parameters).differential_comoving_volume(redshift)
        )

    def differential_spacetime_volume(self, dataset, bounds=False, **parameters):
        r"""
        Compute the differential spacetime volume.

        .. math::

            d\mathcal{V} = \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)

        Parameters
        ----------
        dataset: dict
            Dictionary containing entry "redshift"
        parameters: dict
            Dictionary of parameters

        Returns
        -------
        differential_volume: (float, array-like)
            Differential spacetime volume
        """
        psi_of_z = self.psi_of_z(redshift=dataset["redshift"], **parameters)
        differential_volume = psi_of_z / (1 + dataset["redshift"])
        differential_volume *= self.dvc_dz(redshift=dataset["redshift"], **parameters)
        if bounds:
            differential_volume *= dataset["redshift"] <= self.z_max

        return differential_volume


class PowerLawRedshift(_Redshift):
    r"""
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270
    (`arXiv:1805.10270 <https://arxiv.org/abs/1805.10270>`_
    and Cosmo model :func:`FlatLambdaCDM`.

    .. math::

        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\gamma, \kappa, z_p)

        \psi(z|\gamma, \kappa, z_p) &= (1 + z)^\lambda

    Parameters
    ----------
    lamb: float
        The spectral index.
    """
    base_variable_names = ["lamb"]

    def psi_of_z(self, redshift, **parameters):
        return (1 + redshift) ** parameters["lamb"]


class MadauDickinsonRedshift(_Redshift):
    r"""
    Redshift model from Fishbach+
    (`arXiv:1805.10270 <https://arxiv.org/abs/1805.10270>`_ Eq. (33))
    See Callister+ (`arXiv:2003.12152 <https://arxiv.org/abs/2003.12152>`_
    Eq. (2)) for the normalisation.

    The parameterisation differs a little from there, we use

    .. math::

        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\gamma, \kappa, z_p)

        \psi(z|\gamma, \kappa, z_p) &= \frac{(1 + z)^\gamma}{1 + (\frac{1 + z}{1 + z_p})^\kappa}

    Parameters
    ----------
    gamma: float
        Slope of the distribution at low redshift
    kappa: float
        Slope of the distribution at high redshift
    z_peak: float
        Redshift at which the distribution peaks.
    z_max: float, optional
        The maximum redshift allowed.
    """
    base_variable_names = ["gamma", "kappa", "z_peak"]

    def psi_of_z(self, redshift, **parameters):
        gamma = parameters["gamma"]
        kappa = parameters["kappa"]
        z_peak = parameters["z_peak"]
        psi_of_z = (1 + redshift) ** gamma / (
            1 + ((1 + redshift) / (1 + z_peak)) ** kappa
        )
        psi_of_z *= 1 + (1 + z_peak) ** (-kappa)
        return psi_of_z


def total_four_volume(lamb, analysis_time, max_redshift=2.3):
    from wcosmo.wcosmo import Planck15

    redshifts = xp.linspace(0, max_redshift, 2500)
    psi_of_z = (1 + redshifts) ** lamb
    normalization = 4 * xp.pi / 1e9 * analysis_time
    total_volume = (
        xp.trapz(
            Planck15.differential_comoving_volume(redshifts)
            / (1 + redshifts)
            * psi_of_z,
            redshifts,
        )
        * normalization
    )
    return total_volume
