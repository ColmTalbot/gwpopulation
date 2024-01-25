"""
Implemented redshift models
"""

import numpy as np

from .interped import InterpolatedNoBaseModelIdentical
from ..utils import to_numpy

xp = np


class _Redshift(object):
    """
    Base class for models which include a term like dVc/dz / (1 + z)
    """

    variable_names = None

    def __init__(self, z_max=2.3):
        from astropy.cosmology import Planck15

        self.z_max = z_max
        self.zs_ = np.linspace(1e-3, z_max, 1000)
        self.zs = xp.asarray(self.zs_)
        self.dvc_dz_ = Planck15.differential_comoving_volume(self.zs_).value * 4 * np.pi
        self.dvc_dz = xp.asarray(self.dvc_dz_)
        self.cached_dvc_dz = None

    def __call__(self, dataset, **kwargs):
        return self.probability(dataset=dataset, **kwargs)

    def _cache_dvc_dz(self, redshifts):
        self.cached_dvc_dz = xp.asarray(
            np.interp(to_numpy(redshifts), self.zs_, self.dvc_dz_, left=0, right=0)
        )

    def normalisation(self, parameters):
        r"""
        Compute the normalization or differential spacetime volume.

        .. math::
            \mathcal{V} = \int dz \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters

        Returns
        -------
        (float, array-like): Total spacetime volume
        """
        psi_of_z = self.psi_of_z(redshift=self.zs, **parameters)
        norm = xp.trapz(psi_of_z * self.dvc_dz / (1 + self.zs), self.zs)
        return norm

    def probability(self, dataset, **parameters):
        differential_volume = self.differential_spacetime_volume(
            dataset=dataset, **parameters
        )
        in_bounds = dataset["redshift"] <= self.z_max
        normalisation = self.normalisation(parameters=parameters)
        return differential_volume / normalisation * in_bounds

    def psi_of_z(self, redshift, **parameters):
        raise NotImplementedError

    def differential_spacetime_volume(self, dataset, **parameters):
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
        try:
            differential_volume *= self.cached_dvc_dz
        except (TypeError, ValueError):
            self._cache_dvc_dz(dataset["redshift"])
            differential_volume *= self.cached_dvc_dz
        return differential_volume


class PowerLawRedshift(_Redshift):
    r"""
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270

    .. math::
        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\gamma, \kappa, z_p)

        \psi(z|\gamma, \kappa, z_p) &= (1 + z)^\lambda

    Parameters
    ----------
    lamb: float
        The spectral index.
    """

    variable_names = ["lamb"]

    def psi_of_z(self, redshift, **parameters):
        return (1 + redshift) ** parameters["lamb"]
    


class InterpolatedRedshift(_Redshift, InterpolatedNoBaseModelIdentical):

    def __init__(
        self,
        zmax,
        nodes=10,
        kind="cubic",
        log_nodes=False,
        regularize=False,
    ):
        """ """
        _Redshift.__init__(self, z_max=zmax)
        InterpolatedNoBaseModelIdentical.__init__(
            self,
            parameters=["redshift"],
            minimum=0,
            maximum=zmax,
            nodes=nodes,
            kind=kind,
            log_nodes=log_nodes,
            regularize=regularize,
        )
        self._xs = self.zs_

    @property
    def variable_names(self):
        return InterpolatedNoBaseModelIdentical.variable_names.fget(self)

    def normalisation(self, parameters):
        r"""
        Compute the normalization or differential spacetime volume.

        .. math::
            \mathcal{V} = \int dz \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters

        Returns
        -------
        (float, array-like): Total spacetime volume
        """
        f_splines = xp.array([parameters[key] for key in self.fkeys])
        x_splines = xp.array([parameters[key] for key in self.xkeys])

        psi_of_z = xp.exp(self._norm_spline(y=f_splines))
        psi_of_z *= (self._xs >= x_splines[0]) & (self._xs <= x_splines[-1])
        norm = xp.trapz(psi_of_z * self.dvc_dz / (1 + self.zs), self.zs)
        return norm

    def psi_of_z(self, redshift, **parameters):
        self.infer_n_nodes(**parameters)

        f_splines = xp.array([parameters[key] for key in self.fkeys])
        x_splines = xp.array([parameters[key] for key in self.xkeys])

        return self.p_x_unnormed(
            dict(redshift=redshift), "redshift", x_splines=x_splines, f_splines=f_splines, **parameters
        )


class MadauDickinsonRedshift(_Redshift):
    r"""
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270 (33)
    See https://arxiv.org/abs/2003.12152 (2) for the normalisation

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

    variable_names = ["gamma", "kappa", "z_peak"]

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
    from astropy.cosmology import Planck15

    redshifts = np.linspace(0, max_redshift, 1000)
    psi_of_z = (1 + redshifts) ** lamb
    normalization = 4 * np.pi / 1e9 * analysis_time
    total_volume = (
        np.trapz(
            Planck15.differential_comoving_volume(redshifts).value
            / (1 + redshifts)
            * psi_of_z,
            redshifts,
        )
        * normalization
    )
    return total_volume
