from .. cupy_utils import to_numpy, trapz, xp
from ..utils import powerlaw

import numpy as np

from astropy.cosmology import Planck15


class _Redshift(object):
    """
    Base class for models which include a term like dVc/dz / (1 + z)
    """

    def __init__(self, z_max=1):
        self.z_max = z_max
        self.zs_ = np.linspace(1e-3, z_max, 1000)
        self.zs = xp.asarray(self.zs_)
        self.dvc_dz_ = (
            Planck15.differential_comoving_volume(self.zs_).value * 4 * np.pi)
        self.dvc_dz = xp.asarray(self.dvc_dz_)
        self.cached_dvc_dz = None

    def _cache_dvc_dz(self, redshifts):
        self.cached_dvc_dz = xp.asarray(np.interp(
            to_numpy(redshifts), self.zs_, self.dvc_dz_))


class PowerLawRedshift(_Redshift):
    """
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270

    Parameters
    ----------
    z_max: float, optional
        The maximum redshift allowed.
    """
    def __call__(self, dataset, lamb):
        psi_of_z = powerlaw(
            1 + dataset['redshift'], alpha=lamb, high=self.z_max, low=1)
        normalisation = self.normalisation(lamb)
        p_z = psi_of_z / (1 + dataset['redshift']) / normalisation
        try:
            p_z *= self.cached_dvc_dz
        except (TypeError, ValueError):
            self._cache_dvc_dz(dataset['redshift'])
            p_z *= self.cached_dvc_dz
        return p_z

    def normalisation(self, lamb):
        psi_of_z = powerlaw(1 + self.zs, alpha=lamb, high=self.z_max, low=1)
        norm = trapz(psi_of_z * self.dvc_dz / (1 + self.zs), self.zs)
        return norm


class MaduaDickinsonRedshift(_Redshift):
    """
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270

    Parameters
    ----------
    z_max: float, optional
        The maximum redshift allowed.
    """

    def __call__(self, dataset, a_z, b_z, z_peak):
        psi_of_z = powerlaw(
            1 + dataset['redshift'], alpha=a_z, high=self.z_max, low=1)
        psi_of_z /= 1 + a_z / (b_z - a_z) / (1 + z_peak)**b_z * powerlaw(
            1 + dataset['redshift'], alpha=b_z, high=self.z_max, low=1)
        normalisation = self.normalisation(a_z, b_z, z_peak)
        p_z = psi_of_z / (1 + dataset['redshift']) / normalisation
        try:
            p_z *= self.cached_dvc_dz
        except (TypeError, ValueError):
            self._cache_dvc_dz(dataset['redshift'])
            p_z *= self.cached_dvc_dz
        return p_z

    def normalisation(self, a_z, b_z, z_peak):
        psi_of_z = powerlaw(
            1 + self.zs, alpha=a_z, high=self.z_max, low=1)
        psi_of_z /= 1 + a_z / (b_z - a_z) / (1 + z_peak)**b_z * powerlaw(
            1 + self.zs, alpha=b_z, high=self.z_max, low=1)
        norm = trapz(psi_of_z * self.dvc_dz / (1 + self.zs), self.zs)
        return norm


power_law_redshift = PowerLawRedshift()
