from ..cupy_utils import to_numpy, trapz, xp
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
        self.dvc_dz_ = Planck15.differential_comoving_volume(self.zs_).value * 4 * np.pi
        self.dvc_dz = xp.asarray(self.dvc_dz_)
        self.cached_dvc_dz = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _cache_dvc_dz(self, redshifts):
        self.cached_dvc_dz = xp.asarray(
            np.interp(to_numpy(redshifts), self.zs_, self.dvc_dz_)
        )

    def normalisation(self, parameters):
        psi_of_z = self.psi_of_z(redshift=self.zs, **parameters)
        norm = trapz(psi_of_z * self.dvc_dz / (1 + self.zs), self.zs)
        return norm

    def probability(self, dataset, **parameters):
        psi_of_z = self.psi_of_z(redshift=dataset["redshift"], **parameters)
        normalisation = self.normalisation(parameters=parameters)
        p_z = psi_of_z / (1 + dataset["redshift"]) / normalisation
        try:
            p_z *= self.cached_dvc_dz
        except (TypeError, ValueError):
            self._cache_dvc_dz(dataset["redshift"])
            p_z *= self.cached_dvc_dz
        return p_z

    def psi_of_z(self, redshift, **parameters):
        raise NotImplementedError


class PowerLawRedshift(_Redshift):
    """
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270

    Parameters
    ----------
    z_max: float, optional
        The maximum redshift allowed.
    """

    def __call__(self, dataset, lamb):
        parameters = dict(lamb=lamb)
        return self.probability(dataset=dataset, **parameters)

    def psi_of_z(self, redshift, **parameters):
        return powerlaw(
            1 + redshift, alpha=parameters["lamb"], high=1 + self.z_max, low=1
        )


class MaduaDickinsonRedshift(_Redshift):
    """
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270 (33)

    Parameters
    ----------
    a_z: float
        Slope of the distribution at low redshift
    b_z: float
        Slope of the distribution at high redshift
    z_peak: float
        Redshift at which the distribution peaks.
    z_max: float, optional
        The maximum redshift allowed.
    """

    def __call__(self, dataset, a_z, b_z, z_peak):
        parameters = dict(a_z=a_z, b_z=b_z, z_peak=z_peak)
        return self.probability(dataset=dataset, **parameters)

    def psi_of_z(self, redshift, **parameters):
        a_z = parameters["a_z"]
        b_z = parameters["b_z"]
        z_peak = parameters["z_peak"]
        psi_of_z = powerlaw(1 + redshift, alpha=a_z, high=1 + self.z_max, low=1)
        psi_of_z /= 1 + a_z / (b_z - a_z) / (1 + z_peak) ** b_z * powerlaw(
            1 + redshift, alpha=b_z, high=1 + self.z_max, low=1
        )
        return psi_of_z


power_law_redshift = PowerLawRedshift()
