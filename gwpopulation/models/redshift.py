from .. cupy_utils import to_numpy, trapz, xp
from ..utils import powerlaw

import numpy as np

from astropy.cosmology import Planck15


class PowerLawRedshift(object):
    """
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270
    Note that this is deliberately off by a factor of dVc/dz
    """

    def __init__(self):
        self.zs_ = np.linspace(1e-3, 1, 1000)
        self.zs = xp.asarray(self.zs_)
        self.dvc_dz_ = (
            Planck15.differential_comoving_volume(self.zs_).value * 4 * np.pi)
        self.dvc_dz = xp.asarray(self.dvc_dz_)
        self.cached_dvc_dz = None

    def __call__(self, dataset, lamb):
        p_z = powerlaw(1 + dataset['redshift'], alpha=(lamb - 1),
                       high=(1 + self.zs_[-1]), low=1)
        try:
            p_z *= self.cached_dvc_dz
        except (TypeError, ValueError):
            self._cache_dvc_dz(dataset['redshift'])
            p_z *= self.cached_dvc_dz
        p_z /= self.normalisation(lamb)
        return p_z

    def normalisation(self, lamb):
        p_z_ = powerlaw(1 + self.zs, alpha=(lamb - 1),
                        high=(1 + self.zs_[-1]), low=1)
        norm = trapz(p_z_ * self.dvc_dz, self.zs)
        return norm

    def _cache_dvc_dz(self, redshifts):
        self.cached_dvc_dz = xp.asarray(np.interp(
            to_numpy(redshifts), self.zs_, self.dvc_dz_))


power_law_redshift = PowerLawRedshift()
