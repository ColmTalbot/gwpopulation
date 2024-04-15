import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.cosmology import z_at_value
from scipy.interpolate import splev
from scipy.interpolate import splrep
from astropy import constants

from gwpopulation.utils import to_numpy

xp = np

class _CosmoRedshift(object):
    """
    Base class for models which include a term like dVc/dz / (1 + z) with flexible cosmology model
    """

    variable_names = None

    def __init__(self, z_max=2.3):

        self.z_max = z_max
        self.zs_ = np.linspace(1e-6, z_max, 2500)
        self.zs = xp.asarray(self.zs_)

    def __call__(self, dataset, **kwargs):
        return self.probability(dataset=dataset, **kwargs)

    # def normalisation(self, parameters):
    #     r"""
    #     Compute the normalization or differential spacetime volume.
    #     .. math::
    #         \mathcal{V} = \int dz \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)
    #     Parameters
    #     ----------
    #     parameters: dict
    #         Dictionary of parameters
    #     Returns
    #     -------
    #     (float, array-like): Total spacetime volume
    #     """
    #     psi_of_z = self.psi_of_z(redshift=self.zs, **parameters)
    #     dvc_dz = self.dvc_dz(redshift=self.zs, **parameters)
    #     norm = xp.trapz(psi_of_z * dvc_dz / (1 + self.zs), self.zs)
    #     return norm

    def probability(self, dataset, **parameters):
        #normalization factor
        psi_of_z = self.psi_of_z(redshift=self.zs, **parameters)
        dvc_dz = self.dvc_dz(redshift=self.zs, **parameters)
        norm = xp.trapz(psi_of_z * dvc_dz / (1 + self.zs), self.zs)
        
        differential_volume = self.psi_of_z(redshift=dataset["redshift"], **parameters)/(1 + dataset["redshift"])
        differential_volume *= xp.reshape(xp.interp(xp.ravel(dataset["redshift"]),self.zs,dvc_dz), dataset["redshift"].shape)
        
        # normalisation = self.normalisation(parameters=parameters)
        # differential_volume = self.differential_spacetime_volume(
        #     dataset=dataset, **parameters
        # )
        in_bounds = dataset["redshift"] <= self.z_max
        return differential_volume / normalisation * in_bounds

    def psi_of_z(self, redshift, **parameters):
        raise NotImplementedError

#     def differential_spacetime_volume(self, dataset, **parameters):
#         r"""
#         Compute the differential spacetime volume.
#         .. math::
#             d\mathcal{V} = \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)
#         Parameters
#         ----------
#         dataset: dict
#             Dictionary containing entry "redshift"
#         parameters: dict
#             Dictionary of parameters
#         Returns
#         -------
#         differential_volume: (float, array-like)
#             Differential spacetime volume
#         """
#         psi_of_z = self.psi_of_z(redshift=dataset["redshift"], **parameters)
#         differential_volume = psi_of_z / (1 + dataset["redshift"])
#         differential_volume *= self.dvc_dz(redshift=dataset["redshift"], **parameters)

#         return differential_volume



class CosmoPowerLawRedshift(_CosmoRedshift):
    r"""
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270 and Cosmo model FlatLambdaCDM
    .. math::
        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\gamma, \kappa, z_p)
        \psi(z|\gamma, \kappa, z_p) &= (1 + z)^\lambda
    Parameters
    ----------
    lamb: float
        The spectral index.
    """

    variable_names = ["lamb","H0","Om0"]

    def psi_of_z(self, redshift, **parameters):
        return (1 + redshift) ** parameters["lamb"]

    def astropy_cosmology(self, **parameters):
        Om0 = parameters['Om0']
        H0 = parameters['H0']
        return FlatLambdaCDM(Om0=Om0,H0=H0)    

    def dvc_dz(self, redshift, **parameters):

        astropy_cosmology = self.astropy_cosmology(**parameters)
        dvc_dz =  xp.asarray(4*xp.pi*astropy_cosmology.differential_comoving_volume(to_numpy(redshift)).value)

        return dvc_dz

    def detector_frame_to_source_frame(self, data, H0, Om0, astropy_conv=False):

        cosmo = self.astropy_cosmology(H0=H0,Om0=Om0)

        samples = dict()
        if astropy_conv == True:

            samples['redshift'] = xp.asarray([z_at_value(cosmo.luminosity_distance, d*u.Mpc,zmax=self.z_max) for d in to_numpy(data['luminosity_distance'])])
            samples['mass_1'] = data['mass_1']/(1+samples['redshift'])
            if 'mass_2' in samples:
                samples['mass_2'] = data['mass_2']/(1+samples['redshift'])
                samples['mass_ratio'] = samples['mass_2']/samples['mass_1']
            else:
                samples['mass_ratio'] = data['mass_ratio']
            try:
                samples['a_1'] = data['a_1']
                samples['a_2'] = data['a_2']
            except:
                None
            try:
                samples['cos_tilt_1'] = data['cos_tilt_1']
                samples['cos_tilt_2'] = data['cos_tilt_2']
            except:
                None
        else:
            zs = to_numpy(self.zs)
            dl = cosmo.luminosity_distance(to_numpy(zs)).value
            interp_dl_to_z = splrep(dl,zs,s=0)

            samples['redshift'] = xp.nan_to_num(xp.asarray(splev(to_numpy(data['luminosity_distance']),interp_dl_to_z,ext=0)))
            samples['mass_1'] = data['mass_1']/(1+samples['redshift'])
            if 'mass_2' in samples:
                samples['mass_2'] = data['mass_2']/(1+samples['redshift'])
                samples['mass_ratio'] = samples['mass_2']/samples['mass_1']
            else:
                samples['mass_ratio'] = data['mass_ratio']
            try:
                samples['a_1'] = data['a_1']
                samples['a_2'] = data['a_2']
            except:
                None
            try:
                samples['cos_tilt_1'] = data['cos_tilt_1']
                samples['cos_tilt_2'] = data['cos_tilt_2']
            except:
                None

        return samples

    def detector_to_source_jacobian(self, z, H0, Om0, dl):

        """
        Calculates the detector frame to source frame Jacobian d_det/d_sour for dL and z
        Parameters
        ----------
        z: _np. arrays
            Redshift
        cosmo:  class from the cosmology module
            Cosmology class from the cosmology module
        """
        cosmo = self.astropy_cosmology(H0=H0,Om0=Om0)

        speed_of_light = constants.c.to('km/s').value
        # Calculate the Jacobian of the luminosity distance w.r.t redshift

        # dL_by_dz = dl/(1+z) + speed_of_light*(1+z)/(cosmo.H0.value*self.Efunc(cosmo, z))
        dL_by_dz = dl/(1+z) + speed_of_light*(1+z)/cosmo.H(z).value
        
        return dL_by_dz

    def Efunc(self, cosmo, z):

        return xp.sqrt(cosmo.Om0*xp.power(1+z,3)+(1-cosmo.Om0))

class CosmoMadauDickinsonRedshift(_CosmoRedshift):
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

    variable_names = ["gamma", "kappa", "z_peak", "H0", "Om0"]

    def psi_of_z(self, redshift, **parameters):
        gamma = parameters["gamma"]
        kappa = parameters["kappa"]
        z_peak = parameters["z_peak"]
        psi_of_z = (1 + redshift) ** gamma / (
            1 + ((1 + redshift) / (1 + z_peak)) ** kappa
        )
        psi_of_z *= 1 + (1 + z_peak) ** (-kappa)
        return psi_of_z

    def astropy_cosmology(self, **parameters):
        Om0 = parameters['Om0']
        H0 = parameters['H0']
        return FlatLambdaCDM(Om0=Om0,H0=H0)    

    def dvc_dz(self, redshift, **parameters):

        astropy_cosmology = self.astropy_cosmology(**parameters)
        dvc_dz =  xp.asarray(4*xp.pi*astropy_cosmology.differential_comoving_volume(to_numpy(redshift)).value)

        return dvc_dz

    def detector_frame_to_source_frame(self, data, H0, Om0, astropy_conv=False):

        cosmo = self.astropy_cosmology(H0=H0,Om0=Om0)

        samples = dict()
        if astropy_conv == True:

            samples['redshift'] = xp.asarray([z_at_value(cosmo.luminosity_distance, d*u.Mpc,zmax=self.z_max) for d in to_numpy(data['luminosity_distance'])])
            samples['mass_1'] = data['mass_1']/(1+samples['redshift'])
            if 'mass_2' in samples:
                samples['mass_2'] = data['mass_2']/(1+samples['redshift'])
                samples['mass_ratio'] = samples['mass_2']/samples['mass_1']
            else:
                samples['mass_ratio'] = data['mass_ratio']
            try:
                samples['a_1'] = data['a_1']
                samples['a_2'] = data['a_2']
            except:
                None
            try:
                samples['cos_tilt_1'] = data['cos_tilt_1']
                samples['cos_tilt_2'] = data['cos_tilt_2']
            except:
                None
        else:
            zs = to_numpy(self.zs)
            dl = cosmo.luminosity_distance(to_numpy(zs)).value
            interp_dl_to_z = splrep(dl,zs,s=0)

            samples['redshift'] = xp.nan_to_num(xp.asarray(splev(to_numpy(data['luminosity_distance']),interp_dl_to_z,ext=0)))
            samples['mass_1'] = data['mass_1']/(1+samples['redshift'])
            if 'mass_2' in samples:
                samples['mass_2'] = data['mass_2']/(1+samples['redshift'])
                samples['mass_ratio'] = samples['mass_2']/samples['mass_1']
            else:
                samples['mass_ratio'] = data['mass_ratio']
            try:
                samples['a_1'] = data['a_1']
                samples['a_2'] = data['a_2']
            except:
                None
            try:
                samples['cos_tilt_1'] = data['cos_tilt_1']
                samples['cos_tilt_2'] = data['cos_tilt_2']
            except:
                None

        return samples

    def detector_to_source_jacobian(self, z, H0, Om0, dl):

        """
        Calculates the detector frame to source frame Jacobian d_det/d_sour

        Parameters
        ----------
        z: Redshift
        H0, Om0: cosmological parameters
        dl: luminosity distance
        """
        cosmo = self.astropy_cosmology(H0=H0,Om0=Om0)

        speed_of_light = constants.c.to('km/s').value
        # Calculate the Jacobian of the luminosity distance with regard to redshift

        dL_by_dz = dl/(1+z) + speed_of_light*(1+z)/(cosmo.H0.value*self.Efunc(cosmo, z))

        jacobian = (1+z)*dL_by_dz

        return jacobian

    def Efunc(self, cosmo, z):

        return xp.sqrt(cosmo.Om0*xp.power(1+z,3)+(1-cosmo.Om0))