import numpy as np
from astropy import constants
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, z_at_value
from bilby.hyper.model import Model
from scipy.interpolate import splev, splrep

from gwpopulation.utils import to_numpy

xp = np


class CosmoModel(Model):
    """
    Modified version of bilby.hyper.model.Model that disables caching for jax.
    """

    def __init__(self, model_functions=None):
        super(CosmoModel, self).__init__(model_functions=model_functions)
        for model in self.models:
            if isinstance(model, _CosmoRedshift):
                self.redshift_model = model

    def prob(self, data, **kwargs):
        """
        Compute the total population probability for the provided data given
        the keyword arguments.

        Parameters
        ==========
        data: dict
            Dictionary containing the points at which to evaluate the
            population model.
        kwargs: dict
            The population parameters. These cannot include any of
            :code:`["dataset", "data", "self", "cls"]` unless the
            :code:`variable_names` attribute is available for the relevant
            model.
        """

        samples_in_source = self.redshift_model.detector_frame_to_source_frame(
            data, **self._get_function_parameters(self.redshift_model)
        )
        jac = self.redshift_model.dL_by_dz(
            samples_in_source["redshift"],
            data["luminosity_distance"],
            **self._get_function_parameters(self.redshift_model),
        )  # dL to z
        jac *= 1 + samples_in_source["redshift"]  # (m1_detector, q) to (m1_source, q)
        probability = 1.0  # prob in source frame
        for function in self.models:
            new_probability = function(
                samples_in_source, **self._get_function_parameters(function)
            )
            probability *= new_probability
        probability /= jac  # prob in detector frame

        return probability


class _CosmoRedshift(object):
    """
    Base class for models which include a term like dVc/dz / (1 + z) with flexible cosmology model
    """

    base_variable_names = None

    @property
    def variable_names(self):
        if self.cosmo_model == FlatwCDM:
            vars = ["H0", "Om0", "w0"]
        elif self.cosmo_model == FlatLambdaCDM:
            vars = ["H0", "Om0"]
        else:
            raise ValueError(f"Model {cosmo_model} not found.")
        vars += self.base_variable_names
        return vars

    def __init__(self, cosmo_model, z_max=2.3, astropy_conv=False):

        self.cosmo_model = cosmo_model
        self.z_max = z_max
        self.zs_ = np.linspace(1e-6, z_max, 2500)
        self.zs = xp.asarray(self.zs_)
        self.astropy_conv = astropy_conv

    def __call__(self, dataset, **kwargs):
        return self.probability(dataset=dataset, **kwargs)

    def probability(self, dataset, **parameters):
        # normalization factor
        psi_of_z = self.psi_of_z(redshift=self.zs, **parameters)
        dvc_dz = self.dvc_dz(redshift=self.zs, **parameters)
        norm = xp.trapz(psi_of_z * dvc_dz / (1 + self.zs), self.zs)

        differential_volume = self.psi_of_z(
            redshift=dataset["redshift"], **parameters
        ) / (1 + dataset["redshift"])
        differential_volume *= xp.reshape(
            xp.interp(xp.ravel(dataset["redshift"]), self.zs, dvc_dz),
            dataset["redshift"].shape,
        )

        in_bounds = dataset["redshift"] <= self.z_max
        return differential_volume / norm * in_bounds

    def psi_of_z(self, redshift, **parameters):
        raise NotImplementedError

    def astropy_cosmology(self, **parameters):
        if self.cosmo_model == FlatwCDM:
            return self.cosmo_model(
                H0=parameters["H0"], Om0=parameters["Om0"], w0=parameters["w0"]
            )
        elif self.cosmo_model == FlatLambdaCDM:
            return self.cosmo_model(H0=parameters["H0"], Om0=parameters["Om0"])
        else:
            raise ValueError(f"Model {cosmo_model} not found.")

    def dvc_dz(self, redshift, **parameters):

        astropy_cosmology = self.astropy_cosmology(**parameters)
        dvc_dz = xp.asarray(
            4
            * xp.pi
            * astropy_cosmology.differential_comoving_volume(to_numpy(redshift)).value
        )

        return dvc_dz

    def detector_frame_to_source_frame(self, data, **parameters):

        cosmo = self.astropy_cosmology(**parameters)

        samples = dict()
        if self.astropy_conv == True:

            samples["redshift"] = xp.asarray(
                [
                    z_at_value(cosmo.luminosity_distance, d * u.Mpc, zmax=self.z_max)
                    for d in to_numpy(data["luminosity_distance"])
                ]
            )
            samples["mass_1"] = data["mass_1"] / (1 + samples["redshift"])
            if "mass_2" in samples:
                samples["mass_2"] = data["mass_2"] / (1 + samples["redshift"])
                samples["mass_ratio"] = samples["mass_2"] / samples["mass_1"]
            else:
                samples["mass_ratio"] = data["mass_ratio"]
            try:
                samples["a_1"] = data["a_1"]
                samples["a_2"] = data["a_2"]
            except:
                None
            try:
                samples["cos_tilt_1"] = data["cos_tilt_1"]
                samples["cos_tilt_2"] = data["cos_tilt_2"]
            except:
                None
        else:
            zs = to_numpy(self.zs)
            dl = cosmo.luminosity_distance(to_numpy(zs)).value
            interp_dl_to_z = splrep(dl, zs, s=0)

            samples["redshift"] = xp.nan_to_num(
                xp.asarray(
                    splev(to_numpy(data["luminosity_distance"]), interp_dl_to_z, ext=0)
                )
            )
            samples["mass_1"] = data["mass_1"] / (1 + samples["redshift"])
            if "mass_2" in samples:
                samples["mass_2"] = data["mass_2"] / (1 + samples["redshift"])
                samples["mass_ratio"] = samples["mass_2"] / samples["mass_1"]
            else:
                samples["mass_ratio"] = data["mass_ratio"]
            try:
                samples["a_1"] = data["a_1"]
                samples["a_2"] = data["a_2"]
            except:
                None
            try:
                samples["cos_tilt_1"] = data["cos_tilt_1"]
                samples["cos_tilt_2"] = data["cos_tilt_2"]
            except:
                None

        return samples

    def dL_by_dz(self, z, dl, **parameters):

        """
        Calculates the detector frame to source frame Jacobian d_det/d_sour for dL and z
        Parameters
        ----------
        z: _np. arrays
            Redshift
        cosmo:  class from the cosmology module
            Cosmology class from the cosmology module
        """
        cosmo = self.astropy_cosmology(**parameters)

        speed_of_light = constants.c.to("km/s").value
        # Calculate the Jacobian of the luminosity distance w.r.t redshift

        dL_by_dz = dl / (1 + z) + speed_of_light * (1 + z) / xp.array(
            cosmo.H(to_numpy(z)).value
        )

        return dL_by_dz


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
    base_variable_names = ["lamb"]

    def psi_of_z(self, redshift, **parameters):
        return (1 + redshift) ** parameters["lamb"]


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
