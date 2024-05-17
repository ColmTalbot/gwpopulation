import numpy as np
from astropy import constants
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, FlatwCDM
from bilby.hyper.model import Model
from scipy.interpolate import splev, splrep

from gwpopulation.utils import to_numpy

xp = np
speed_of_light = constants.c.to("km/s").value


class CosmoMixin:
    def detector_frame_to_source_frame(self, data, **parameters):
        """
        Convert detector frame samples to sourece frame samples given cosmological parameters. Calculate the corresponding d_detector/d_source Jacobian term.

        Parameters
        ==========
        data: dict
            Dictionary containing the samples in detector frame.
        parameters: dict
            The cosmological parameters for relevant cosmology model.
        """

        cosmo = self.redshift_model.cosmology_model(**parameters)
        jac = 1
        if "luminosity_distance" in data.keys():
            zs = self.redshift_model.zs_
            dl = cosmo.luminosity_distance(zs).value
            interp_dl_to_z = splrep(dl, zs, s=0)

            data["redshift"] = xp.nan_to_num(
                xp.asarray(
                    splev(to_numpy(data["luminosity_distance"]), interp_dl_to_z, ext=0)
                )
            )
            jac *= data["luminosity_distance"] / (
                1 + data["redshift"]
            ) + speed_of_light * (1 + data["redshift"]) / xp.array(
                cosmo.H(to_numpy(data["redshift"])).value
            )  # luminosity_distance_to_redshift_jacobian, dL_by_dz
        elif "redshift" not in data:
            raise ValueError(
                f"Either luminosity distance or redshift provided in detector frame to source frame samples conversion"
            )

        for key in list(data.keys()):
            if key.endswith("_detector"):
                data[key[:-9]] = data[key] / (1 + data["redshift"])
                jac *= 1 + data["redshift"]

        return data, jac


class CosmoModel(Model, CosmoMixin):
    """
    Modified version of bilby.hyper.model.Model that disables caching for jax.
    """

    def __init__(self, model_functions=None):
        super(CosmoModel, self).__init__(model_functions=model_functions)
        for model in self.models:
            if isinstance(model, _BaseRedshift):
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

        data, jac = self.detector_frame_to_source_frame(
            data,
            **self._get_function_parameters(self.redshift_model),
        )  # convert samples to source frame and calculate the Jacobian term.
        probability = 1.0  # prob in source frame
        for function in self.models:
            new_probability = function(data, **self._get_function_parameters(function))
            probability *= new_probability
        probability /= jac  # prob in detector frame

        return probability


class _BaseRedshift:
    """
    Base class for models which include a term like dVc/dz / (1 + z)
    """

    base_variable_names = None

    @property
    def variable_names(self):
        if self.cosmo_model == None:
            vars = []
        if self.cosmo_model == FlatwCDM:
            vars = ["H0", "Om0", "w0"]
        elif self.cosmo_model == FlatLambdaCDM:
            vars = ["H0", "Om0"]
        else:
            raise ValueError(f"Model {cosmo_model} not found.")
        vars += self.base_variable_names
        return vars

    def __init__(self, z_max=2.3, cosmo_model=None):

        self.z_max = z_max
        self.zs_ = np.linspace(1e-3, z_max, 1000)
        self.zs = xp.asarray(self.zs_)
        if cosmo_model == None:
            self.cosmo_model = None
            from astropy.cosmology import Planck15

            self.dvc_dz_ = (
                Planck15.differential_comoving_volume(self.zs_).value * 4 * np.pi
            )
            self.dvc_dz = xp.asarray(self.dvc_dz_)
            self.cached_dvc_dz = None
        else:
            self.cosmo_model = cosmo_model

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
        if self.cosmo_model is not None:
            self.update_dvc_dz(**parameters)
        normalisation = self.normalisation(parameters=parameters)
        differential_volume = self.differential_spacetime_volume(
            dataset=dataset, **parameters
        )
        in_bounds = dataset["redshift"] <= self.z_max
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

    def cosmology_model(self, **parameters):
        if self.cosmo_model == FlatwCDM:
            return self.cosmo_model(
                H0=parameters["H0"], Om0=parameters["Om0"], w0=parameters["w0"]
            )
        elif self.cosmo_model == FlatLambdaCDM:
            return self.cosmo_model(H0=parameters["H0"], Om0=parameters["Om0"])
        else:
            raise ValueError(f"Model {cosmo_model} not found.")

    def update_dvc_dz(self, **parameters):
        self.dvc_dz_ = (
            self.cosmology_model(**parameters)
            .differential_comoving_volume(self.zs_)
            .value
            * 4
            * xp.pi
        )
        self.dvc_dz = xp.asarray(self.dvc_dz_)
        self.cached_dvc_dz = None


class CosmoPowerLawRedshift(_BaseRedshift):
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


class CosmoMadauDickinsonRedshift(_BaseRedshift):
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
