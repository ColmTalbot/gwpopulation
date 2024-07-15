"""
Cosmological functionality in :code:`GWPopulation` is based on the :code:`wcosmo` package.
For more details see the `wcosmo documentation <https://wcosmo.readthedocs.io/en/latest/>`_.

We provide a mixin class :func:`gwpopulation.experimental.cosmo_models.CosmoMixin` that
can be used to add cosmological functionality to a population model.
"""

import numpy as xp
from wcosmo import z_at_value
from wcosmo.astropy import WCosmoMixin, available
from wcosmo.utils import disable_units as wcosmo_disable_units

from .jax import NonCachingModel


class CosmoMixin:
    """
    Mixin class that provides cosmological functionality to a subclass.

    Parameters
    ==========
    cosmo_model: str
        The cosmology model to use. Default is :code:`Planck15`.
        Should be of :code:`wcosmo.available.keys()`.
    """

    def __init__(self, cosmo_model="Planck15"):
        wcosmo_disable_units()
        self.cosmo_model = cosmo_model
        if self.cosmo_model == "FlatwCDM":
            self.cosmology_names = ["H0", "Om0", "w0"]
        elif self.cosmo_model == "FlatLambdaCDM":
            self.cosmology_names = ["H0", "Om0"]
        else:
            self.cosmology_names = []
        self._cosmo = available[cosmo_model]

    def cosmology_variables(self, parameters):
        """
        Extract the cosmological parameters from the provided parameters.

        Parameters
        ==========
        parameters: dict
            The parameters for the cosmology model.

        Returns
        =======
        dict
            A dictionary containing :code:`self.cosmology_names` as keys.
        """
        return {key: parameters[key] for key in self.cosmology_names}

    def cosmology(self, parameters):
        """
        Return the cosmology model given the parameters.

        Parameters
        ==========
        parameters: dict
            The parameters for the cosmology model.

        Returns
        =======
        wcosmo.astropy.WCosmoMixin
            The cosmology model.
        """
        if isinstance(self._cosmo, WCosmoMixin):
            return self._cosmo
        else:
            return self._cosmo(**self.cosmology_variables(parameters))

    def detector_frame_to_source_frame(self, data, **parameters):
        r"""
        Convert detector frame samples to sourece frame samples given cosmological
        parameters. Calculate the corresponding
        :math:`\frac{d \theta_{\rm detector}}{d \theta_{\rm source}}` Jacobian term.
        This includes factors of :math:`(1 + z)` for redshifted quantities.

        Parameters
        ==========
        data: dict
            Dictionary containing the samples in detector frame.
        parameters: dict
            The cosmological parameters for relevant cosmology model.

        Returns
        =======
        samples: dict
            Dictionary containing the samples in source frame.
        jacobian: array-like
            The Jacobian term.
        """

        samples = dict()
        if "luminosity_distance" in data.keys():
            cosmo = self.cosmology(self.parameters)
            samples["redshift"] = z_at_value(
                cosmo.luminosity_distance,
                data["luminosity_distance"],
            )
            jacobian = cosmo.dDLdz(samples["redshift"])
        elif "redshift" not in data:
            raise ValueError(
                f"Either luminosity distance or redshift provided in detector frame to source frame samples conversion"
            )
        else:
            jacobian = xp.ones(data["redshift"].shape)

        for key in list(data.keys()):
            if key.endswith("_detector"):
                samples[key[:-9]] = data[key] / (1 + samples["redshift"])
                jacobian *= 1 + samples["redshift"]
            elif key != "luminosity_distance":
                samples[key] = data[key]

        return samples, jacobian


class CosmoModel(NonCachingModel, CosmoMixin):
    """
    Modified version of :code:`bilby.hyper.model.Model` that automatically
    updates the source-frame quantities given the detector-frame quantities and
    cosmology and disables caching due to the source-frame quantities changing
    every iteration.

    Parameters
    ==========
    model_functions: list
        List containing the model functions.
    cosmo_model: str
        The cosmology model to use. Default is :code:`Planck15`.
        Should be of :code:`wcosmo.available.keys()`.
    """

    def __init__(self, model_functions=None, cosmo_model="Planck15"):
        NonCachingModel.__init__(self, model_functions=model_functions)
        CosmoMixin.__init__(self, cosmo_model=cosmo_model)

    def prob(self, data, **kwargs):
        """
        Compute the total population probability for the provided data given
        the keyword arguments.

        This method augments :code:`bilby.hyper.model.Model.prob` by converting
        the detector frame samples to source frame samples and dividing by the
        corresponding Jacobian term.

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

        data, jacobian = self.detector_frame_to_source_frame(data)
        probability = super().prob(data, **kwargs)
        probability /= jacobian

        return probability
