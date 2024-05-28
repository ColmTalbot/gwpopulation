import numpy as xp
from wcosmo import FlatwCDM, available, z_at_value

from .jax import NonCachingModel


class CosmoMixin:
    def __init__(self, cosmo_model="Planck15"):

        self.cosmo_model = cosmo_model
        if self.cosmo_model == "FlatwCDM":
            self.cosmology_names = ["H0", "Om0", "w0"]
        elif self.cosmo_model == "FlatLambdaCDM":
            self.cosmology_names = ["H0", "Om0"]
        else:
            self.cosmology_names = []
        self._cosmo = available[cosmo_model]

    def cosmology_variables(self, parameters):
        return {key: parameters[key] for key in self.cosmology_names}

    def cosmology(self, parameters):
        if isinstance(self._cosmo, FlatwCDM):
            return self._cosmo
        else:
            return self._cosmo(**self.cosmology_variables(parameters))

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
    Modified version of bilby.hyper.model.Model that disables caching for jax.
    """

    def __init__(self, model_functions=None, cosmo_model="Planck15"):
        NonCachingModel.__init__(self, model_functions=model_functions)
        CosmoMixin.__init__(self, cosmo_model=cosmo_model)

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

        data, jacobian = self.detector_frame_to_source_frame(data)
        probability = super().prob(data, **kwargs)
        probability /= jacobian

        return probability
