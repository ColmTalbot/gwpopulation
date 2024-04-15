"""
Likelihood for joint population & cosmology fit (spectral siren)
"""

from ..hyperpe import HyperparameterLikelihood
import numpy as np
from astropy import units as u
from astropy.cosmology import z_at_value
from gwpopulation.experimental.cosmo_models import _CosmoRedshift

xp = np

class CosmoHyperparameterLikelihood(HyperparameterLikelihood):

    def __init__(
        self,
        posterior,
        hyper_prior,
        astropy_conv=False,
        ln_evidences=None,
        max_samples=1e100,
        selection_function=lambda args: 1,
        conversion_function=lambda args: (args, None),
        cupy=True,
        maximum_uncertainty=xp.inf,
        mass_ratio=True,
    ):
        """
        Parameters
        ----------
        posteriors: list
            An list of pandas data frames of samples sets of samples.
            Each set may have a different size.
            These can contain a `prior` column containing the original prior
            values.
            Note in spectral siren, we need posterior samples in detector frame.
            
        hyper_prior: `gwpopulation.experimental.jax.NonCachingModel`
            The population model without cached information.
            Note the redshift model here should be fleixible in cosmology model choices.
            
        astropy_conv: boolean
            Wether luminosity distance - redshift conversions are done with astropy
        ln_evidences: list, optional
            Log evidences for single runs to ensure proper normalisation
            of the hyperparameter likelihood. If not provided, the original
            evidences will be set to 0. This produces a Bayes factor between
            the sampling power_prior and the hyperparameterised model.
        selection_function: func
            Function which evaluates your population selection function.
            Note the selection function in spectral siren analysis should be also calculated from detector frame. 
            
        conversion_function: func
            Function which converts a dictionary of sampled parameter to a
            dictionary of parameters of the population model.
        max_samples: int, optional
            Maximum number of samples to use from each set.
        cupy: bool
            If True and a compatible CUDA environment is available,
            cupy will be used for performance.
            Note: this requires setting up your hyper_prior properly.
        maximum_uncertainty: float
            The maximum allowed uncertainty in the natural log likelihood.
            If the uncertainty is larger than this value a log likelihood of
            -inf will be returned. Default = inf
        mass_ratio: boolean
            If true, mass is modelled by mass ratio
            If False, mass is modellded by mass_2
        """
        super(CosmoHyperparameterLikelihood, self).__init__(posterior, hyper_prior,
        ln_evidences, max_samples, selection_function, conversion_function, cupy, maximum_uncertainty)
        self.astropy_conv = astropy_conv
        self.mass_ratio = mass_ratio
        for model in self.hyper_prior.models:
            if isinstance(model, _CosmoRedshift):
                self.redshift_model = model

    def _compute_per_event_ln_bayes_factors(self, return_uncertainty=True):
        samples_in_source = self.redshift_model.detector_frame_to_source_frame(self.data, self.parameters['H0'], self.parameters['Om0'], self.astropy_conv) #convert samples to source frame given cosmological parameters
        jac = self.redshift_model.detector_to_source_jacobian(samples_in_source['redshift'], self.parameters['H0'], self.parameters['Om0'], self.data['luminosity_distance']) # calculate the jacobian term of the luminosity distance w.r.t redshift
        if self.mass_ratio:
            jac *= (1+samples_in_source['redshift'])
        else:
            jac *= (1+samples_in_source['redshift'])**2

        weights = self.hyper_prior.prob(samples_in_source) / self.sampling_prior / jac
        expectation = xp.mean(weights, axis=-1)
        if return_uncertainty:
            square_expectation = xp.mean(weights**2, axis=-1)
            variance = (square_expectation - expectation**2) / (
                self.samples_per_posterior * expectation**2
            )
            return xp.log(expectation), variance
        else:
            return xp.log(expectation)