"""
Likelihoods for population inference
"""

import types

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.core.utils import logger
from bilby.hyper.model import Model

from .cupy_utils import CUPY_LOADED, to_numpy, xp


class HyperparameterLikelihood(Likelihood):
    """
    A likelihood for inferring hyperparameter posterior distributions with
    including selection effects.

    See Eq. (34) of https://arxiv.org/abs/1809.02293 for a definition.

    For the uncertainty calculation see the Appendix of
    `Golomb and Talbot <https://arxiv.org/abs/2106.15745>`_ and
    `Farr <https://arxiv.org/abs/1904.10879>`_.
    """

    def __init__(
        self,
        posteriors,
        hyper_prior,
        ln_evidences=None,
        max_samples=1e100,
        selection_function=lambda args: 1,
        conversion_function=lambda args: (args, None),
        cupy=True,
        maximum_uncertainty=xp.inf,
    ):
        """
        Parameters
        ----------
        posteriors: list
            An list of pandas data frames of samples sets of samples.
            Each set may have a different size.
            These can contain a `prior` column containing the original prior
            values.
        hyper_prior: `bilby.hyper.model.Model`
            The population model, this can alternatively be a function.
        ln_evidences: list, optional
            Log evidences for single runs to ensure proper normalisation
            of the hyperparameter likelihood. If not provided, the original
            evidences will be set to 0. This produces a Bayes factor between
            the sampling power_prior and the hyperparameterised model.
        selection_function: func
            Function which evaluates your population selection function.
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
        """
        if cupy and not CUPY_LOADED:
            logger.warning("Cannot import cupy, falling back to numpy.")

        self.samples_per_posterior = max_samples
        self.data = self.resample_posteriors(posteriors, max_samples=max_samples)

        if isinstance(hyper_prior, types.FunctionType):
            hyper_prior = Model([hyper_prior])
        elif not (
            hasattr(hyper_prior, "parameters")
            and callable(getattr(hyper_prior, "prob"))
        ):
            raise AttributeError(
                "hyper_prior must either be a function, "
                "or a class with attribute 'parameters' and method 'prob'"
            )
        self.hyper_prior = hyper_prior
        super(HyperparameterLikelihood, self).__init__(hyper_prior.parameters)

        if "prior" in self.data:
            self.sampling_prior = self.data.pop("prior")
        else:
            logger.info("No prior values provided, defaulting to 1.")
            self.sampling_prior = 1

        if ln_evidences is not None:
            self.total_noise_evidence = np.sum(ln_evidences)
        else:
            self.total_noise_evidence = np.nan

        self.conversion_function = conversion_function
        self.selection_function = selection_function

        self.n_posteriors = len(posteriors)
        self.maximum_uncertainty = maximum_uncertainty
        self._inf = np.nan_to_num(np.inf)

    __doc__ += __init__.__doc__

    @property
    def maximum_uncertainty(self):
        return self._maximum_uncertainty

    @maximum_uncertainty.setter
    def maximum_uncertainty(self, value):
        self._maximum_uncertainty = value
        if value in [xp.inf, np.inf]:
            self._max_variance = value
        else:
            self._max_variance = value**2

    def ln_likelihood_and_variance(self):
        """
        Compute the ln likelihood estimator and its variance.
        """
        self.parameters, added_keys = self.conversion_function(self.parameters)
        self.hyper_prior.parameters.update(self.parameters)
        ln_bayes_factors, variances = self._compute_per_event_ln_bayes_factors()
        ln_l = xp.sum(ln_bayes_factors)
        variance = xp.sum(variances)
        selection, selection_variance = self._get_selection_factor()
        variance += selection_variance
        ln_l += selection
        self._pop_added(added_keys)
        return ln_l, float(variance)

    def log_likelihood_ratio(self):
        ln_l, variance = self.ln_likelihood_and_variance()
        if variance > self._max_variance or xp.isnan(ln_l):
            return -self._inf
        else:
            return float(xp.nan_to_num(ln_l))

    def noise_log_likelihood(self):
        return self.total_noise_evidence

    def log_likelihood(self):
        return self.noise_log_likelihood() + self.log_likelihood_ratio()

    def _pop_added(self, added_keys):
        if added_keys is not None:
            for key in added_keys:
                self.parameters.pop(key)

    def _compute_per_event_ln_bayes_factors(self, return_uncertainty=True):
        weights = self.hyper_prior.prob(self.data) / self.sampling_prior
        expectation = xp.mean(weights, axis=-1)
        if return_uncertainty:
            square_expectation = xp.mean(weights**2, axis=-1)
            variance = (square_expectation - expectation**2) / (
                self.samples_per_posterior * expectation**2
            )
            return xp.log(expectation), variance
        else:
            return xp.log(expectation)

    def _get_selection_factor(self, return_uncertainty=True):
        selection, variance = self._selection_function_with_uncertainty()
        total_selection = -self.n_posteriors * xp.log(selection)
        if return_uncertainty:
            total_variance = self.n_posteriors**2 * variance / selection**2
            return total_selection, total_variance
        else:
            return total_selection

    def _selection_function_with_uncertainty(self):
        result = self.selection_function(self.parameters)
        if isinstance(result, tuple):
            selection, variance = result
        else:
            selection = result
            variance = 0.0
        return selection, variance

    def generate_extra_statistics(self, sample):
        """
        Given an input sample, add extra statistics

        Adds the ln BF for each of the events in the data and the selection
        function

        Parameters
        ----------
        sample: dict
            Input sample to compute the extra things for.

        Returns
        -------
        sample: dict
            The input dict, modified in place.
        """
        self.parameters.update(sample.copy())
        self.parameters, added_keys = self.conversion_function(self.parameters)
        self.hyper_prior.parameters.update(self.parameters)
        ln_ls, variances = self._compute_per_event_ln_bayes_factors(
            return_uncertainty=True
        )
        total_variance = float(sum(variances))
        for ii in range(self.n_posteriors):
            sample[f"ln_bf_{ii}"] = float(ln_ls[ii])
            sample[f"var_{ii}"] = float(variances[ii])
        selection, variance = self._selection_function_with_uncertainty()
        sample["selection"] = selection
        sample["selection_variance"] = variance
        total_variance += variance
        sample["variance"] = float(total_variance)
        if added_keys is not None:
            for key in added_keys:
                self.parameters.pop(key)
        return sample

    def generate_rate_posterior_sample(self):
        r"""
        Generate a sample from the posterior distribution for rate assuming a
        :math:`1 / R` prior.

        The likelihood evaluated is analytically marginalized over rate.
        However the rate dependent likelihood can be trivially written.

        .. math::
            p(R) = \Gamma(n=N, \text{scale}=\mathcal{V})

        Here :math:`\Gamma` is the Gamma distribution, :math:`N` is the number
        of events being analyzed and :math:`\mathcal{V}` is the total observed 4-volume.

        Returns
        -------
        rate: float
            A sample from the posterior distribution for rate.
        """
        from scipy.stats import gamma

        if hasattr(self.selection_function, "detection_efficiency") and hasattr(
            self.selection_function, "surveyed_hypervolume"
        ):
            efficiency, _ = self.selection_function.detection_efficiency(
                self.parameters
            )
            vt = efficiency * self.selection_function.surveyed_hypervolume(
                self.parameters
            )
        else:
            vt = self.selection_function(self.parameters)
        rate = gamma(a=self.n_posteriors).rvs() / vt
        return rate

    def resample_posteriors(self, posteriors, max_samples=1e300):
        """
        Convert list of pandas DataFrame object to dict of arrays.

        Parameters
        ----------
        posteriors: list
            List of pandas DataFrame objects.
        max_samples: int, opt
            Maximum number of samples to take from each posterior,
            default is length of shortest posterior chain.

        Returns
        -------
        data: dict
            Dictionary containing arrays of size (n_posteriors, max_samples)
            There is a key for each shared key in posteriors.
        """
        for posterior in posteriors:
            max_samples = min(len(posterior), max_samples)
        data = {key: [] for key in posteriors[0]}
        logger.debug(f"Downsampling to {max_samples} samples per posterior.")
        self.samples_per_posterior = max_samples
        for posterior in posteriors:
            temp = posterior.sample(self.samples_per_posterior)
            for key in data:
                data[key].append(temp[key])
        for key in data:
            data[key] = xp.array(data[key])
        return data

    def posterior_predictive_resample(self, samples, return_weights=False):
        """
        Resample the original single event posteriors to use the PPD from each
        of the other events as the prior.

        There may be something weird going on with rate.

        Parameters
        ----------
        samples: pd.DataFrame, dict, list
            The samples to do the weighting over, typically the posterior from
            some run.
        return_weights: bool, optional
            Whether to return the per-sample weights, default = False

        Returns
        -------
        new_samples: dict
            Dictionary containing the weighted posterior samples for each of
            the events.
        weights: array-like
            Weights to apply to the samples, only if return_weights == True.
        """
        import pandas as pd
        from tqdm.auto import tqdm

        if isinstance(samples, pd.DataFrame):
            samples = [dict(samples.iloc[ii]) for ii in range(len(samples))]
        elif isinstance(samples, dict):
            samples = [samples]
        weights = xp.zeros((self.n_posteriors, self.samples_per_posterior))
        event_weights = xp.zeros(self.n_posteriors)
        for sample in tqdm(samples):
            self.parameters.update(sample.copy())
            self.parameters, added_keys = self.conversion_function(self.parameters)
            new_weights = self.hyper_prior.prob(self.data) / self.sampling_prior
            event_weights += xp.mean(new_weights, axis=-1)
            new_weights = (new_weights.T / xp.sum(new_weights, axis=-1)).T
            weights += new_weights
            if added_keys is not None:
                for key in added_keys:
                    self.parameters.pop(key)
        weights = (weights.T / xp.sum(weights, axis=-1)).T
        new_idxs = xp.empty_like(weights, dtype=int)
        for ii in range(self.n_posteriors):
            new_idxs[ii] = xp.asarray(
                np.random.choice(
                    range(self.samples_per_posterior),
                    size=self.samples_per_posterior,
                    replace=True,
                    p=to_numpy(weights[ii]),
                )
            )
        new_samples = {
            key: xp.vstack(
                [self.data[key][ii, new_idxs[ii]] for ii in range(self.n_posteriors)]
            )
            for key in self.data
        }
        event_weights = list(event_weights)
        weight_string = " ".join([f"{float(weight):.1f}" for weight in event_weights])
        logger.info(f"Resampling done, sum of weights for events are {weight_string}")
        if return_weights:
            return new_samples, weights
        else:
            return new_samples

    @property
    def meta_data(self):
        return dict(
            model=[model.__name__ for model in self.hyper_prior.models],
            data={key: to_numpy(self.data[key]) for key in self.data},
            n_events=self.n_posteriors,
            sampling_prior=to_numpy(self.sampling_prior),
            samples_per_posterior=self.samples_per_posterior,
        )


class RateLikelihood(HyperparameterLikelihood):
    """
    A likelihood for inferring hyperparameter posterior distributions
    and estimating rates with including selection effects.

    See Eq. (34) of https://arxiv.org/abs/1809.02293 for a definition.

    """

    __doc__ += HyperparameterLikelihood.__init__.__doc__

    def _get_selection_factor(self, return_uncertainty=True):
        selection, variance = self._selection_function_with_uncertainty()
        n_expected = selection * self.parameters["rate"]
        total_selection = -n_expected + self.n_posteriors * xp.log(
            self.parameters["rate"]
        )
        if return_uncertainty:
            total_variance = n_expected * variance / selection**2
            return total_selection, total_variance
        else:
            return total_selection

    def generate_rate_posterior_sample(self):
        """
        Since the rate is a sampled parameter,
        this simply returns the current value of the rate parameter.
        """
        return self.parameters["rate"]
