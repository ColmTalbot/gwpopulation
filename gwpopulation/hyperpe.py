import numpy as np
import pandas as pd
from tqdm import tqdm

from bilby.core.utils import logger
from bilby.core.likelihood import Likelihood
from bilby.hyper.model import Model

from .cupy_utils import CUPY_LOADED, to_numpy, xp

INF = xp.nan_to_num(xp.inf)


class HyperparameterLikelihood(Likelihood):
    """
    A likelihood for inferring hyperparameter posterior distributions with
    including selection effects.

    See Eq. (34) of https://arxiv.org/abs/1809.02293 for a definition.

    Parameters
    ----------
    posteriors: list
        An list of pandas data frames of samples sets of samples.
        Each set may have a different size.
    hyper_prior: `bilby.hyper.model.Model`
        The population model, this can alternatively be a function.
    sampling_prior: `bilby.hyper.model.Model`
        The sampling prior, this can alternatively be a function.
    log_evidences: list, optional
        Log evidences for single runs to ensure proper normalisation
        of the hyperparameter likelihood. If not provided, the original
        evidences will be set to 0. This produces a Bayes factor between
        the sampling prior and the hyperparameterised model.
    max_samples: int, optional
        Maximum number of samples to use from each set.
    cupy: bool
        If True and a compatible CUDA environment is available,
        cupy will be used for performance.
        Note: this requires setting up your hyper_prior properly.
    """

    def __init__(
        self,
        posteriors,
        hyper_prior,
        sampling_prior=None,
        ln_evidences=None,
        max_samples=1e100,
        selection_function=lambda args: 1,
        conversion_function=lambda args: (args, None),
        cupy=True,
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
        sampling_prior: array-like *DEPRECATED*
            The sampling prior, this can alternatively be a function.
            THIS WILL BE REMOVED IN THE NEXT RELEASE.
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
        """
        if cupy and not CUPY_LOADED:
            logger.warning("Cannot import cupy, falling back to numpy.")

        self.samples_per_posterior = max_samples
        self.data = self.resample_posteriors(posteriors, max_samples=max_samples)

        if not isinstance(hyper_prior, Model):
            hyper_prior = Model([hyper_prior])
        self.hyper_prior = hyper_prior
        Likelihood.__init__(self, hyper_prior.parameters)

        if sampling_prior is not None:
            raise ValueError(
                "Passing a sampling_prior is deprecated and will be removed "
                "in the next release. This should be passed as a 'prior' "
                "column in the posteriors."
            )
        elif "prior" in self.data:
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

    def log_likelihood_ratio(self):
        self.parameters, added_keys = self.conversion_function(self.parameters)
        self.hyper_prior.parameters.update(self.parameters)
        ln_l = xp.sum(self._compute_per_event_ln_bayes_factors())
        ln_l += self._get_selection_factor()
        if added_keys is not None:
            for key in added_keys:
                self.parameters.pop(key)
        if xp.isnan(ln_l):
            return float(-INF)
        else:
            return float(xp.nan_to_num(ln_l))

    def noise_log_likelihood(self):
        return self.total_noise_evidence

    def log_likelihood(self):
        return self.noise_log_likelihood() + self.log_likelihood_ratio()

    def _compute_per_event_ln_bayes_factors(self):
        return -np.log(self.samples_per_posterior) + xp.log(
            xp.sum(self.hyper_prior.prob(self.data) / self.sampling_prior, axis=-1)
        )

    def _get_selection_factor(self):
        return -self.n_posteriors * xp.log(self.selection_function(self.parameters))

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
        ln_ls = self._compute_per_event_ln_bayes_factors()
        for ii in range(self.n_posteriors):
            sample[f"ln_bf_{ii}"] = float(ln_ls[ii])
        sample["selection"] = float(self.selection_function(self.parameters))
        if added_keys is not None:
            for key in added_keys:
                self.parameters.pop(key)
        return sample

    def generate_rate_posterior_sample(self):
        raise NotImplementedError

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


class RateLikelihood(HyperparameterLikelihood):
    """
    A likelihood for inferring hyperparameter posterior distributions
    and estimating rates with including selection effects.

    See Eq. (34) of https://arxiv.org/abs/1809.02293 for a definition.
    """

    def _get_selection_factor(self):
        ln_l = -self.selection_function(self.parameters) * self.parameters["rate"]
        ln_l += self.n_posteriors * xp.log(self.parameters["rate"])
        return ln_l

    def generate_rate_posterior_sample(self):
        pass
