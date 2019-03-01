from __future__ import division, print_function

import numpy as np
try:
    import cupy as xp
    CUPY_LOADED = True
except ImportError:
    xp = np
    CUPY_LOADED = False

from bilby.core.utils import logger
from bilby.core.likelihood import Likelihood
from bilby.hyper.model import Model


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

    def __init__(self, posteriors, hyper_prior, sampling_prior,
                 ln_evidences=None, max_samples=1e100,
                 selection_function=lambda args: 1,
                 conversion_function=lambda args: (args, None), cupy=True):
        if cupy and not CUPY_LOADED:
            logger.warning('Cannot import cupy, falling back to numpy.')

        self.samples_per_posterior = max_samples
        self.data = self.resample_posteriors(
            posteriors, max_samples=max_samples)

        if not isinstance(hyper_prior, Model):
            hyper_prior = Model([hyper_prior])
        self.hyper_prior = hyper_prior
        Likelihood.__init__(self, hyper_prior.parameters)

        if not isinstance(sampling_prior, Model):
            sampling_prior = Model([sampling_prior])
        self.sampling_prior = sampling_prior.prob(self.data)

        if ln_evidences is not None:
            self.total_noise_evidence = np.sum(ln_evidences)
        else:
            self.total_noise_evidence = np.nan

        self.conversion_function = conversion_function
        self.selection_function = selection_function

        self.n_posteriors = len(posteriors)
        self.samples_factor =\
            - self.n_posteriors * np.log(self.samples_per_posterior)

    def log_likelihood_ratio(self):
        self.parameters, added_keys = self.conversion_function(self.parameters)
        self.hyper_prior.parameters.update(self.parameters)
        ln_l = xp.sum(xp.log(xp.sum(self.hyper_prior.prob(self.data) /
                                    self.sampling_prior, axis=-1)))
        ln_l += self.samples_factor
        if added_keys is not None:
            for key in added_keys:
                self.parameters.pop(key)
        return float(xp.nan_to_num(ln_l))

    def noise_log_likelihood(self):
        return self.total_noise_evidence

    def log_likelihood(self):
        return self.noise_log_likelihood() + self.log_likelihood_ratio()

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
        logger.debug('Downsampling to {} samples per posterior.'.format(
            max_samples))
        self.samples_per_posterior = max_samples
        for posterior in posteriors:
            temp = posterior.sample(self.samples_per_posterior)
            for key in data:
                data[key].append(temp[key])
        for key in data:
            data[key] = xp.array(data[key])
        return data


class RateLikelihood(HyperparameterLikelihood):
    """ A likelihood for infering hyperparameter posterior distributions and
    rate estimates

    See Eq. (1) of https://arxiv.org/abs/1801.02699, Eq. (4)
    https://arxiv.org/abs/1805.06442 for a definition.

    Parameters
    ----------
    posteriors: list
        An list of pandas data frames of samples sets of samples. Each set
        may have a different size.
    hyper_prior: func
        Function which calculates the new prior probability for the data.
    sampling_prior: func
        Function which calculates the prior probability used to sample.
    max_samples: int
        Maximum number of samples to use from each set.

    """

    def __init__(self, posteriors, hyper_prior, sampling_prior,
                 ln_evidences=None, max_samples=1e100,
                 selection_function=lambda args: 1,
                 conversion_function=lambda args: (args, None), cupy=True):
        super(RateLikelihood, self).__init__(
            posteriors=posteriors, hyper_prior=hyper_prior,
            sampling_prior=sampling_prior, ln_evidences=ln_evidences,
            max_samples=max_samples, conversion_function=conversion_function,
            selection_function=selection_function, cupy=cupy)

    def log_likelihood_ratio(self):
        self.parameters, added_keys = self.conversion_function(self.parameters)
        log_l = HyperparameterLikelihood.log_likelihood_ratio(self)
        log_l += self.n_posteriors * xp.log(self.parameters['rate'])
        log_l -= self.selection_function(self.parameters) *\
            self.parameters['rate']
        if added_keys is not None:
            for key in added_keys:
                self.parameters.pop(key)
        return float(xp.nan_to_num(log_l))
