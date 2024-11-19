r"""
Gravitational-wave transient surveys provide a biased sample of the astrophysical population.
The likelihood function used for population inference is given be

.. math::

    {\cal L}(\{d_i\} | \Lambda) &= \prod_i {\cal L}(d_i | \Lambda, {\rm det})
    
    &= \prod_i \frac{{\cal L}(d_i | \Lambda)}{P_{\rm det}(\Lambda)}

    &= \frac{1}{P_{\rm det}^{N}(\Lambda)} \prod_i \int d\theta_i p(d_i | \theta_i) \pi(\theta_i | \Lambda).

The quantity :math:`P_{\rm det}(\Lambda)` is the detection probability for a single source (see `<selection.html>`_).

The integrals over the per-event parameters :math:`\theta_i` are typically performed using Monte Carlo integration

.. math::

    \hat{{\cal L}}(d_i | \Lambda) = \frac{1}{K} \sum_{k=1}^K \frac{\pi(\theta_k | \Lambda)}{\pi(\theta_k | \varnothing)}.

The full approximate log-likelihood is then given by

.. math::

    \ln \hat{\cal L}(\{d_i\} | \Lambda) = \sum_i \ln \hat{{\cal L}}(d_i | \Lambda) - N \ln \hat{P}_{\rm det}(\Lambda).

This approximation is implemented in :class:`gwpopulation.hyperpe.HyperparameterLikelihood`.

There is another related expression for the likelihood as the result of an inhomoegeneous Poisson process.
In this case the likelihood is given by

.. math::

    \ln {\cal L}(\{d_i\} | \Lambda) = N \ln R - N_{\rm exp}(\Lambda)
    + \sum_i \ln \hat{{\cal L}}(d_i | \Lambda)

Here :math:`R` is the total merger rate and :math:`T` is the total observation time
and :math:`N_{\rm exp}(\Lambda) = RT\hat{P}_{\rm det}(\Lambda)`.
This is implemented in :class:`gwpopulation.hyperpe.RateLikelihood`.

Each of these Monte Carlo integrals have associated uncertainties which are propagated through the likelihood calculation
and can be calculated using :func:`gwpopulation.hyperpe.HyperparameterLikelihood.ln_likelihood_and_variance`.
"""

import types

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.core.utils import logger
from bilby.hyper.model import Model

from .utils import get_name, to_number, to_numpy

xp = np

__all__ = ["HyperparameterLikelihood", "RateLikelihood", "xp"]


class HyperparameterLikelihood(Likelihood):
    """
    A likelihood for inferring hyperparameter posterior distributions with
    including selection effects.

    See Eq. (34) of `Thrane and Talbot <https://arxiv.org/abs/1809.02293>`_
    for a definition.

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
        cupy=False,
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
            DEPRECATED: if you want to use cupy, you should manually set the
            backend using :code:`gwpopulation.set_backend`.
        maximum_uncertainty: float
            The maximum allowed uncertainty in the natural log likelihood.
            If the uncertainty is larger than this value a log likelihood of
            -inf will be returned. Default = inf
        """
        if cupy:
            logger.warning(
                f"Setting the backend to cupy is no longer supported in the "
                "likelihood. Use gwpopulation.set_backend instead."
            )

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
        """
        The maximum allowed uncertainty in the estimate of the log-likelihood.
        If the uncertainty is larger than this value a log likelihood of -inf
        is returned.
        """
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
        return ln_l, to_number(variance, float)

    def log_likelihood_ratio(self):
        ln_l, variance = self.ln_likelihood_and_variance()
        ln_l = xp.nan_to_num(ln_l, nan=-xp.inf)
        ln_l -= xp.nan_to_num(xp.inf * (self.maximum_uncertainty < variance), nan=0)
        return to_number(xp.nan_to_num(ln_l), float)

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
            total_variance = self.n_posteriors**2 * xp.divide(
                variance, selection**2
            )
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
        r"""
        Given an input sample, add extra statistics

        Adds:

        - :code:`ln_bf_idx`: :math:`\frac{\ln {\cal L}(d_{i} | \Lambda)}
          {\ln {\cal L}(d_{i} | \varnothing)}`
          for each of the events in the data
        - :code:`selection`: :math:`P_{\rm det}`
        - :code:`var_idx`, :code:`selection_variance`: the uncertainty in
          each Monte Carlo integral
        - :code:`total_variance`: the total variance in the likelihood

        .. note::

            The quantity :code:`selection_variance` is the variance in
            :code:`P_{\rm det}` and not the total variance from the contribution
            of the selection function to the likelihood.

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
        total_variance = sum(variances)
        for ii in range(self.n_posteriors):
            sample[f"ln_bf_{ii}"] = to_number(ln_ls[ii], float)
            sample[f"var_{ii}"] = to_number(variances[ii], float)
        selection, variance = self._selection_function_with_uncertainty()
        variance /= selection**2
        selection_variance = variance * self.n_posteriors**2
        sample["selection"] = selection
        sample["selection_variance"] = variance
        total_variance += selection_variance
        sample["variance"] = to_number(total_variance, float)
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

        .. note::

            This function only uses the :code:`numpy` backend. It can be used
            with the other backends as it returns a float, but does not support
            e.g., autodifferentiation with :code:`jax`.

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

        Parameters
        ----------
        samples: pd.DataFrame, dict, list
            The samples to do the weighting over, typically the posterior from
            some run.
        return_weights: bool, optional
            Whether to return the per-sample weights, default = :code:`False`

        Returns
        -------
        new_samples: dict
            Dictionary containing the weighted posterior samples for each of
            the events.
        weights: array-like
            Weights to apply to the samples, only if :code:`return_weights == True`.
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
            if "jax" in xp.__name__:
                from jax import random

                rng_key = random.PRNGKey(np.random.randint(10000000))
                new_idxs = new_idxs.at[ii].set(
                    random.choice(
                        rng_key,
                        xp.arange(self.samples_per_posterior),
                        shape=(self.samples_per_posterior,),
                        replace=True,
                        p=weights[ii],
                    )
                )
            else:
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
            model=[get_name(model) for model in self.hyper_prior.models],
            data={key: to_numpy(self.data[key]) for key in self.data},
            n_events=self.n_posteriors,
            sampling_prior=to_numpy(self.sampling_prior),
            samples_per_posterior=self.samples_per_posterior,
        )


class RateLikelihood(HyperparameterLikelihood):
    """
    A likelihood for inferring hyperparameter posterior distributions
    and estimating rates with including selection effects.

    See Eq. (34) of `Thrane and Talbot <https://arxiv.org/abs/1809.02293>`_
    for a definition.

    """

    __doc__ += HyperparameterLikelihood.__init__.__doc__

    def _get_selection_factor(self, return_uncertainty=True):
        r"""
        The selection factor for the rate likelihood is

        .. math::

            \ln P_{\rm det} = N \ln R - N_{\rm exp}(\Lambda)

        The uncertainty is given by

        .. math::

            \sigma^2 = \frac{N_{\rm exp}(\Lambda) \sigma^2_{\rm det}}{P_{\rm det}^2}

        Parameters
        ----------
        return_uncertainty: bool
            Whether to return the uncertainty in the selection factor.
        """
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
