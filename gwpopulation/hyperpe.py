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

These sums are computed in log space, from
:math:`\ln \pi(\theta_k | \Lambda) - \ln \pi(\theta_k | \varnothing)`, so they stay
finite where the individual terms underflow, e.g., in single precision.
The population model's log density is used if it provides one (see :class:`Model`),
and the sampling prior :math:`\pi(\theta_k | \varnothing)` can be given in each
posterior as :code:`prior` or, preferably, its log as :code:`ln_prior`.

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

import sys
import types

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.core.utils import infer_args_from_function_except_n_args, logger
from bilby.hyper.model import Model as _BilbyModel

from .utils import get_name, logsumexp, segment_logsumexp, to_number, to_numpy

xp = np

__all__ = ["HyperparameterLikelihood", "Model", "RateLikelihood", "xp"]


class Model(_BilbyModel):
    r"""
    Population model that combines a set of factorizable models and can be
    evaluated in log space.

    .. math::

        \ln p(\theta | \Lambda) = \sum_{i} \ln p_{i}(\theta | \Lambda)

    This is a drop-in replacement for :code:`bilby.hyper.model.Model`;
    :func:`prob` is unchanged (the product of the component probabilities) and
    :func:`ln_prob` sums the component log probabilities.
    Evaluating in log space avoids underflow far from the bulk of the
    population, e.g., when using single precision.

    For each component the log probability is resolved, in order, as:

    - the :code:`ln_prob` attribute of the component, if present. This
      applies to classes defining an :code:`ln_prob` method and to functions
      with an attached :code:`ln_prob` attribute, e.g.,
      :code:`my_model.ln_prob = ln_my_model`, which is how a log density can be
      provided for an arbitrary function.
    - a function named :code:`ln_<name>` in the module defining the
      component function :code:`<name>`, e.g.,
      :func:`gwpopulation.models.mass.ln_two_component_primary_mass_ratio` for
      :func:`gwpopulation.models.mass.two_component_primary_mass_ratio`.
      This is only used if the hyperparameters of the two functions match.
    - :code:`xp.log` of the linear component, so all existing models work.

    The hyperparameters passed to the log probability are found from the
    linear component using the same rules as :code:`bilby.hyper.model.Model`,
    so the log version must take the same arguments.

    Parameters
    ==========
    model_functions: list
        List of callables to compute the probability.
        If this includes classes, the :code:`__call__`: method
        should return the probability.
        The requires variables are chosen at run time based on either
        inspection or querying a :code:`variable_names` attribute.
    cache: bool
        Whether to cache the value returned by the model functions,
        default=:code:`True`. The caching only looks at the parameters
        not the data, so should be used with caution. The caching also
        breaks :code:`jax` JIT compilation. The probabilities and log
        probabilities are cached separately.
    """

    def __init__(self, model_functions=None, cache=True):
        super().__init__(model_functions=model_functions, cache=cache)
        self._cached_ln_parameters = {model: None for model in self.models}
        self._cached_ln_probability = {model: None for model in self.models}
        self._ln_functions = dict()

    def ln_prob(self, data, **kwargs):
        """
        Compute the total population log probability for the provided data
        given the keyword arguments.

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

        Returns
        =======
        ln_probability: array-like
            The log probability, :code:`-inf` where the probability is zero.
        """
        ln_probability = 0.0
        for function in self.models:
            function_parameters = self._get_function_parameters(function, **kwargs)
            if (
                self.cache
                and self._cached_ln_parameters.get(function) == function_parameters
            ):
                new_ln_probability = self._cached_ln_probability[function]
            else:
                ln_function = self.ln_function(function)
                new_ln_probability = ln_function(data, **function_parameters)
                if self.cache:
                    self._cached_ln_parameters[function] = function_parameters
                    self._cached_ln_probability[function] = new_ln_probability
            ln_probability = ln_probability + new_ln_probability
        return ln_probability

    def ln_function(self, function):
        """
        Find the callable returning the log probability for a component.
        See the class docstring for the resolution order.

        Parameters
        ==========
        function: callable
            One of :code:`self.models`.

        Returns
        =======
        ln_function: callable
            A callable with the same signature as :code:`function` returning
            the log probability.
        """
        if function not in self._ln_functions:
            self._ln_functions[function] = self._resolve_ln_function(function)
        return self._ln_functions[function]

    @staticmethod
    def _resolve_ln_function(function):
        ln_function = getattr(function, "ln_prob", None)
        if callable(ln_function):
            return ln_function

        name = getattr(function, "__name__", None)
        module = sys.modules.get(getattr(function, "__module__", None) or "")
        ln_function = getattr(module, f"ln_{name}", None) if name else None
        if callable(ln_function):
            if hasattr(function, "variable_names") or (
                infer_args_from_function_except_n_args(function, n=0)
                == infer_args_from_function_except_n_args(ln_function, n=0)
            ):
                return ln_function
            logger.warning(
                f"Not using ln_{name} for {name} as the arguments differ, "
                "falling back to the log of the probability."
            )

        def ln_function(data, **parameters):
            return xp.log(function(data, **parameters))

        return ln_function


def model_ln_prob(model, data, parameters):
    """
    The log probability of a population model: :code:`model.ln_prob` if it
    has one (e.g., :class:`Model`), else the log of :code:`model.prob`.
    """
    if callable(getattr(model, "ln_prob", None)):
        return model.ln_prob(data, **parameters)
    return xp.log(model.prob(data, **parameters))


def as_ln_model(model):
    """
    Convert a :code:`bilby.hyper.model.Model` to a :class:`Model` with the
    same components, so that their log probabilities are used, e.g.,
    :code:`ln_<name>` functions. Anything else, including subclasses of the
    :code:`bilby` model, which may change :code:`prob`, is returned unchanged.
    """
    if type(model) is not _BilbyModel:
        return model
    converted = Model(model.models, cache=model.cache)
    converted.parameters = model.parameters
    return converted


def ln_sampling_prior(data):
    """
    Pop the sampling prior from :code:`data` and return its log.

    :code:`ln_prior` is used if present, otherwise the log of :code:`prior`,
    otherwise zero (a sampling prior of one). Passing :code:`ln_prior`
    avoids prior values that underflow in single precision. The linear
    prior is returned too, :code:`None` when only :code:`ln_prior` is given.
    """
    prior = data.pop("prior", None)
    if "ln_prior" in data:
        return data.pop("ln_prior"), prior
    elif prior is not None:
        return xp.log(prior), prior
    return None, None


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
        maximum_uncertainty=xp.inf,
        require_equal_samples=False,
    ):
        """
        Parameters
        ----------
        posteriors: list
            An list of pandas data frames of samples sets of samples.
            Each set may have a different size.
            These can contain a `prior` column containing the original prior
            values, or an `ln_prior` column containing their log, which is
            preferred if present.
        hyper_prior: `bilby.hyper.model.Model`
            The population model, this can alternatively be a function.
            The Monte Carlo integrals use its :code:`ln_prob` method if it has
            one (e.g., :class:`Model`), else the log of :code:`prob`.
            A :code:`bilby.hyper.model.Model` is converted to a
            :class:`Model` with the same components (see
            :func:`as_ln_model`), so :code:`self.hyper_prior` is a new object.
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
        maximum_uncertainty: float
            The maximum allowed uncertainty in the natural log likelihood.
            If the uncertainty is larger than this value a log likelihood of
            -inf will be returned. Default = inf
        require_equal_samples: bool
            Whether to require an equal number of samples per posterior.
            Set to :code:`True` for backward compatibility.
        """

        self.samples_per_posterior = max_samples
        self.equal_samples = require_equal_samples
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
        self.hyper_prior = as_ln_model(hyper_prior)
        super().__init__()

        self.ln_sampling_prior, self.sampling_prior = ln_sampling_prior(self.data)
        if self.ln_sampling_prior is None:
            logger.info("No prior values provided, defaulting to 1.")
            self.ln_sampling_prior = 0.0
            self.sampling_prior = 1
        elif self.sampling_prior is None:
            self.sampling_prior = xp.exp(self.ln_sampling_prior)

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

    def ln_likelihood_and_variance(self, parameters):
        """
        Compute the ln likelihood estimator and its variance.
        """
        parameters, added_keys = self.conversion_function(parameters)
        ln_bayes_factors, variances = self._compute_per_event_ln_bayes_factors(
            parameters
        )
        ln_l = xp.sum(ln_bayes_factors)
        variance = xp.sum(variances)
        selection, selection_variance = self._get_selection_factor(
            parameters=parameters
        )
        variance += selection_variance
        ln_l += selection
        return ln_l, to_number(variance, float)

    def log_likelihood_ratio(self, parameters):
        ln_l, variance = self.ln_likelihood_and_variance(parameters=parameters)
        ln_l = xp.nan_to_num(ln_l, nan=-xp.inf)
        ln_l -= xp.nan_to_num(xp.inf * (self.maximum_uncertainty < variance), nan=0)
        return to_number(xp.nan_to_num(ln_l), float)

    def noise_log_likelihood(self):
        return self.total_noise_evidence

    def log_likelihood(self, parameters):
        return self.noise_log_likelihood() + self.log_likelihood_ratio(
            parameters=parameters
        )

    def _compute_per_event_ln_bayes_factors(
        self, parameters, *, return_uncertainty=True
    ):
        ln_weights = (
            model_ln_prob(self.hyper_prior, self.data, parameters)
            - self.ln_sampling_prior
        )
        ln_expectation = self._ln_weight_expectation(ln_weights)
        if return_uncertainty:
            # (<w^2> - <w>^2) / (n <w>^2), with the ratio formed in log space
            ln_square_expectation = self._ln_weight_expectation(2 * ln_weights)
            variance = (
                xp.exp(ln_square_expectation - 2 * ln_expectation) - 1
            ) / self.samples_per_posterior
            return ln_expectation, variance
        else:
            return ln_expectation

    def _ln_weight_expectation(self, ln_weights):
        """
        The log of the mean weight for each event, from the log weights.
        """
        if self.equal_samples:
            ln_total = logsumexp(ln_weights, axis=-1)
        else:
            ln_total = segment_logsumexp(ln_weights, self.samples_per_posterior)
        return ln_total - xp.log(self.samples_per_posterior)

    def _weight_expectation(self, weights):
        if self.equal_samples:
            expectation = xp.mean(weights, axis=-1)
        else:
            cumulative = xp.concat([xp.zeros(1), xp.cumsum(weights)])
            transitions = xp.concat(
                [xp.zeros(1), xp.cumsum(self.samples_per_posterior)]
            ).astype(int)
            expectation = (
                cumulative[transitions[1:]] - cumulative[transitions[:-1]]
            ) / self.samples_per_posterior
        return expectation

    def _get_selection_factor(self, parameters, *, return_uncertainty=True):
        selection, variance = self._selection_function_with_uncertainty(
            parameters=parameters
        )
        total_selection = -self.n_posteriors * xp.log(selection)
        if return_uncertainty:
            total_variance = self.n_posteriors**2 * xp.divide(variance, selection**2)
            return total_selection, total_variance
        else:
            return total_selection

    def _selection_function_with_uncertainty(self, parameters):
        result = self.selection_function(parameters)
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
        parameters, added_keys = self.conversion_function(sample.copy())
        ln_ls, variances = self._compute_per_event_ln_bayes_factors(
            parameters, return_uncertainty=True
        )
        total_variance = sum(variances)
        for ii in range(self.n_posteriors):
            sample[f"ln_bf_{ii}"] = to_number(ln_ls[ii], float)
            sample[f"var_{ii}"] = to_number(variances[ii], float)
        selection, variance = self._selection_function_with_uncertainty(
            parameters=parameters
        )
        variance /= selection**2
        selection_variance = variance * self.n_posteriors**2
        sample["selection"] = selection
        sample["selection_variance"] = variance
        total_variance += selection_variance
        sample["variance"] = to_number(total_variance, float)
        return sample

    def generate_rate_posterior_sample(self, parameters):
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
            efficiency, _ = self.selection_function.detection_efficiency(parameters)
            vt = efficiency * self.selection_function.surveyed_hypervolume(parameters)
        else:
            vt = self.selection_function(parameters)
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
            Maximum number of samples to take from each posterior.

        Returns
        -------
        data: dict
            Dictionary containing arrays of size (n_posteriors, max_samples)
            There is a key for each shared key in posteriors.
        """
        data = {key: [] for key in posteriors[0]}

        if self.equal_samples:
            for posterior in posteriors:
                max_samples = min(len(posterior), max_samples)
            logger.debug(f"Downsampling to {max_samples} samples per posterior.")
            self.samples_per_posterior = max_samples
            for posterior in posteriors:
                temp = posterior.sample(self.samples_per_posterior)
                for key in data:
                    data[key].append(temp[key])
        else:
            self.samples_per_posterior = np.asarray(
                [min(len(posterior), max_samples) for posterior in posteriors]
            )
            transitions = np.concat(
                [np.zeros(1), np.cumsum(self.samples_per_posterior)]
            ).astype(int)
            for posterior, nsamples in zip(posteriors, self.samples_per_posterior):
                temp = posterior.sample(nsamples)
                for key in data:
                    data[key].extend(temp[key])
            self.samples_per_posterior = xp.asarray(self.samples_per_posterior)

        for key in data:
            data[key] = xp.asarray(data[key])

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
        if self.equal_samples:
            weights = xp.zeros((self.n_posteriors, self.samples_per_posterior))
        else:
            weights = xp.zeros(int(xp.sum(self.samples_per_posterior)))

        event_weights = xp.zeros(self.n_posteriors)
        for sample in tqdm(samples):
            parameters, added_keys = self.conversion_function(sample.copy())
            new_weights = (
                self.hyper_prior.prob(self.data, **parameters) / self.sampling_prior
            )
            expectation = self._weight_expectation(new_weights)
            event_weights += expectation
            if self.equal_samples:
                denominator = expectation * self.samples_per_posterior
            else:
                denominator = xp.concat(
                    [
                        xp.ones(nsamples) * weight * nsamples
                        for nsamples, weight in zip(
                            self.samples_per_posterior, expectation
                        )
                    ]
                )
            new_weights = (new_weights.T / denominator).T
            weights += new_weights

        new_idxs = xp.empty_like(weights, dtype=int)
        for ii in range(self.n_posteriors):
            if self.equal_samples:
                sl = ii
                start = 0
                nsamples = self.samples_per_posterior
            else:
                transitions = np.concat(
                    [xp.zeros(1), xp.cumsum(self.samples_per_posterior)]
                ).astype(int)
                sl = slice(transitions[ii], transitions[ii + 1])
                start = transitions[ii]
                nsamples = int(self.samples_per_posterior[ii])
            wts = weights[sl]
            wts /= wts.sum()
            if "jax" in xp.__name__:
                from jax import random

                rng_key = random.PRNGKey(np.random.randint(10000000))
                new_idxs = new_idxs.at[sl].set(
                    random.choice(
                        rng_key,
                        xp.arange(nsamples) + start,
                        shape=(nsamples,),
                        replace=True,
                        p=wts,
                    )
                )
            else:
                new_idxs[sl] = xp.asarray(
                    np.random.choice(
                        np.arange(nsamples) + start,
                        size=nsamples,
                        replace=True,
                        p=to_numpy(wts),
                    )
                )

        if self.equal_samples:
            new_samples = {
                key: xp.vstack(
                    [
                        self.data[key][ii, new_idxs[ii]]
                        for ii in range(self.n_posteriors)
                    ]
                )
                for key in self.data
            }
        else:
            new_samples = {key: self.data[key][new_idxs] for key in self.data}
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

    def _get_selection_factor(self, parameters, *, return_uncertainty=True):
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
        selection, variance = self._selection_function_with_uncertainty(
            parameters=parameters
        )
        n_expected = selection * parameters["rate"]
        total_selection = -n_expected + self.n_posteriors * xp.log(parameters["rate"])
        if return_uncertainty:
            total_variance = n_expected * variance / selection**2
            return total_selection, total_variance
        else:
            return total_selection

    def generate_rate_posterior_sample(self, parameters):
        """
        Since the rate is a sampled parameter,
        this simply returns the current value of the rate parameter.
        """
        return parameters["rate"]


class NullHyperparameterLikelihood(HyperparameterLikelihood):
    """
    A likelihood that can be used to sample the prior space subject to the
    maximum_uncertainty constraint imposed on the variance.

    For the uncertainty calculation see the Appendix of
    `Golomb and Talbot <https://arxiv.org/abs/2106.15745>`_ and
    `Farr <https://arxiv.org/abs/1904.10879>`_.
    """

    def ln_likelihood_and_variance(self, parameters=None):
        """
        Compute the ln likelihood estimator and its variance.
        """
        _, variance = super().ln_likelihood_and_variance(parameters)
        return 0.0, variance
