import jax
import jax.numpy as jnp
import numpy as np
from numpyro import deterministic, factor, sample
from numpyro.distributions import Gamma, Normal, Unit

from .jax import generic_bilby_likelihood_function


def bilby_to_numpyro_prior(value):
    from numpyro.distributions import (
        Cauchy,
        Chi2,
        Delta,
        Exponential,
        Gamma,
        HalfNormal,
        Logistic,
        LogNormal,
        LogUniform,
        Normal,
        StudentT,
        TruncatedNormal,
        Uniform,
    )

    mapping = dict(
        Cauchy=lambda x: Cauchy(x.alpha, x.beta),
        ChiSquared=lambda x: Chi2(x.nu),
        Constraint=lambda x: ValueError("Constraint prior not yet supported."),
        DeltaFunction=lambda x: Delta(x.peak),
        Exponential=lambda x: Exponential(x.mu),
        Gamma=lambda x: Gamma(x.k, x.theta),
        HalfNormal=lambda x: HalfNormal(x.sigma),
        HalfGaussian=lambda x: HalfNormal(x.sigma),
        Logisitic=lambda x: Logistic(x.mu, x.sigma),
        LogGaussian=lambda x: LogNormal(x.mu, x.sigma),
        LogNormal=lambda x: LogNormal(x.mu, x.sigma),
        LogUniform=lambda x: LogUniform(x.minimum, x.maximum),
        Normal=lambda x: Normal(x.mu, x.sigma),
        Gaussian=lambda x: Normal(x.mu, x.sigma),
        StudentT=lambda x: StudentT(x.df, x.mu, x.sigma),
        TruncatedGaussian=lambda x: TruncatedNormal(
            x.mu, x.sigma, low=x.minimum, high=x.maximum
        ),
        TruncatedNormal=lambda x: TruncatedNormal(
            x.mu, x.sigma, low=x.minimum, high=x.maximum
        ),
        Uniform=lambda x: Uniform(x.minimum, x.maximum),
    )
    if value.__class__.__name__ in mapping:
        return mapping[value.__class__.__name__](value)
    else:
        print(f"No matching prior class for {value}, defaulting to uniform")
        return Uniform(value.minimum, value.maximum)


def bilby_to_numpyro_priors(priors):
    jpriors = dict()
    for key, value in priors.items():
        jpriors[key] = bilby_to_numpyro_prior(value)
    return jpriors


def construct_numpyro_model(
    likelihood, priors, likelihood_func=generic_bilby_likelihood_function, **kwargs
):
    from bilby.core.prior import DeltaFunction

    priors = priors.copy()
    priors.convert_floats_to_delta_functions()
    delta_fns = {
        k: v for k, v in priors.items() if isinstance(v, (int, float, DeltaFunction))
    }
    priors = bilby_to_numpyro_priors(priors)

    def model():
        parameters = sample_prior(priors, delta_fns)
        ln_l = jnp.nan_to_num(
            likelihood_func(likelihood, parameters, **kwargs), nan=-jnp.inf
        )
        sample("log_likelihood", Unit(ln_l), obs=ln_l)

    return model


def sample_prior(priors, delta_fns):
    parameters = dict()
    base = Normal(0, 1)
    for k, v in priors.items():
        if k in delta_fns:
            parameters[k] = float(deterministic(k, v.mean))
            continue
        try:
            temp = sample(f"{k}_scaled", Normal(0, 1))
            parameters[k] = deterministic(k, v.icdf(base.cdf(temp)))
        except Exception:
            print("Prior sampling failed", k, v)
            raise
    return parameters


def gwpopulation_likelihood_model(
    likelihood,
    hyper_params,
    varmax=np.inf,
    apply_selection=True,
    predictive_resample=True,
    fit_keys=None,
):
    likelihood.parameters.update(hyper_params)
    likelihood.parameters, added_keys = likelihood.conversion_function(
        likelihood.parameters
    )

    likelihood.hyper_prior.parameters.update(likelihood.parameters)
    weights = likelihood.hyper_prior.prob(likelihood.data) / likelihood.sampling_prior
    expectations = jnp.mean(weights, axis=-1)
    square_expectations = jnp.mean(weights**2, axis=-1)
    variances = deterministic(
        "variances",
        (square_expectations - expectations**2)
        / (likelihood.samples_per_posterior * expectations**2),
    )
    ln_bayes_factors = deterministic("ln_bayes_factors", jnp.log(expectations))

    ln_l_events = jnp.sum(ln_bayes_factors)
    variance_events = jnp.sum(variances)

    if apply_selection:
        selector = likelihood.selection_function
        selector.model.parameters.update(likelihood.parameters)
        injection_weights = selector.model.prob(selector.data) / selector.data["prior"]
        efficiency = jnp.sum(injection_weights) / selector.total_injections
        selection_uncertainty = (
            jnp.sum(injection_weights**2) / selector.total_injections**2
            - efficiency**2 / selector.total_injections
        )
        selection = deterministic(
            "selection", -jnp.log(efficiency) * likelihood.n_posteriors
        )
        selection_variance = deterministic(
            "selection_variance",
            selection_uncertainty / efficiency**2 * likelihood.n_posteriors**2,
        )

        total_hypervolume = likelihood.selection_function.surveyed_hypervolume(
            likelihood.parameters
        )
        vt = total_hypervolume * efficiency
        _ = sample("rate", Gamma(likelihood.n_posteriors, 1 / vt))
        if predictive_resample:
            posterior_predictive_resample(
                selector.data,
                injection_weights,
                "injections",
                shape=(likelihood.n_posteriors,),
                fit_keys=fit_keys,
            )
    else:
        selection = 0
        selection_variance = 0

    ln_l = deterministic("ln_l", ln_l_events + selection)
    variance = deterministic("variance", variance_events + selection_variance)

    likelihood._pop_added(added_keys)

    if predictive_resample:
        for event in range(likelihood.n_posteriors):
            posterior_predictive_resample(
                likelihood.data, weights, event, event=event, fit_keys=fit_keys
            )
    if varmax < np.inf:
        ln_l = jnp.where(
            jnp.isnan(ln_l) | (variance > varmax),
            jnp.nan_to_num(-jnp.inf),
            ln_l,
        )
    return ln_l


def posterior_predictive_resample(
    data, weights, label, event=None, shape=(), fit_keys=None
):
    if event is not None:
        weights = weights[:, event]
    weights /= jnp.sum(weights)
    n_points = int(weights.shape[0])
    idx = jax.random.choice(
        jax.random.PRNGKey(0),
        jnp.arange(n_points),
        p=weights,
        shape=shape,
        replace=False,
    )
    if fit_keys is None:
        fit_keys = data.keys()
    for key in fit_keys:
        if event is None:
            deterministic(f"{key}_{label}", data[key][idx])
        else:
            deterministic(f"{key}_{label}", data[key][event, idx])
