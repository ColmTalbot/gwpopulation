from functools import partial

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.hyper.model import Model


def generic_bilby_likelihood_function(likelihood, parameters, use_ratio=True):
    """
    A wrapper to allow a :code:`Bilby` likelihood to be used with :code:`jax`.

    Parameters
    ==========
    likelihood: bilby.core.likelihood.Likelihood
        The likelihood to evaluate.
    parameters: dict
        The parameters to evaluate the likelihood at.
    use_ratio: bool, optional
        Whether to evaluate the likelihood ratio or the full likelihood.
        Default is :code:`True`.
    """
    likelihood.parameters.update(parameters)
    if use_ratio:
        return likelihood.log_likelihood_ratio()
    else:
        return likelihood.log_likelihood()


class NonCachingModel(Model):
    """
    Modified version of bilby.hyper.model.Model that disables caching for jax.
    """

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
        probability = 1.0
        for function in self.models:
            new_probability = function(data, **self._get_function_parameters(function))
            probability *= new_probability
        return probability


class JittedLikelihood(Likelihood):
    """
    A wrapper to just-in-time compile a :code:`Bilby` likelihood for use with :code:`jax`.

    .. note::

        This is currently hardcoded to return the log likelihood ratio, regardless of
        the input.

    Parameters
    ==========
    likelihood: bilby.core.likelihood.Likelihood
        The likelihood to wrap.
    likelihood_func: callable, optional
        The function to use to evaluate the likelihood. Default is
        :code:`generic_bilby_likelihood_function`. This function should take the
        likelihood and parameters as arguments along with additional keyword arguments.
    kwargs: dict, optional
        Additional keyword arguments to pass to the likelihood function.
    """

    def __init__(
        self, likelihood, likelihood_func=generic_bilby_likelihood_function, kwargs=None
    ):
        from jax import jit

        if kwargs is None:
            kwargs = dict()
        self.kwargs = kwargs
        self._likelihood = likelihood
        self.likelihood_func = jit(partial(likelihood_func, likelihood))
        super().__init__(dict())

    def __getattr__(self, name):
        return getattr(self._likelihood, name)

    def log_likelihood_ratio(self):
        return float(
            np.nan_to_num(self.likelihood_func(self.parameters, **self.kwargs))
        )
