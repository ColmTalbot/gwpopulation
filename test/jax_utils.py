from functools import partial

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.hyper.model import Model
from jax import jit


def generic_bilby_likelihood_function(likelihood, parameters, use_ratio=True):
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
    def __init__(
        self, likelihood, likelihood_func=generic_bilby_likelihood_function, kwargs=None
    ):
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
