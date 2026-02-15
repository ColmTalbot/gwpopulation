import warnings
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
from bilby.core.likelihood import Likelihood


def generic_bilby_likelihood_function(likelihood: Likelihood, parameters: dict[str, Any], use_ratio: bool = True) -> float:
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
    if use_ratio:
        return likelihood.log_likelihood_ratio(parameters)
    else:
        return likelihood.log_likelihood(parameters)


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
        self, likelihood: Likelihood, likelihood_func: Callable = generic_bilby_likelihood_function, kwargs: dict[str, Any] | None = None
    ) -> None:
        from jax import jit

        if kwargs is None:
            kwargs = dict()
        self.kwargs = kwargs
        self._likelihood = likelihood
        self.likelihood_func = jit(partial(likelihood_func, likelihood))
        super().__init__()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._likelihood, name)

    def log_likelihood_ratio(self, parameters: dict[str, Any]) -> float:
        return float(np.nan_to_num(self.likelihood_func(parameters, **self.kwargs)))
