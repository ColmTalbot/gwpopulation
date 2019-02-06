from __future__ import division, print_function

try:
    import cupy as xp
    CUPY_LOADED = True
except ImportError:
    import numpy as xp
    CUPY_LOADED = False

from bilby.hyper.likelihood import HyperparameterLikelihood


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
            max_samples=max_samples, cupy=cupy)
        self.conversion_function = conversion_function
        self.selection_function = selection_function

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
