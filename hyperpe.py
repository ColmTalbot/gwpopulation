from __future__ import division, print_function
from bilby.hyper.likelihood import HyperparameterLikelihood
import numpy as np
import models


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
                 max_samples=1e100, analysis_time=1,
                 conversion_function=lambda args: (args, None)):
        super(RateLikelihood, self).__init__(posteriors, hyper_prior,
                                             sampling_prior, max_samples)
        self.analysis_time = analysis_time
        self.conversion_function = conversion_function

    def log_likelihood(self):
        self.parameters, added_keys = self.conversion_function(self.parameters)
        log_l = HyperparameterLikelihood.log_likelihood(self)
        log_l += self.n_posteriors * np.log(self.parameters['rate'])
        log_l -= models.norm_vt(self.parameters) * self.parameters['rate'] *\
            self.analysis_time
        for key in added_keys:
            self.parameters.pop(key)
        return np.nan_to_num(log_l)
