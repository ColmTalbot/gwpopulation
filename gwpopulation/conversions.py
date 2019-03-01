from __future__ import division


def convert_to_beta_parameters(parameters, remove=True):
    """
    Convert to parameters for standard beta distribution
    """
    added_keys = list()
    converted = parameters.copy()

    for suffix in ['', '_1', '_2']:
        alpha = 'alpha_chi{}'.format(suffix)
        beta = 'beta_chi{}'.format(suffix)
        mu = 'mu_chi{}'.format(suffix)
        sigma = 'sigma_chi{}'.format(suffix)
        amax = 'amax{}'.format(suffix)

        if alpha not in parameters.keys() or beta not in parameters.keys():
            needed = True
        elif converted[alpha] is None or converted[beta] is None:
            needed = True
        else:
            needed = False

        if needed:
            if mu in converted.keys() and sigma in converted.keys():
                converted[alpha], converted[beta], _ =\
                    mu_chi_var_chi_max_to_alpha_beta_max(
                        parameters[mu], parameters[sigma],
                        parameters[amax])
                if remove:
                    added_keys.append(alpha)
                    added_keys.append(beta)

    return converted, added_keys


def alpha_beta_max_to_mu_chi_var_chi_max(alpha, beta, amax):
    """
    Convert between parameters for beta distribution
    """
    mu_chi = alpha / (alpha + beta) * amax
    var_chi = alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1)) * amax**2
    return mu_chi, var_chi, amax


def mu_chi_var_chi_max_to_alpha_beta_max(mu_chi, var_chi, amax):
    """
    Convert between parameters for beta distribution
    """
    mu_chi /= amax
    var_chi /= amax**2
    alpha = (mu_chi**2 * (1 - mu_chi) - mu_chi * var_chi) / var_chi
    beta = (mu_chi * (1 - mu_chi)**2 - (1 - mu_chi) * var_chi) / var_chi
    return alpha, beta, amax
