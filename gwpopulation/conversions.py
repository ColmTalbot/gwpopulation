from __future__ import division


def convert_to_beta_parameters(parameters, remove=True):
    """
    Convert to parameters for standard beta distribution
    """
    added_keys = []
    converted_parameters = parameters.copy()

    for ii in [1, 2]:
        if 'alpha_chi_{}'.format(ii) not in parameters.keys()\
                or 'beta_chi_{}'.format(ii) not in parameters.keys():
            if 'mu_chi_{}'.format(ii) in converted_parameters.keys():
                if 'sigma_chi_{}'.format(ii) in converted_parameters.keys():
                    converted_parameters['alpha_chi_{}'.format(ii)],\
                        converted_parameters['beta_chi_{}'.format(ii)], _ =\
                        mu_chi_var_chi_max_to_alpha_beta_max(
                            parameters['mu_chi_{}'.format(ii)],
                            parameters['sigma_chi_{}'.format(ii)],
                            parameters['amax'])
                    if remove:
                        added_keys.append('alpha_chi_{}'.format(ii))
                        added_keys.append('beta_chi_{}'.format(ii))
        elif converted_parameters['alpha_chi_{}'.format(ii)] is None or\
                converted_parameters['beta_chi_{}'.format(ii)] is None:
            if 'mu_chi_{}'.format(ii) in converted_parameters.keys():
                if 'sigma_chi_{}'.format(ii) in converted_parameters.keys():
                    converted_parameters['alpha_chi_{}'.format(ii)],\
                        converted_parameters['beta_chi_{}'.format(ii)], _ =\
                        mu_chi_var_chi_max_to_alpha_beta_max(
                            parameters['mu_chi_{}'.format(ii)],
                            parameters['sigma_chi_{}'.format(ii)],
                            parameters['amax'])
                    if remove:
                        added_keys.append('alpha_chi_{}'.format(ii))
                        added_keys.append('beta_chi_{}'.format(ii))

    if 'alpha_chi' not in parameters.keys() or 'beta_chi' not in\
            parameters.keys():
        if 'mu_chi' in converted_parameters.keys():
            if 'sigma_chi' in converted_parameters.keys():
                converted_parameters['alpha_chi'],\
                    converted_parameters['beta_chi'], _ =\
                    mu_chi_var_chi_max_to_alpha_beta_max(
                        parameters['mu_chi'], parameters['sigma_chi'],
                        parameters['amax'])
                if remove:
                    added_keys.append('alpha_chi')
                    added_keys.append('beta_chi')
    elif converted_parameters['alpha_chi'] is None or\
            converted_parameters['beta_chi'] is None:
        if 'mu_chi' in converted_parameters.keys():
            if 'sigma_chi' in converted_parameters.keys():
                converted_parameters['alpha_chi'],\
                    converted_parameters['beta_chi'], _ =\
                    mu_chi_var_chi_max_to_alpha_beta_max(
                        parameters['mu_chi'], parameters['sigma_chi'],
                        parameters['amax'])
                if remove:
                    added_keys.append('alpha_chi')
                    added_keys.append('beta_chi')
    # print(converted_parameters)
    return converted_parameters, added_keys


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
