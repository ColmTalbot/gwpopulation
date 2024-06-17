"""
Parameter conversions
"""


def convert_to_beta_parameters(parameters, remove=True):
    """
    Convert to parameters for standard beta distribution.

    Calls gwpopulation.conversions.mu_var_max_to_alpha_beta_max

    Parameters
    ==========
    parameters: dict
        The input parameters
    remove: bool
        Whether to list the added keys

    Returns
    =======
    converted: dict
        The dictionary of parameters with the new parameters added
    added_keys: list
        The keys added to the dictionary, only non-empty if `remove=True`
    """
    added_keys = list()
    converted = parameters.copy()

    def _convert(suffix):
        alpha = f"alpha_chi{suffix}"
        beta = f"beta_chi{suffix}"
        mu = f"mu_chi{suffix}"
        sigma = f"sigma_chi{suffix}"
        amax = f"amax{suffix}"

        if alpha not in parameters.keys() or beta not in parameters.keys():
            needed = True
        elif converted[alpha] is None or converted[beta] is None:
            needed = True
        else:
            needed = False
            done = True

        if needed:
            if mu in converted.keys() and sigma in converted.keys():
                done = True
                (converted[alpha], converted[beta], _,) = mu_var_max_to_alpha_beta_max(
                    parameters[mu], parameters[sigma], parameters[amax]
                )
                if remove:
                    added_keys.append(alpha)
                    added_keys.append(beta)
            else:
                done = False
        return done

    done = False

    for suffix in ["_1", "_2"]:
        _done = _convert(suffix)
        done = done or _done
    if not done:
        _ = _convert("")

    return converted, added_keys


def alpha_beta_max_to_mu_var_max(alpha, beta, amax):
    r"""
    Convert between parameters for beta distribution

    .. math::
        \mu &= a_\max \frac{\alpha}{\alpha + \beta}

        \sigma^2 &= a_\max^2 \frac{\alpha\beta}{(\alpha + \beta)^2 + (\alpha + \beta + 1)^2}

    Parameters
    ==========
    alpha: float
        The Beta alpha parameter (:math:`\alpha`)
    beta: float
        The Beta beta parameter (:math:`\beta`)
    amax: float
        The maximum value (:math:`a_\max`)

    Returns
    =======
    mu: float
        The mean (:math:`\mu`)
    var: float
        The variance (:math:`\sigma^2`)
    amax: float
        The maximum spin (:math:`a_\max`)
    """
    mu = alpha / (alpha + beta) * amax
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)) * amax**2
    return mu, var, amax


def mu_var_max_to_alpha_beta_max(mu, var, amax):
    r"""
    Convert between parameters for beta distribution

    .. math::
        \alpha &= \frac{\mu^2 (a_\max - \mu) - \mu \sigma^2}{a_\max\sigma^2}

        \beta &= \frac{\mu (a_\max - \mu)^2 - (a_\max - \mu) \sigma^2}{a_\max\sigma^2}

    Parameters
    ==========
    mu: float
        The mean (:math:`\mu`)
    var: float
        The variance (:math:`\sigma^2`)
    amax: float
        The maximum value (:math:`a_\max`)

    Returns
    =======
    alpha: float
        The Beta alpha parameter (:math:`\alpha`)
    beta: float
        The Beta beta parameter (:math:`\beta`)
    amax: float
        The maximum spin (:math:`a_\max`)
    """
    mu /= amax
    var /= amax**2
    alpha = (mu**2 * (1 - mu) - mu * var) / var
    beta = (mu * (1 - mu) ** 2 - (1 - mu) * var) / var
    return alpha, beta, amax
