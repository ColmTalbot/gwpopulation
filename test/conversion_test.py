from gwpopulation.conversions import *

known_values = dict(
    alpha=[1, 2, 2],
    beta=[1, 4, 4],
    amax=[1, 1, 0.5],
    mu=[1 / 2, 1 / 3, 1 / 6],
    var=[1 / 12, 2 / 63, 1 / 126],
)
suffices = ["", "_1", "_2"]


def test_mu_chi_var_chi_max_to_alpha_beta_max():
    for ii in range(3):
        mu = known_values["mu"][ii]
        var = known_values["var"][ii]
        amax = known_values["amax"][ii]
        alpha, beta, _ = mu_var_max_to_alpha_beta_max(mu, var, amax)
        assert abs(alpha - known_values["alpha"][ii]) < 1e-6
        assert abs(beta == known_values["beta"][ii]) < 1e-6
        assert amax == known_values["amax"][ii]


def test_alpha_beta_max_to_mu_chi_var_chi_max():
    for ii in range(3):
        alpha = known_values["alpha"][ii]
        beta = known_values["beta"][ii]
        amax = known_values["amax"][ii]
        mu, var, _ = alpha_beta_max_to_mu_var_max(alpha, beta, amax)
        assert mu == known_values["mu"][ii]
        assert var == known_values["var"][ii]


def test_convert_to_beta_parameters():
    for ii, suffix in enumerate(suffices):
        params = dict()
        params["mu_chi" + suffix] = known_values["mu"][ii]
        params["sigma_chi" + suffix] = known_values["var"][ii]
        params["amax" + suffix] = known_values["amax"][ii]
        new_params, _ = convert_to_beta_parameters(params, remove=True)
        full_dict = params.copy()
        alpha, beta, _ = mu_var_max_to_alpha_beta_max(
            mu=params["mu_chi" + suffix],
            var=params["sigma_chi" + suffix],
            amax=params["amax" + suffix],
        )
        full_dict["alpha_chi" + suffix] = alpha
        full_dict["beta_chi" + suffix] = beta
        print(new_params, full_dict)
        for key in full_dict:
            assert new_params[key] == full_dict[key]


def test_convert_to_beta_parameters_with_none():
    params = dict(amax=1, alpha_chi=None, beta_chi=None, mu_chi=0.5, sigma_chi=0.1)
    _, added = convert_to_beta_parameters(params, remove=True)
    assert len(added) == 2


def test_convert_to_beta_parameters_unnecessary():
    params = dict(amax=1, alpha_chi=1, beta_chi=1)
    _, added = convert_to_beta_parameters(params, remove=True)
    assert len(added) == 0
