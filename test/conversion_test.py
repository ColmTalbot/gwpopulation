import unittest

from gwpopulation.conversions import *


class TestBetaConversion(unittest.TestCase):
    def setUp(self):
        self.known_values = dict(
            alpha=[1, 2, 2],
            beta=[1, 4, 4],
            amax=[1, 1, 0.5],
            mu=[1 / 2, 1 / 3, 1 / 6],
            var=[1 / 12, 2 / 63, 1 / 126],
        )
        self.suffices = ["", "_1", "_2"]

    def tearDown(self):
        pass

    def test_mu_chi_var_chi_max_to_alpha_beta_max(self):
        for ii in range(3):
            mu = self.known_values["mu"][ii]
            var = self.known_values["var"][ii]
            amax = self.known_values["amax"][ii]
            alpha, beta, _ = mu_var_max_to_alpha_beta_max(mu, var, amax)
            self.assertAlmostEqual(alpha, self.known_values["alpha"][ii])
            self.assertAlmostEqual(beta, self.known_values["beta"][ii])
            self.assertAlmostEqual(amax, self.known_values["amax"][ii])

    def test_alpha_beta_max_to_mu_chi_var_chi_max(self):
        for ii in range(3):
            alpha = self.known_values["alpha"][ii]
            beta = self.known_values["beta"][ii]
            amax = self.known_values["amax"][ii]
            mu, var, _ = alpha_beta_max_to_mu_var_max(alpha, beta, amax)
            self.assertAlmostEqual(mu, self.known_values["mu"][ii])
            self.assertAlmostEqual(var, self.known_values["var"][ii])

    def test_convert_to_beta_parameters(self):
        for ii, suffix in enumerate(self.suffices):
            params = dict()
            params["mu_chi" + suffix] = self.known_values["mu"][ii]
            params["sigma_chi" + suffix] = self.known_values["var"][ii]
            params["amax" + suffix] = self.known_values["amax"][ii]
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
                self.assertAlmostEqual(new_params[key], full_dict[key])

    def test_convert_to_beta_parameters_with_none(self):
        params = dict(amax=1, alpha_chi=None, beta_chi=None, mu_chi=0.5, sigma_chi=0.1)
        new_params, added = convert_to_beta_parameters(params, remove=True)
        self.assertTrue(len(added) == 2)

    def test_convert_to_beta_parameters_unnecessary(self):
        params = dict(amax=1, alpha_chi=1, beta_chi=1)
        new_params, added = convert_to_beta_parameters(params, remove=True)
        self.assertTrue(len(added) == 0)
