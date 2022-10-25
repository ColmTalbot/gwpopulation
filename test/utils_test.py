import unittest

import numpy as np
from scipy.stats import vonmises

from gwpopulation import utils


class TestBetaDist(unittest.TestCase):
    def setUp(self):
        self.n_test = 100
        self.alphas = np.random.uniform(0, 10, self.n_test)
        self.betas = np.random.uniform(0, 10, self.n_test)
        self.scales = np.random.uniform(0, 10, self.n_test)
        pass

    def test_beta_dist_zero_below_zero(self):
        equal_zero = True
        for ii in range(self.n_test):
            vals = utils.beta_dist(
                xx=-1, alpha=self.alphas[ii], beta=self.betas[ii], scale=self.scales[ii]
            )
            equal_zero = equal_zero & (vals == 0.0)
        self.assertTrue(equal_zero)

    def test_beta_dist_zero_above_scale(self):
        equal_zero = True
        for ii in range(self.n_test):
            xx = self.scales[ii] + 1
            vals = utils.beta_dist(
                xx=xx, alpha=self.alphas[ii], beta=self.betas[ii], scale=self.scales[ii]
            )
            equal_zero = equal_zero & (vals == 0.0)
        self.assertTrue(equal_zero)

    def test_beta_dist_alpha_below_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            utils.beta_dist(xx=0.5, alpha=-1, beta=1, scale=1)

    def test_beta_dist_beta_below_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            utils.beta_dist(xx=0.5, alpha=1, beta=-1, scale=1)


class TestPowerLaw(unittest.TestCase):
    def setUp(self):
        self.n_test = 100
        self.alphas = np.random.uniform(-10, 10, self.n_test)
        self.lows = np.random.uniform(5, 15, self.n_test)
        self.highs = np.random.uniform(20, 30, self.n_test)

    def test_powerlaw_zero_below_low(self):
        equal_zero = True
        for ii in range(self.n_test):
            xx = self.lows[ii] - 1
            vals = utils.powerlaw(
                xx=xx, alpha=self.alphas[ii], low=self.lows[ii], high=self.highs[ii]
            )
            equal_zero = equal_zero & (vals == 0.0)
        self.assertTrue(equal_zero)

    def test_powerlaw_zero_above_high(self):
        equal_zero = True
        for ii in range(self.n_test):
            xx = self.highs[ii] + 1
            vals = utils.powerlaw(
                xx=xx, alpha=self.alphas[ii], low=self.lows[ii], high=self.highs[ii]
            )
            equal_zero = equal_zero & (vals == 0.0)
        self.assertTrue(equal_zero)

    def test_powerlaw_low_below_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            utils.powerlaw(xx=0, alpha=3, high=10, low=-4)

    def test_powerlaw_alpha_equal_zero(self):
        self.assertEqual(
            utils.powerlaw(xx=1.0, alpha=-1, low=0.5, high=2), 1 / np.log(4)
        )


class TestTruncNorm(unittest.TestCase):
    def setUp(self):
        self.n_test = 100
        self.mus = np.random.uniform(-10, 10, self.n_test)
        self.sigmas = np.random.uniform(0, 10, self.n_test)
        self.lows = np.random.uniform(-30, -20, self.n_test)
        self.highs = np.random.uniform(20, 30, self.n_test)
        pass

    def test_truncnorm_zero_below_low(self):
        equal_zero = True
        for ii in range(self.n_test):
            xx = self.lows[ii] - 1
            vals = utils.truncnorm(
                xx=xx,
                mu=self.mus[ii],
                sigma=self.sigmas[ii],
                low=self.lows[ii],
                high=self.highs[ii],
            )
            equal_zero = equal_zero & (vals == 0.0)
        self.assertTrue(equal_zero)

    def test_truncnorm_zero_above_high(self):
        equal_zero = True
        for ii in range(self.n_test):
            xx = self.highs[ii] + 1
            vals = utils.truncnorm(
                xx=xx,
                mu=self.mus[ii],
                sigma=self.sigmas[ii],
                low=self.lows[ii],
                high=self.highs[ii],
            )
            equal_zero = equal_zero & (vals == 0.0)
        self.assertTrue(equal_zero)

    def test_truncnorm_sigma_below_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            utils.truncnorm(xx=0, mu=0, sigma=-1, high=10, low=-10)


class TestVonMises(unittest.TestCase):
    def setUp(self):
        self.n_test = 100
        self.mus = np.random.uniform(-np.pi, np.pi, self.n_test)
        self.kappas = np.random.uniform(0, 15, self.n_test)
        self.xx = np.linspace(0, 2 * np.pi, 1000)

    def test_matches_scipy(self):
        for ii in range(self.n_test):
            mu = self.mus[ii]
            kappa = self.kappas[ii]
            gwpop_vals = utils.von_mises(self.xx, mu, kappa)
            scipy_vals = vonmises(kappa=kappa, loc=mu).pdf(self.xx)
            self.assertAlmostEqual(max(abs(gwpop_vals - scipy_vals)), 0)
