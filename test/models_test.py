import unittest
import os

import numpy as np
try:
    import cupy as xp
    from gwpopulation.cupy_utils import trapz
except ImportError:
    xp = np
    trapz = np.trapz

from bilby.core.prior import PriorDict, Uniform
from gwpopulation import models


class TestMassModel(unittest.TestCase):

    def setUp(self):
        self.test_data = dict(
            mass_1=models.norm_array['mass_1'],
            mass_ratio=models.norm_array['mass_ratio'])
        self.test_data['mass_2'] =\
            self.test_data['mass_1'] * self.test_data['mass_ratio']
        self.test_params = dict(
            alpha=1, mmin=9, mmax=40, lam=0, mpp=50, sigpp=1,
            beta=3, delta_m=0)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.prior = PriorDict('{}/test.prior'.format(current_dir))
        self.vt_array = dict(
            mass_1=models.norm_array['mass_1'],
            mass_ratio=models.norm_array['mass_ratio'],
            vt=models.norm_array['mass_1']**0 * 2)
        self.n_test = 10

    def tearDown(self):
        del self.test_data
        del self.test_params
        del self.prior
        del self.vt_array
        del self.n_test

    def test_p_model_1d_normalised(self):
        norms = list()
        for ii in range(self.n_test):
            parameters = self.prior.sample()
            temp = models.pmodel1d(models.m1s, parameters)
            norms.append(trapz(temp, models.m1s))
        self.assertAlmostEqual(float(xp.max(abs(1 - xp.asarray(norms)))), 0)

    def test_p_model_2d_normalised(self):
        norms = list()
        for ii in range(self.n_test):
            parameters = self.prior.sample()
            temp = models.pmodel2d(
                self.test_data['mass_1'], self.test_data['mass_ratio'],
                parameters)
            norms.append(trapz(trapz(temp, models.m1s), models.qs))
        self.assertAlmostEqual(float(xp.max(abs(1 - xp.asarray(norms)))), 0)

    def test_mass_distribution_no_vt_normalised(self):
        norms = list()
        for ii in range(self.n_test):
            parameters = self.prior.sample()
            parameters = models.extract_mass_parameters(parameters)
            temp = models.mass_distribution_no_vt(self.test_data, *parameters)
            norms.append(trapz(trapz(temp, models.m1s), models.qs))
        self.assertAlmostEqual(float(xp.max(abs(1 - xp.asarray(norms)))), 0)

    def test_mass_distribution_vt_normalised(self):
        models.set_vt(self.vt_array)
        norms = list()
        for ii in range(self.n_test):
            parameters = self.prior.sample()
            parameters = models.extract_mass_parameters(parameters)
            temp = models.mass_distribution(self.test_data, *parameters)
            norms.append(trapz(trapz(temp, models.m1s), models.qs))
        self.vt_array['vt'] = xp.ones_like(self.vt_array['mass_ratio']) * 1
        models.set_vt(self.vt_array)
        self.assertAlmostEqual(float(xp.max(abs(0.5 - xp.asarray(norms)))), 0)

    def test_norm_vt(self):
        models.set_vt(self.vt_array)
        norms = list()
        for ii in range(self.n_test):
            parameters = self.prior.sample()
            norms.append(models.norm_vt(parameters))
        self.vt_array['vt'] = xp.ones_like(self.vt_array['mass_ratio']) * 1
        models.set_vt(self.vt_array)
        self.assertAlmostEqual(max(abs(xp.asarray(norms) - 2)), 0)

    def test_iid_mass_normalised(self):
        test_data = dict(
            mass_1=xp.random.uniform(3, 100, 1000000),
            mass_2=xp.random.uniform(3, 100, 1000000))
        norms = list()
        for ii in range(self.n_test):
            parameters = self.prior.sample()
            for key in ['beta', 'xi', 'sigma_1', 'sigma_2',
                        'amax', 'alpha_chi', 'beta_chi', 'rate']:
                parameters.pop(key)
            temp = models.iid_mass(test_data, **parameters)
            norms.append(xp.mean(temp) * (97**2))
        self.assertAlmostEqual(float(xp.mean(norms)), 1, 2)

    def test_marginal_powerlaw_scaling(self):
        parameters = dict(lam=0, mpp=35, sigpp=1, delta_m=0)
        for key in ['lam', 'mpp', 'sigpp', 'delta_m', 'xi', 'sigma_1',
                    'sigma_2', 'amax', 'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        ratios = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = trapz(xp.nan_to_num(
                models.mass_distribution_no_vt(
                    self.test_data, **parameters)).T, models.qs)
            ratios.append((p_pop[200] / p_pop[100]) * (models.m1s[200] /
                          models.m1s[100])**parameters['alpha'])
        self.assertAlmostEqual(max(abs(xp.asarray(ratios) - 1)), 0)

    def test_mass_ratio_scaling(self):
        parameters = dict(lam=0, mpp=35, sigpp=1, delta_m=0)
        for key in ['lam', 'mpp', 'sigpp', 'delta_m', 'xi', 'sigma_1',
                    'sigma_2', 'amax', 'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        ratios = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = xp.nan_to_num(models.mass_distribution_no_vt(
                self.test_data, **parameters)).T
            for ii in range(len(models.qs)):
                if (p_pop[ii, -1] == 0) or (p_pop[ii, 200] == 0):
                    continue
                ratios.append(
                    (p_pop[ii, -1] / p_pop[ii, 200]) /
                    (models.qs[-1] / models.qs[200])**parameters['beta'])
        self.assertAlmostEqual(max(abs(xp.asarray(ratios) - 1)), 0)

    def test_mass_distribution_no_vt_non_negative(self):
        models.set_vt(self.vt_array)
        parameters = dict()
        for key in ['xi', 'sigma_1', 'sigma_2', 'amax',
                    'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        minima = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = xp.nan_to_num(models.mass_distribution_no_vt(
                self.test_data, **parameters))
            minima.append(xp.min(p_pop))
        self.assertGreaterEqual(min(minima), 0)

    def test_mass_distribution_no_vt_returns_zero_below_mmin(self):
        parameters = dict()
        for key in ['xi', 'sigma_1', 'sigma_2', 'amax',
                    'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        max_out_of_bounds = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = xp.nan_to_num(models.mass_distribution_no_vt(
                self.test_data, **parameters))
            max_out_of_bounds.append(xp.max(p_pop[
                (self.test_data['mass_2'] < parameters['mmin'])]))
        self.assertEqual(max(abs(xp.asarray(max_out_of_bounds))), 0)

    def test_powerlaw_mass_distribution_no_vt_returns_zero_above_mmax(self):
        parameters = dict(lam=0.0)
        for key in ['lam', 'xi', 'sigma_1', 'sigma_2', 'amax',
                    'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        max_out_of_bounds = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = xp.nan_to_num(models.mass_distribution_no_vt(
                self.test_data, **parameters))
            max_out_of_bounds.append(xp.max(p_pop[
                (self.test_data['mass_1'] > parameters['mmax'])]))
        self.assertEqual(max(abs(xp.asarray(max_out_of_bounds))), 0)

    def test_mass_distribution_non_negative(self):
        models.set_vt(self.vt_array)
        parameters = dict()
        for key in ['xi', 'sigma_1', 'sigma_2', 'amax',
                    'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        minima = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = xp.nan_to_num(models.mass_distribution(
                self.test_data, **parameters))
            minima.append(xp.min(p_pop))
        self.assertGreaterEqual(min(minima), 0)

    def test_mass_distribution_returns_zero_below_mmin(self):
        models.set_vt(self.vt_array)
        parameters = dict()
        for key in ['xi', 'sigma_1', 'sigma_2', 'amax',
                    'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        max_out_of_bounds = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = xp.nan_to_num(models.mass_distribution(
                self.test_data, **parameters))
            max_out_of_bounds.append(xp.max(p_pop[
                (self.test_data['mass_2'] < parameters['mmin'])]))
        self.assertEqual(max(abs(xp.asarray(max_out_of_bounds))), 0)

    def test_powerlaw_mass_distribution_returns_zero_above_mmax(self):
        parameters = dict(lam=0.0)
        for key in ['lam', 'xi', 'sigma_1', 'sigma_2', 'amax',
                    'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        max_out_of_bounds = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = xp.nan_to_num(models.mass_distribution(
                self.test_data, **parameters))
            max_out_of_bounds.append(xp.max(p_pop[
                (self.test_data['mass_1'] > parameters['mmax'])]))
        self.assertEqual(max(abs(xp.asarray(max_out_of_bounds))), 0)


class TestSpinOrientation(unittest.TestCase):

    def setUp(self):
        self.costilts = xp.linspace(-1, 1, 1000)
        self.test_data = dict(
            cos_tilt_1=xp.einsum('i,j->ij', self.costilts,
                                 xp.ones_like(self.costilts)),
            cos_tilt_2=xp.einsum('i,j->ji', self.costilts,
                                 xp.ones_like(self.costilts)))
        self.prior = PriorDict(
            dict(xi=Uniform(0, 1), sigma_1=Uniform(0, 4),
                 sigma_2=Uniform(0, 4)))
        self.n_test = 100

    def tearDown(self):
        del self.test_data
        del self.prior
        del self.costilts
        del self.n_test

    def test_spin_orientation_normalised(self):
        norms = list()
        for ii in range(self.n_test):
            parameters = self.prior.sample()
            temp = models.spin_orientation_likelihood(
                self.test_data, **parameters)
            norms.append(trapz(trapz(temp, self.costilts), self.costilts))
        self.assertAlmostEqual(float(xp.max(abs(1 - xp.asarray(norms)))), 0, 5)


class TestSpinMagnitude(unittest.TestCase):

    def setUp(self):
        self.a_array = xp.linspace(0, 1, 1000)
        self.test_data = dict(
            a_1=xp.einsum('i,j->ij', self.a_array, xp.ones_like(self.a_array)),
            a_2=xp.einsum('i,j->ji', self.a_array, xp.ones_like(self.a_array)))
        self.prior = PriorDict(
            dict(amax=Uniform(0.3, 1), alpha_chi=Uniform(1, 4),
                 beta_chi=Uniform(1, 4)))
        self.n_test = 100

    def tearDown(self):
        del self.test_data
        del self.prior
        del self.a_array
        del self.n_test

    def test_spin_magnitude_normalised(self):
        norms = list()
        for ii in range(self.n_test):
            parameters = self.prior.sample()
            temp = models.iid_spin_magnitude(self.test_data, **parameters)
            norms.append(trapz(trapz(temp, self.a_array), self.a_array))
        self.assertAlmostEqual(float(xp.max(abs(1 - xp.asarray(norms)))), 0, 1)


if __name__ == '__main__':
    unittest.main()
