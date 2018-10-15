import unittest
import os

import numpy as np

from bilby.core.prior import PriorSet, Uniform
from population import models


class TestMassModel(unittest.TestCase):

    def setUp(self):
        self.test_data = dict(
            m1_source=models.norm_array['m1'], q=models.norm_array['q'],
            arg_m1s=np.einsum('i,j->ji', np.arange(0, len(models.m1s)),
                              np.ones_like(models.qs)).astype(int))
        self.test_data['m2_source'] =\
            self.test_data['m1_source'] * self.test_data['q']
        self.test_params = dict(
            alpha=1, mmin=9, mmax=40, lam=0, mpp=50, sigpp=1,
            beta=3, delta_m=0)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.prior = PriorSet('{}/test.prior'.format(current_dir))
        self.vt_array = dict(
            m1=models.norm_array['m1'], q=models.norm_array['q'],
            vt=models.norm_array['m1']**0 * 2)
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
            pow_norm = models.norm_ppow(parameters)
            pp_norm = models.norm_pnorm(parameters)
            temp = models.pmodel1d(models.m1s, parameters, pow_norm, pp_norm)
            norms.append(np.trapz(temp, models.m1s))
        self.assertAlmostEqual(np.max(abs(1 - np.array(norms))), 0)

    def test_p_model_2d_normalised(self):
        norms = list()
        for ii in range(self.n_test):
            parameters = self.prior.sample()
            pow_norm, pp_norm, qnorms = models.norms(parameters)
            temp = models.pmodel2d(
                self.test_data['m1_source'], self.test_data['q'],
                parameters, pow_norm, pp_norm, qnorms)
            norms.append(np.trapz(np.trapz(temp, models.m1s), models.qs))
        self.assertAlmostEqual(np.max(abs(1 - np.array(norms))), 0)

    def test_mass_distribution_no_vt_normalised(self):
        norms = list()
        for ii in range(self.n_test):
            parameters = self.prior.sample()
            parameters = models.extract_mass_parameters(parameters)
            temp = models.mass_distribution_no_vt(self.test_data, *parameters)
            norms.append(np.trapz(np.trapz(temp, models.m1s), models.qs))
        self.assertAlmostEqual(np.max(abs(1 - np.array(norms))), 0)

    def test_mass_distribution_vt_normalised(self):
        models.set_vt(self.vt_array)
        norms = list()
        for ii in range(self.n_test):
            parameters = self.prior.sample()
            parameters = models.extract_mass_parameters(parameters)
            temp = models.mass_distribution(self.test_data, *parameters)
            norms.append(np.trapz(np.trapz(temp, models.m1s), models.qs))
        self.vt_array['vt'] = np.ones_like(self.vt_array['q']) * 1
        models.set_vt(self.vt_array)
        self.assertAlmostEqual(np.max(abs(0.5 - np.array(norms))), 0)

    def test_norm_vt(self):
        models.set_vt(self.vt_array)
        norms = list()
        for ii in range(self.n_test):
            parameters = self.prior.sample()
            norms.append(models.norm_vt(parameters))
        self.vt_array['vt'] = np.ones_like(self.vt_array['q']) * 1
        models.set_vt(self.vt_array)
        self.assertAlmostEqual(max(abs(np.array(norms))), 2)

    # def test_iid_mass_normalised(self):
    #     norms = list()
    #     for ii in range(100):
    #         parameters = self.prior.sample()
    #         for key in ['beta', 'xi', 'sigma_1', 'sigma_2',
    #                     'amax', 'alpha_chi', 'beta_chi', 'rate']:
    #             parameters.pop(key)
    #         temp = models.iid_mass(self.test_data, **parameters)
    #         norms.append(np.trapz(np.trapz(
    #             temp.T * self.test_data['m1_source'].T, models.qs),
    #             models.m1s))
    #     self.assertAlmostEqual(np.max(abs(1 - np.array(norms))), 0)

    def test_marginal_powerlaw_scaling(self):
        parameters = dict(lam=0, mpp=35, sigpp=1, delta_m=0)
        for key in ['lam', 'mpp', 'sigpp', 'delta_m', 'xi', 'sigma_1',
                    'sigma_2', 'amax', 'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        ratios = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = np.trapz(np.nan_to_num(
                models.mass_distribution_no_vt(
                    self.test_data, **parameters)).T, models.qs)
            ratios.append((p_pop[200] / p_pop[100]) * (models.m1s[200] /
                          models.m1s[100])**parameters['alpha'])
        self.assertAlmostEqual(max(abs(np.array(ratios) - 1)), 0)

    def test_mass_ratio_scaling(self):
        parameters = dict(lam=0, mpp=35, sigpp=1, delta_m=0)
        for key in ['lam', 'mpp', 'sigpp', 'delta_m', 'xi', 'sigma_1',
                    'sigma_2', 'amax', 'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        ratios = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = np.nan_to_num(models.mass_distribution_no_vt(
                self.test_data, **parameters)).T
            for ii in range(len(models.qs)):
                if (p_pop[ii, -1] == 0) or (p_pop[ii, 200] == 0):
                    continue
                ratios.append(
                    (p_pop[ii, -1] / p_pop[ii, 200]) /
                    (models.qs[-1] / models.qs[200])**parameters['beta'])
        self.assertAlmostEqual(max(abs(np.array(ratios) - 1)), 0)

    def test_mass_distribution_no_vt_non_negative(self):
        models.set_vt(self.vt_array)
        parameters = dict()
        for key in ['xi', 'sigma_1', 'sigma_2', 'amax',
                    'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        minima = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = np.nan_to_num(models.mass_distribution_no_vt(
                self.test_data, **parameters))
            minima.append(np.min(p_pop))
        self.assertGreaterEqual(min(minima), 0)

    def test_mass_distribution_no_vt_returns_zero_below_mmin(self):
        parameters = dict()
        for key in ['xi', 'sigma_1', 'sigma_2', 'amax',
                    'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        max_out_of_bounds = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = np.nan_to_num(models.mass_distribution_no_vt(
                self.test_data, **parameters))
            max_out_of_bounds.append(np.max(p_pop[
                (self.test_data['m2_source'] < parameters['mmin'])]))
        self.assertEqual(max(abs(np.array(max_out_of_bounds))), 0)

    def test_powerlaw_mass_distribution_no_vt_returns_zero_above_mmax(self):
        parameters = dict(lam=0.0)
        for key in ['lam', 'xi', 'sigma_1', 'sigma_2', 'amax',
                    'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        max_out_of_bounds = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = np.nan_to_num(models.mass_distribution_no_vt(
                self.test_data, **parameters))
            max_out_of_bounds.append(np.max(p_pop[
                (self.test_data['m1_source'] > parameters['mmax'])]))
        self.assertEqual(max(abs(np.array(max_out_of_bounds))), 0)

    def test_mass_distribution_non_negative(self):
        models.set_vt(self.vt_array)
        parameters = dict()
        for key in ['xi', 'sigma_1', 'sigma_2', 'amax',
                    'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        minima = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = np.nan_to_num(models.mass_distribution(
                self.test_data, **parameters))
            minima.append(np.min(p_pop))
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
            p_pop = np.nan_to_num(models.mass_distribution(
                self.test_data, **parameters))
            max_out_of_bounds.append(np.max(p_pop[
                (self.test_data['m2_source'] < parameters['mmin'])]))
        self.assertEqual(max(abs(np.array(max_out_of_bounds))), 0)
        
    def test_powerlaw_mass_distribution_returns_zero_above_mmax(self):
        parameters = dict(lam=0.0)
        for key in ['lam', 'xi', 'sigma_1', 'sigma_2', 'amax',
                    'alpha_chi', 'beta_chi', 'rate']:
            self.prior.pop(key)
        max_out_of_bounds = list()
        for ii in range(self.n_test):
            parameters.update(self.prior.sample())
            p_pop = np.nan_to_num(models.mass_distribution(
                self.test_data, **parameters))
            max_out_of_bounds.append(np.max(p_pop[
                (self.test_data['m1_source'] > parameters['mmax'])]))
        self.assertEqual(max(abs(np.array(max_out_of_bounds))), 0)


class TestSpinOrientation(unittest.TestCase):

    def setUp(self):
        self.costilts = np.linspace(-1, 1, 1000)
        self.test_data = dict(
            costilt1=np.einsum('i,j->ij', self.costilts,
                               np.ones_like(self.costilts)),
            costilt2=np.einsum('i,j->ji', self.costilts,
                               np.ones_like(self.costilts)))
        self.prior = PriorSet(
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
            norms.append(np.trapz(np.trapz(temp, self.costilts), self.costilts))
        self.assertAlmostEqual(np.max(abs(1 - np.array(norms))), 0, 5)


class TestSpinMagnitude(unittest.TestCase):

    def setUp(self):
        self.a_array = np.linspace(0, 1, 1000)
        self.test_data = dict(
            a1=np.einsum('i,j->ij', self.a_array, np.ones_like(self.a_array)),
            a2=np.einsum('i,j->ji', self.a_array, np.ones_like(self.a_array)))
        self.prior = PriorSet(
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
            norms.append(np.trapz(np.trapz(temp, self.a_array), self.a_array))
        self.assertAlmostEqual(np.max(abs(1 - np.array(norms))), 0, 2)


if __name__ == '__main__':
    unittest.main()
