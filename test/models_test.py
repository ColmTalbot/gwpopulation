import unittest

import numpy as np

from bilby.core.prior import PriorSet, Uniform, LogUniform
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
        self.prior = PriorSet('./test.prior')
        self.vt_array = dict(
            m1=models.norm_array['m1'], q=models.norm_array['q'],
            vt=models.norm_array['m1']**0 * 2)

    def tearDown(self):
        del self.test_data
        del self.test_params
        del self.prior
        del self.vt_array

    def test_p_model_1d_normalised(self):
        norms = list()
        for ii in range(100):
            parameters = self.prior.sample()
            pow_norm = models.norm_ppow(parameters)
            pp_norm = models.norm_pnorm(parameters)
            temp = models.pmodel1d(models.m1s, parameters, pow_norm, pp_norm)
            norms.append(np.trapz(temp, models.m1s))
        self.assertAlmostEqual(np.max(abs(1 - np.array(norms))), 0)

    def test_p_model_2d_normalised(self):
        norms = list()
        for ii in range(100):
            parameters = self.prior.sample()
            pow_norm, pp_norm, qnorms = models.norms(parameters)
            temp = models.pmodel2d(
                self.test_data['m1_source'], self.test_data['q'],
                parameters, pow_norm, pp_norm, qnorms)
            norms.append(np.trapz(np.trapz(temp, models.m1s), models.qs))
        self.assertAlmostEqual(np.max(abs(1 - np.array(norms))), 0)

    def test_mass_distribution_no_vt_normalised(self):
        norms = list()
        for ii in range(100):
            parameters = self.prior.sample()
            parameters = models.extract_mass_parameters(parameters)
            temp = models.mass_distribution_no_vt(self.test_data, *parameters)
            norms.append(np.trapz(np.trapz(temp, models.m1s), models.qs))
        self.assertAlmostEqual(np.max(abs(1 - np.array(norms))), 0)

    def test_mass_distribution_vt_normalised(self):
        models.set_vt(self.vt_array)
        norms = list()
        for ii in range(100):
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
        for ii in range(100):
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

    def tearDown(self):
        del self.test_data
        del self.prior
        del self.costilts

    def test_spin_orientation_normalised(self):
        norms = list()
        for ii in range(100):
            parameters = self.prior.sample()
            temp = models.spin_orientation_likelihood(
                self.test_data, **parameters)
            norms.append(np.trapz(np.trapz(temp, self.costilts), self.costilts))
        self.assertAlmostEqual(np.max(abs(1 - np.array(norms))), 0, 5)


class TestSpinMagnitude(unittest.TestCase):

    def setUp(self):
        self.a_array = np.linspace(1e-5, 1 - 1e-5, 1000)
        self.test_data = dict(
            a1=np.einsum('i,j->ij', self.a_array, np.ones_like(self.a_array)),
            a2=np.einsum('i,j->ji', self.a_array, np.ones_like(self.a_array)))
        self.prior = PriorSet(
            dict(amax=Uniform(0, 1), alpha_chi=LogUniform(1, 1e5),
                 beta_chi=LogUniform(1, 1e5)))

    def tearDown(self):
        del self.test_data
        del self.prior
        del self.a_array

    def test_spin_magnitude_normalised(self):
        norms = list()
        for ii in range(100):
            parameters = self.prior.sample()
            temp = models.iid_spin_magnitude(self.test_data, **parameters)
            norms.append(np.trapz(np.trapz(temp, self.a_array), self.a_array))
        self.assertAlmostEqual(np.max(abs(1 - np.array(norms))), 0)


if __name__ == '__main__':
    unittest.main()
