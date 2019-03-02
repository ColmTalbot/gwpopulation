import unittest

from bilby.core.prior import PriorDict, Uniform

from gwpopulation.cupy_utils import trapz, xp
from gwpopulation.models import spin


class TestSpinOrientation(unittest.TestCase):

    def setUp(self):
        self.costilts = xp.linspace(-1, 1, 1000)
        self.test_data = dict(
            cos_tilt_1=xp.einsum('i,j->ij', self.costilts,
                                 xp.ones_like(self.costilts)),
            cos_tilt_2=xp.einsum('i,j->ji', self.costilts,
                                 xp.ones_like(self.costilts)))
        self.prior = PriorDict(
            dict(xi_spin=Uniform(0, 1), sigma_spin=Uniform(0, 4)))
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
            temp = spin.iid_spin_orientation_gaussian_isotropic(
                self.test_data, **parameters)
            norms.append(trapz(trapz(temp, self.costilts), self.costilts))
        self.assertAlmostEqual(float(xp.max(xp.abs(1 - xp.asarray(norms)))), 0, 5)

    def test_iid_matches_independent_tilts(self):
        iid_params = dict(xi_spin=0.5, sigma_spin=0.5)
        ind_params = dict(xi_spin=0.5, sigma_1=0.5, sigma_2=0.5)
        self.assertEquals(0.0, xp.max(
            spin.iid_spin_orientation_gaussian_isotropic(
                self.test_data, **iid_params) -
            spin.independent_spin_orientation_gaussian_isotropic(
                self.test_data, **ind_params)))


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
            temp = spin.iid_spin_magnitude_beta(
                self.test_data, **parameters)
            norms.append(trapz(trapz(temp, self.a_array), self.a_array))
        self.assertAlmostEqual(
            float(xp.max(xp.abs(1 - xp.asarray(norms)))), 0, 1)

    def test_returns_zero_alpha_beta_less_zero(self):
        parameters = self.prior.sample()
        for key in ['alpha_chi', 'beta_chi']:
            parameters[key] = -1
            self.assertEquals(
                spin.iid_spin_magnitude_beta(
                    self.test_data, **parameters), 0)

    def test_iid_matches_independent_magnitudes(self):
        iid_params = self.prior.sample()
        ind_params = dict()
        ind_params.update({key + '_1': iid_params[key] for key in iid_params})
        ind_params.update({key + '_2': iid_params[key] for key in iid_params})
        self.assertEquals(0.0, xp.max(
            spin.iid_spin_magnitude_beta(self.test_data, **iid_params) -
            spin.independent_spin_magnitude_beta(
                self.test_data, **ind_params)))


class TestIIDSpin(unittest.TestCase):
    def setUp(self):
        self.a_array = xp.linspace(0, 1, 1000)
        self.costilts = xp.linspace(-1, 1, 1000)
        self.test_data = dict(
            a_1=xp.einsum('i,j->ij', self.a_array, xp.ones_like(self.a_array)),
            a_2=xp.einsum('i,j->ji', self.a_array, xp.ones_like(self.a_array)),
            cos_tilt_1=xp.einsum('i,j->ij', self.costilts,
                                 xp.ones_like(self.costilts)),
            cos_tilt_2=xp.einsum('i,j->ji', self.costilts,
                                 xp.ones_like(self.costilts)))
        self.prior = PriorDict(
            dict(amax=Uniform(0.3, 1), alpha_chi=Uniform(1, 4),
                 beta_chi=Uniform(1, 4), xi_spin=Uniform(0, 1),
                 sigma_spin=Uniform(0, 4)))
        self.n_test = 100

    def test_iid_matches_independent(self):
        params = self.prior.sample()
        mag_params = {key: params[key] for key in ['amax', 'alpha_chi', 'beta_chi']}
        tilt_params = {key: params[key] for key in ['xi_spin', 'sigma_spin']}
        self.assertEquals(0.0, xp.max(
            spin.iid_spin(self.test_data, **params) -
            spin.iid_spin_magnitude_beta(self.test_data, **mag_params) *
            spin.iid_spin_orientation_gaussian_isotropic(
                self.test_data, **tilt_params)))


