import unittest

from bilby.core.prior import PriorDict, Uniform

from gwpopulation.cupy_utils import trapz, xp
from gwpopulation.models import spin
from gwpopulation.utils import truncnorm


class TestSpinOrientation(unittest.TestCase):
    def setUp(self):
        self.costilts = xp.linspace(-1, 1, 1000)
        self.test_data = dict(
            cos_tilt_1=xp.einsum("i,j->ij", self.costilts, xp.ones_like(self.costilts)),
            cos_tilt_2=xp.einsum("i,j->ji", self.costilts, xp.ones_like(self.costilts)),
        )
        self.prior = PriorDict(dict(xi_spin=Uniform(0, 1), sigma_spin=Uniform(0, 4)))
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
                self.test_data, **parameters
            )
            norms.append(trapz(trapz(temp, self.costilts), self.costilts))
        self.assertAlmostEqual(float(xp.max(xp.abs(1 - xp.asarray(norms)))), 0, 5)

    def test_iid_matches_independent_tilts(self):
        iid_params = dict(xi_spin=0.5, sigma_spin=0.5)
        ind_params = dict(xi_spin=0.5, sigma_1=0.5, sigma_2=0.5)
        self.assertEqual(
            0.0,
            xp.max(
                spin.iid_spin_orientation_gaussian_isotropic(
                    self.test_data, **iid_params
                )
                - spin.independent_spin_orientation_gaussian_isotropic(
                    self.test_data, **ind_params
                )
            ),
        )


class TestSpinMagnitude(unittest.TestCase):
    def setUp(self):
        self.a_array = xp.linspace(0, 1, 1000)
        self.test_data = dict(
            a_1=xp.einsum("i,j->ij", self.a_array, xp.ones_like(self.a_array)),
            a_2=xp.einsum("i,j->ji", self.a_array, xp.ones_like(self.a_array)),
        )
        self.prior = PriorDict(
            dict(amax=Uniform(0.3, 1), alpha_chi=Uniform(1, 4), beta_chi=Uniform(1, 4))
        )
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
            temp = spin.iid_spin_magnitude_beta(self.test_data, **parameters)
            norms.append(trapz(trapz(temp, self.a_array), self.a_array))
        self.assertAlmostEqual(float(xp.max(xp.abs(1 - xp.asarray(norms)))), 0, 1)

    def test_returns_zero_alpha_beta_less_zero(self):
        parameters = self.prior.sample()
        for key in ["alpha_chi", "beta_chi"]:
            parameters[key] = -1
            self.assertEqual(
                spin.iid_spin_magnitude_beta(self.test_data, **parameters), 0
            )

    def test_iid_matches_independent_magnitudes(self):
        iid_params = self.prior.sample()
        ind_params = dict()
        ind_params.update({key + "_1": iid_params[key] for key in iid_params})
        ind_params.update({key + "_2": iid_params[key] for key in iid_params})
        self.assertEqual(
            0.0,
            xp.max(
                spin.iid_spin_magnitude_beta(self.test_data, **iid_params)
                - spin.independent_spin_magnitude_beta(self.test_data, **ind_params)
            ),
        )


class TestIIDSpin(unittest.TestCase):
    def setUp(self):
        self.a_array = xp.linspace(0, 1, 1000)
        self.costilts = xp.linspace(-1, 1, 1000)
        self.test_data = dict(
            a_1=xp.einsum("i,j->ij", self.a_array, xp.ones_like(self.a_array)),
            a_2=xp.einsum("i,j->ji", self.a_array, xp.ones_like(self.a_array)),
            cos_tilt_1=xp.einsum("i,j->ij", self.costilts, xp.ones_like(self.costilts)),
            cos_tilt_2=xp.einsum("i,j->ji", self.costilts, xp.ones_like(self.costilts)),
        )
        self.prior = PriorDict(
            dict(
                amax=Uniform(0.3, 1),
                alpha_chi=Uniform(1, 4),
                beta_chi=Uniform(1, 4),
                xi_spin=Uniform(0, 1),
                sigma_spin=Uniform(0, 4),
            )
        )
        self.n_test = 100

    def test_iid_matches_independent(self):
        params = self.prior.sample()
        mag_params = {key: params[key] for key in ["amax", "alpha_chi", "beta_chi"]}
        tilt_params = {key: params[key] for key in ["xi_spin", "sigma_spin"]}
        self.assertEqual(
            0.0,
            xp.max(
                spin.iid_spin(self.test_data, **params)
                - spin.iid_spin_magnitude_beta(self.test_data, **mag_params)
                * spin.iid_spin_orientation_gaussian_isotropic(
                    self.test_data, **tilt_params
                )
            ),
        )


class TestGaussianSpin(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_gaussian_chi_eff(self):
        self.assertTrue(
            xp.all(
                spin.gaussian_chi_eff(
                    dict(chi_eff=xp.linspace(-2, 2, 1001)),
                    mu_chi_eff=0.4,
                    sigma_chi_eff=0.1,
                )
                == truncnorm(
                    xp.linspace(-2, 2, 1001), mu=0.4, sigma=0.1, low=-1, high=1
                )
            )
        )

    def test_gaussian_chi_p(self):
        self.assertTrue(
            xp.all(
                spin.gaussian_chi_p(
                    dict(chi_p=xp.linspace(-2, 2, 1001)), mu_chi_p=0.4, sigma_chi_p=0.1
                )
                == truncnorm(xp.linspace(-2, 2, 1001), mu=0.4, sigma=0.1, low=0, high=1)
            )
        )

    def test_2d_gaussian_normalized(self):
        model = spin.GaussianChiEffChiP()
        chi_eff, chi_p = xp.meshgrid(xp.linspace(-1, 1, 501), xp.linspace(0, 1, 300))
        parameters = dict(
            mu_chi_eff=0.1,
            mu_chi_p=0.3,
            sigma_chi_eff=0.6,
            sigma_chi_p=0.5,
            spin_covariance=0.9,
        )
        prob = model(dict(chi_eff=chi_eff, chi_p=chi_p), **parameters)
        self.assertAlmostEqual(
            xp.trapz(xp.trapz(prob, xp.linspace(-1, 1, 501)), xp.linspace(0, 1, 300)),
            1,
            5,
        )

    def test_2d_gaussian_no_covariance_matches_independent(self):
        model = spin.GaussianChiEffChiP()
        chi_eff, chi_p = xp.meshgrid(xp.linspace(-1, 1, 501), xp.linspace(0, 1, 300))
        data = dict(chi_eff=chi_eff, chi_p=chi_p)
        self.assertTrue(
            xp.all(
                spin.gaussian_chi_eff(data, mu_chi_eff=0.6, sigma_chi_eff=0.2)
                * spin.gaussian_chi_p(data, mu_chi_p=0.4, sigma_chi_p=0.1)
                == model(
                    data,
                    mu_chi_eff=0.6,
                    mu_chi_p=0.4,
                    sigma_chi_eff=0.2,
                    sigma_chi_p=0.1,
                    spin_covariance=0.0,
                )
            )
        )
