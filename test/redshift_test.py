import unittest

import numpy as np
from astropy.cosmology import Planck15
from bilby.core.prior import PriorDict, Uniform

from gwpopulation.cupy_utils import trapz, xp
from gwpopulation.models import redshift


class TestRedshift(unittest.TestCase):
    def setUp(self):
        self.zs = xp.linspace(1e-3, 2.3, 1000)
        self.test_data = dict(redshift=self.zs)
        self.n_test = 100

    def _run_model_normalisation(self, model, priors):
        norms = list()
        for _ in range(self.n_test):
            p_z = model(self.test_data, **priors.sample())
            norms.append(trapz(p_z, self.zs))
        self.assertAlmostEqual(xp.max(xp.abs(xp.asarray(norms) - 1)), 0.0)

    def test_powerlaw_normalised(self):
        model = redshift.PowerLawRedshift()
        priors = PriorDict()
        priors["lamb"] = Uniform(-15, 15)
        self._run_model_normalisation(model=model, priors=priors)

    def test_madau_dickinson_normalised(self):
        model = redshift.MadauDickinsonRedshift()
        priors = PriorDict()
        priors["gamma"] = Uniform(-15, 15)
        priors["kappa"] = Uniform(-15, 15)
        priors["z_peak"] = Uniform(0, 5)
        self._run_model_normalisation(model=model, priors=priors)

    def test_powerlaw_volume(self):
        """
        Test that the total volume matches the expected value for a
        trivial case
        """
        model = redshift.PowerLawRedshift()
        parameters = dict(lamb=1)
        total_volume = np.trapz(
            Planck15.differential_comoving_volume(self.zs).value * 4 * np.pi,
            self.zs,
        )
        self.assertAlmostEqual(
            total_volume,
            model.normalisation(parameters),
            places=3,
        )

    def test_zero_outside_domain(self):
        model = redshift.PowerLawRedshift(z_max=1)
        parameters = dict(lamb=1)
        p_z = model(self.test_data, **parameters)
        self.assertTrue(all(p_z[self.zs > 1] == 0))


class TestFourVolume(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_four_volume(self):
        self.assertAlmostEqual(
            Planck15.comoving_volume(2.3).value / 1e9,
            redshift.total_four_volume(lamb=1, analysis_time=1, max_redshift=2.3),
            4,
        )
