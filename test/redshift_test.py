from __future__ import division

import unittest

from astropy.cosmology import Planck15
from bilby.core.prior import PriorDict, Uniform
import numpy as np

from gwpopulation.models import redshift
from gwpopulation.cupy_utils import trapz, xp


class TestRedshift(unittest.TestCase):
    def setUp(self):
        self.zs = xp.linspace(1e-3, 1, 1000)
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
            Planck15.differential_comoving_volume(self.zs).value * 4 * np.pi, self.zs,
        )
        self.assertEqual(
            total_volume, model.total_spacetime_volume(**parameters),
        )
