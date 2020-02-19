from __future__ import division

import unittest

from bilby.core.prior import PriorDict, Uniform

from gwpopulation.models import redshift
from gwpopulation.cupy_utils import trapz, xp


class TestRedshift(unittest.TestCase):
    def setUp(self):
        self.zs = xp.linspace(1e-3, 1, 1000)
        self.test_data = dict(redshift=self.zs)
        self.n_test = 100

    def _run_model(self, model, priors):
        norms = list()
        for _ in range(self.n_test):
            p_z = model(self.test_data, **priors.sample())
            norms.append(trapz(p_z, self.zs))
        self.assertAlmostEqual(xp.max(xp.abs(xp.asarray(norms) - 1)), 0.0)

    def test_powerlaw_normalised(self):
        model = redshift.PowerLawRedshift()
        priors = PriorDict()
        priors["lamb"] = Uniform(-15, 15)
        self._run_model(model=model, priors=priors)

    def test_madau_dickinson_normalised(self):
        model = redshift.MaduaDickinsonRedshift()
        priors = PriorDict()
        priors["a_z"] = Uniform(-15, 15)
        priors["b_z"] = Uniform(-15, 15)
        priors["z_peak"] = Uniform(0, model.z_max)
        self._run_model(model=model, priors=priors)
