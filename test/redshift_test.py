from __future__ import division

import unittest

import numpy as np

from gwpopulation.models import redshift
from gwpopulation.cupy_utils import trapz, xp


class TestFHFRedshift(unittest.TestCase):

    def setUp(self):
        self.zs = xp.linspace(1e-3, 1, 1000)
        self.test_data = dict(redshift=self.zs)
        self.n_test = 100

    def test_fhf_normalised(self):
        norms = list()
        for _ in range(self.n_test):
            lamb = np.random.uniform(-15, 15)
            p_z = redshift.power_law_redshift(self.test_data, lamb)
            norms.append(trapz(p_z, self.zs))
        self.assertAlmostEqual(xp.max(xp.abs(xp.asarray(norms) - 1)), 0.0)
