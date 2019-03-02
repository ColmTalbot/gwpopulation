import unittest

from gwpopulation import vt
from gwpopulation.cupy_utils import xp


class TestGridVT(unittest.TestCase):

    def setUp(self):
        model = lambda x: xp.ones_like(x['a'])
        data = dict(a=xp.linspace(0, 1, 1000), vt=2)
        self.n_test = 100
        self.vt = vt.GridVT(model, data)

    def test_vt_correct(self):
        self.assertEqual(self.vt(dict()), 2)
