import unittest

from gwpopulation import vt
from gwpopulation.cupy_utils import xp


class TestGridVT(unittest.TestCase):
    def setUp(self):
        model = lambda dataset: xp.ones_like(dataset["a"])
        data = dict(a=xp.linspace(0, 1, 1000), vt=2)
        self.n_test = 100
        self.vt = vt.GridVT(model, data)

    def test_vt_correct(self):
        self.assertEqual(self.vt(dict()), 2)


class TestResamplingVT(unittest.TestCase):
    def setUp(self) -> None:
        model = lambda dataset: xp.exp(- (dataset["a"] - 0.5) ** 2 / 2) / (2 * xp.pi) ** 0.5
        data = dict(a=xp.linspace(0, 1, 1000), prior=xp.ones(1000), vt=2)
        self.vt = vt.ResamplingVT(data=data, model=model, n_events=0)

    def test_vt_correct(self):
        self.assertEqual(self.vt(dict()), 0.38289325179141254)

    def test_returns_inf_when_n_effective_too_small(self):
        self.vt.n_events = xp.inf
        self.assertEqual(self.vt(dict()), xp.inf)


