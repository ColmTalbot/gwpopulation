import unittest

from bilby.hyper.model import Model

from gwpopulation import vt
from gwpopulation.cupy_utils import xp
from gwpopulation.models.redshift import PowerLawRedshift, total_four_volume


def dummy_function(dataset, alpha):
    return 1


class TestBase(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_initialize_with_function(self):
        evaluator = vt._BaseVT(model=dummy_function, data=dict())
        self.assertTrue(evaluator.model.models == [dummy_function])

    def test_initialize_with_list_of_functions(self):
        evaluator = vt._BaseVT(model=[dummy_function, dummy_function], data=dict())
        self.assertTrue(evaluator.model.models == [dummy_function, dummy_function])

    def test_initialize_with_bilby_model(self):
        model = Model([dummy_function, dummy_function])
        evaluator = vt._BaseVT(model=model, data=dict())
        self.assertTrue(evaluator.model.models == [dummy_function, dummy_function])

    def test_base_cannot_be_called(self):
        model = Model([dummy_function, dummy_function])
        evaluator = vt._BaseVT(model=model, data=dict())
        with self.assertRaises(NotImplementedError):
            evaluator()


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
        model = (
            lambda dataset: xp.exp(-((dataset["a"] - 0.5) ** 2) / 2)
            / (2 * xp.pi) ** 0.5
        )
        self.data = dict(a=xp.linspace(0, 1, 1000), prior=xp.ones(1000), vt=2)
        self.vt = vt.ResamplingVT(data=self.data, model=model, n_events=0)

    def test_vt_correct(self):
        self.assertEqual(self.vt(dict()), 0.38289325179141254)

    def test_returns_inf_when_n_effective_too_small(self):
        self.vt.n_events = xp.inf
        self.assertEqual(self.vt(dict()), xp.inf)

    def test_observed_volume_with_no_redshift_model(self):
        self.assertEqual(
            self.vt.surveyed_hypervolume(dict()),
            total_four_volume(lamb=0, analysis_time=1),
        )

    def test_observed_volume_with_redshift_model(self):
        model = PowerLawRedshift()
        self.vt = vt.ResamplingVT(data=self.data, model=model, n_events=0)
        self.assertAlmostEqual(
            self.vt.surveyed_hypervolume(dict(lamb=4)),
            total_four_volume(lamb=4, analysis_time=1),
            4,
        )
