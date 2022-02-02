import unittest

import numpy as np
from bilby.core.prior import PriorDict, Uniform

from gwpopulation.cupy_utils import trapz, xp
from gwpopulation.models import mass


class TestDoublePowerLaw(unittest.TestCase):
    def setUp(self):
        self.m1s = np.linspace(3, 100, 1000)
        self.qs = np.linspace(0.01, 1, 500)
        m1s_grid, qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.dataset = dict(mass_1=m1s_grid, mass_ratio=qs_grid)
        self.power_prior = PriorDict()
        self.power_prior["alpha_1"] = Uniform(minimum=-4, maximum=12)
        self.power_prior["alpha_2"] = Uniform(minimum=-4, maximum=12)
        self.power_prior["beta"] = Uniform(minimum=-4, maximum=12)
        self.power_prior["mmin"] = Uniform(minimum=3, maximum=10)
        self.power_prior["mmax"] = Uniform(minimum=40, maximum=100)
        self.power_prior["break_fraction"] = Uniform(minimum=40, maximum=100)
        self.n_test = 10

    def test_double_power_law_zero_below_mmin(self):
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            del parameters["beta"]
            p_m = mass.double_power_law_primary_mass(self.m1s, **parameters)
            self.assertEqual(xp.max(p_m[self.m1s <= parameters["mmin"]]), 0.0)

    def test_power_law_primary_mass_ratio_zero_above_mmax(self):
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            p_m = mass.double_power_law_primary_power_law_mass_ratio(
                self.dataset, **parameters
            )
            self.assertEqual(
                xp.max(p_m[self.dataset["mass_1"] >= parameters["mmax"]]), 0.0
            )


class TestPrimaryMassRatio(unittest.TestCase):
    def setUp(self):
        self.m1s = np.linspace(3, 100, 1000)
        self.qs = np.linspace(0.01, 1, 500)
        m1s_grid, qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.dataset = dict(mass_1=m1s_grid, mass_ratio=qs_grid)
        self.power_prior = PriorDict()
        self.power_prior["alpha"] = Uniform(minimum=-4, maximum=12)
        self.power_prior["beta"] = Uniform(minimum=-4, maximum=12)
        self.power_prior["mmin"] = Uniform(minimum=3, maximum=10)
        self.power_prior["mmax"] = Uniform(minimum=40, maximum=100)
        self.gauss_prior = PriorDict()
        self.gauss_prior["lam"] = Uniform(minimum=0, maximum=1)
        self.gauss_prior["mpp"] = Uniform(minimum=20, maximum=60)
        self.gauss_prior["sigpp"] = Uniform(minimum=0, maximum=10)
        self.n_test = 10

    def test_power_law_primary_mass_ratio_zero_below_mmin(self):
        m2s = self.dataset["mass_1"] * self.dataset["mass_ratio"]
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            p_m = mass.power_law_primary_mass_ratio(self.dataset, **parameters)
            self.assertEqual(xp.max(p_m[m2s <= parameters["mmin"]]), 0.0)

    def test_power_law_primary_mass_ratio_zero_above_mmax(self):
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            p_m = mass.power_law_primary_mass_ratio(self.dataset, **parameters)
            self.assertEqual(
                xp.max(p_m[self.dataset["mass_1"] >= parameters["mmax"]]), 0.0
            )

    def test_two_component_primary_mass_ratio_zero_below_mmin(self):
        m2s = self.dataset["mass_1"] * self.dataset["mass_ratio"]
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            parameters.update(self.gauss_prior.sample())
            p_m = mass.two_component_primary_mass_ratio(self.dataset, **parameters)
            self.assertEqual(xp.max(p_m[m2s <= parameters["mmin"]]), 0.0)


class TestPrimarySecondary(unittest.TestCase):
    def setUp(self):
        self.ms = np.linspace(3, 100, 1000)
        self.dm = self.ms[1] - self.ms[0]
        m1s_grid, m2s_grid = xp.meshgrid(self.ms, self.ms)
        self.dataset = dict(mass_1=m1s_grid, mass_2=m2s_grid)
        self.power_prior = PriorDict()
        self.power_prior["alpha"] = Uniform(minimum=-4, maximum=12)
        self.power_prior["beta"] = Uniform(minimum=-4, maximum=12)
        self.power_prior["mmin"] = Uniform(minimum=3, maximum=10)
        self.power_prior["mmax"] = Uniform(minimum=40, maximum=100)
        self.gauss_prior = PriorDict()
        self.gauss_prior["lam"] = Uniform(minimum=0, maximum=1)
        self.gauss_prior["mpp"] = Uniform(minimum=20, maximum=60)
        self.gauss_prior["sigpp"] = Uniform(minimum=0, maximum=10)
        self.n_test = 10

    def test_power_law_primary_secondary_zero_below_mmin(self):
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            p_m = mass.power_law_primary_secondary_independent(
                self.dataset, **parameters
            )
            self.assertEqual(
                xp.max(p_m[self.dataset["mass_2"] <= parameters["mmin"]]), 0.0
            )

    def test_power_law_primary_secondary_zero_above_mmax(self):
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            del parameters["beta"]
            p_m = mass.power_law_primary_secondary_identical(self.dataset, **parameters)
            self.assertEqual(
                xp.max(p_m[self.dataset["mass_1"] >= parameters["mmax"]]), 0.0
            )

    def test_two_component_primary_secondary_zero_below_mmin(self):
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            parameters.update(self.gauss_prior.sample())
            del parameters["beta"]
            p_m = mass.two_component_primary_secondary_identical(
                self.dataset, **parameters
            )
            self.assertEqual(
                xp.max(p_m[self.dataset["mass_2"] <= parameters["mmin"]]), 0.0
            )


class TestSmoothedMassDistribution(unittest.TestCase):
    def setUp(self):
        self.m1s = np.linspace(2, 100, 1000)
        self.qs = np.linspace(0.01, 1, 500)
        self.dm = self.m1s[1] - self.m1s[0]
        self.dq = self.qs[1] - self.qs[0]
        m1s_grid, qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.dataset = dict(mass_1=m1s_grid, mass_ratio=qs_grid)
        self.power_prior = PriorDict()
        self.power_prior["alpha"] = Uniform(minimum=-4, maximum=12)
        self.power_prior["beta"] = Uniform(minimum=-4, maximum=12)
        self.power_prior["mmin"] = Uniform(minimum=3, maximum=10)
        self.power_prior["mmax"] = Uniform(minimum=30, maximum=100)
        self.gauss_prior = PriorDict()
        self.gauss_prior["lam"] = Uniform(minimum=0, maximum=1)
        self.gauss_prior["mpp"] = Uniform(minimum=20, maximum=60)
        self.gauss_prior["sigpp"] = Uniform(minimum=0, maximum=10)
        self.double_gauss_prior = PriorDict()
        self.double_gauss_prior["lam"] = Uniform(minimum=0, maximum=1)
        self.double_gauss_prior["lam_1"] = Uniform(minimum=0, maximum=1)
        self.double_gauss_prior["mpp_1"] = Uniform(minimum=20, maximum=60)
        self.double_gauss_prior["mpp_2"] = Uniform(minimum=20, maximum=100)
        self.double_gauss_prior["sigpp_1"] = Uniform(minimum=0, maximum=10)
        self.double_gauss_prior["sigpp_2"] = Uniform(minimum=0, maximum=10)
        self.broken_power_prior = PriorDict()
        self.broken_power_prior["alpha_1"] = Uniform(minimum=-4, maximum=12)
        self.broken_power_prior["alpha_2"] = Uniform(minimum=-4, maximum=12)
        self.broken_power_prior["break_fraction"] = Uniform(minimum=0, maximum=1)
        self.broken_power_prior["beta"] = Uniform(minimum=-4, maximum=12)
        self.broken_power_prior["mmin"] = Uniform(minimum=3, maximum=10)
        self.broken_power_prior["mmax"] = Uniform(minimum=30, maximum=100)
        self.broken_power_peak_prior = PriorDict()
        self.broken_power_peak_prior["alpha_1"] = Uniform(minimum=-4, maximum=12)
        self.broken_power_peak_prior["alpha_2"] = Uniform(minimum=-4, maximum=12)
        self.broken_power_peak_prior["break_fraction"] = Uniform(minimum=0, maximum=1)
        self.broken_power_peak_prior["beta"] = Uniform(minimum=-4, maximum=12)
        self.broken_power_peak_prior["mmin"] = Uniform(minimum=3, maximum=10)
        self.broken_power_peak_prior["mmax"] = Uniform(minimum=30, maximum=100)
        self.broken_power_peak_prior["lam"] = Uniform(minimum=0, maximum=1)
        self.broken_power_peak_prior["mpp"] = Uniform(minimum=20, maximum=60)
        self.broken_power_peak_prior["sigpp"] = Uniform(minimum=0, maximum=10)
        self.smooth_prior = PriorDict()
        self.smooth_prior["delta_m"] = Uniform(minimum=0, maximum=10)
        self.n_test = 10

    def test_single_peak_delta_m_zero_matches_two_component_primary_mass_ratio(self):
        max_diffs = list()
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            parameters.update(self.gauss_prior.sample())
            p_m1 = mass.two_component_primary_mass_ratio(self.dataset, **parameters)
            parameters["delta_m"] = 0
            p_m2 = mass.SinglePeakSmoothedMassDistribution()(self.dataset, **parameters)
            max_diffs.append(_max_abs_difference(p_m1, p_m2))
        self.assertAlmostEqual(max(max_diffs), 0.0)

    def test_double_peak_delta_m_zero_matches_two_component_primary_mass_ratio(self):
        max_diffs = list()
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            parameters.update(self.double_gauss_prior.sample())
            del parameters["beta"]
            p_m1 = mass.three_component_single(
                mass=self.dataset["mass_1"], **parameters
            )
            parameters["delta_m"] = 0
            p_m2 = mass.MultiPeakSmoothedMassDistribution().p_m1(
                self.dataset, **parameters
            )
            max_diffs.append(_max_abs_difference(p_m1, p_m2))
        self.assertAlmostEqual(max(max_diffs), 0.0)

    def test_single_peak_normalised(self):
        norms = list()
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            parameters.update(self.gauss_prior.sample())
            parameters.update(self.smooth_prior.sample())
            p_m = mass.SinglePeakSmoothedMassDistribution()(self.dataset, **parameters)
            norms.append(trapz(trapz(p_m, self.m1s), self.qs))
        self.assertAlmostEqual(_max_abs_difference(norms, 1.0), 0.0, 2)

    def test_double_peak_normalised(self):
        norms = list()
        for ii in range(self.n_test):
            parameters = self.power_prior.sample()
            parameters.update(self.double_gauss_prior.sample())
            parameters.update(self.smooth_prior.sample())
            p_m = mass.MultiPeakSmoothedMassDistribution()(self.dataset, **parameters)
            norms.append(trapz(trapz(p_m, self.m1s), self.qs))
        self.assertAlmostEqual(_max_abs_difference(norms, 1.0), 0.0, 2)

    def test_broken_power_law_normalised(self):
        norms = list()
        for ii in range(self.n_test):
            parameters = self.broken_power_prior.sample()
            parameters.update(self.smooth_prior.sample())
            p_m = mass.BrokenPowerLawSmoothedMassDistribution()(
                self.dataset, **parameters
            )
            norms.append(trapz(trapz(p_m, self.m1s), self.qs))
        self.assertAlmostEqual(_max_abs_difference(norms, 1.0), 0.0, 2)

    def test_broken_power_law_peak_normalised(self):
        norms = list()
        for ii in range(self.n_test):
            parameters = self.broken_power_peak_prior.sample()
            parameters.update(self.smooth_prior.sample())
            p_m = mass.BrokenPowerLawPeakSmoothedMassDistribution()(
                self.dataset, **parameters
            )
            norms.append(trapz(trapz(p_m, self.m1s), self.qs))
        self.assertAlmostEqual(_max_abs_difference(norms, 1.0), 0.0, 2)

    def test_set_minimum_and_maximum(self):
        model = mass.SinglePeakSmoothedMassDistribution(mmin=5, mmax=150)
        parameters = self.gauss_prior.sample()
        parameters.update(self.power_prior.sample())
        parameters.update(self.smooth_prior.sample())
        parameters["mpp"] = 130
        parameters["sigpp"] = 1
        parameters["lam"] = 0.5
        parameters["mmin"] = 5
        self.assertEqual(
            model(
                dict(mass_1=8 * np.ones(5), mass_ratio=0.5 * np.ones(5)), **parameters
            )[0],
            0,
        )
        self.assertGreater(
            model(
                dict(mass_1=130 * np.ones(5), mass_ratio=0.9 * np.ones(5)), **parameters
            )[0],
            0,
        )

    def test_mmin_below_global_minimum_raises_error(self):
        model = mass.SinglePeakSmoothedMassDistribution(mmin=5, mmax=150)
        parameters = self.gauss_prior.sample()
        parameters.update(self.power_prior.sample())
        parameters.update(self.smooth_prior.sample())
        parameters["mmin"] = 2
        with self.assertRaises(ValueError):
            model(self.dataset, **parameters)

    def test_mmax_above_global_maximum_raises_error(self):
        model = mass.SinglePeakSmoothedMassDistribution(mmin=5, mmax=150)
        parameters = self.gauss_prior.sample()
        parameters.update(self.power_prior.sample())
        parameters.update(self.smooth_prior.sample())
        parameters["mmax"] = 200
        with self.assertRaises(ValueError):
            model(self.dataset, **parameters)


def _max_abs_difference(array, comparison):
    return float(xp.max(xp.abs(comparison - xp.asarray(array))))


if __name__ == "__main__":
    unittest.main()
