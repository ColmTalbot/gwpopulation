import unittest

import numpy as np
import pandas as pd
from bilby.core.prior import PriorDict, Uniform
from bilby.hyper.model import Model

import gwpopulation
from gwpopulation.hyperpe import HyperparameterLikelihood, RateLikelihood
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution

xp = np


class Likelihoods(unittest.TestCase):
    def setUp(self):
        gwpopulation.set_backend("numpy")
        self.params = dict(a=1, b=1, c=1)
        self.model = lambda dataset, a, b, c: dataset["a"]
        one_data = pd.DataFrame({key: xp.ones(500) for key in self.params})
        self.data = [one_data] * 5
        self.ln_evidences = [0] * 5
        self.selection_function = lambda args: 2.0
        self.conversion_function = lambda args: (args, ["bar"])

    def tearDown(self):
        pass

    def test_hpe_likelihood_requires_posteriors(self):
        with self.assertRaises(TypeError):
            _ = HyperparameterLikelihood(hyper_prior=self.model)

    def test_hpe_likelihood_requires_hyper_prior(self):
        with self.assertRaises(TypeError):
            _ = HyperparameterLikelihood(posteriors=self.data)

    def test_likelihood_pass_sampling_prior_raises_error(self):
        with self.assertRaises(TypeError):
            _ = HyperparameterLikelihood(
                posteriors=self.data,
                hyper_prior=self.model,
                sampling_prior=lambda dataset: 5,
            )

    def test_prior_in_posteriors(self):
        for frame in self.data:
            frame["prior"] = 1
        like = HyperparameterLikelihood(posteriors=self.data, hyper_prior=self.model)
        self.assertTrue(
            xp.array_equal(like.sampling_prior, xp.ones_like(like.data["a"]))
        )

    def test_not_passing_prior(self):
        like = HyperparameterLikelihood(posteriors=self.data, hyper_prior=self.model)
        self.assertEqual(like.sampling_prior, 1)

    def test_hpe_likelihood_set_evidences(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model, ln_evidences=self.ln_evidences
        )
        self.assertEqual(like.total_noise_evidence, 0)

    def test_hpe_likelihood_dont_set_evidences(self):
        like = HyperparameterLikelihood(posteriors=self.data, hyper_prior=self.model)
        self.assertTrue(xp.isnan(like.total_noise_evidence))

    def test_hpe_likelihood_set_conversion(self):
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            conversion_function=self.conversion_function,
        )
        self.assertEqual(like.conversion_function("foo"), ("foo", ["bar"]))

    def test_hpe_likelihood_set_selection(self):
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
        )
        self.assertEqual(like.selection_function("foo"), 2.0)

    def test_hpe_likelihood_set_max_samples(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model, max_samples=10
        )
        self.assertEqual(like.data["a"].shape, (5, 10))

    def test_hpe_likelihood_log_likelihood_ratio(self):
        like = HyperparameterLikelihood(posteriors=self.data, hyper_prior=self.model)
        like.parameters.update(self.params)
        self.assertEqual(like.log_likelihood_ratio(), 0.0)

    def test_hpe_likelihood_converts_nan_to_neginf(self):
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=lambda *args, **kwargs: (np.nan, 0),
        )
        like.parameters.update(self.params)
        self.assertEqual(like.log_likelihood_ratio(), np.nan_to_num(-np.inf))

    def test_hpe_likelihood_noise_likelihood_ratio(self):
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences,
        )
        like.parameters.update(self.params)
        self.assertEqual(like.noise_log_likelihood(), 0)

    def test_hpe_likelihood_log_likelihood_equal_ratio_zero_evidence(self):
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences,
        )
        like.parameters.update(self.params)
        self.assertEqual(like.log_likelihood_ratio(), like.log_likelihood())

    def test_hpe_likelihood_population_variance_too_large_returns_neginf(self):
        xp.random.seed(10)
        self.data[0]["a"] *= xp.random.uniform(0, 2, self.data[0]["a"].shape)
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model, maximum_uncertainty=1e-5
        )
        like.parameters.update(self.params)
        self.assertEqual(like.log_likelihood_ratio(), np.nan_to_num(-np.inf))

    def test_hpe_likelihood_selection_variance_too_large_returns_neginf(self):
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            maximum_uncertainty=0.1,
            selection_function=lambda *args, **kwargs: (1e-5, 1),
        )
        # like._get_selection_factor = lambda *args, **kwargs: (0, 1)
        like.parameters.update(self.params)
        self.assertEqual(like.log_likelihood_ratio(), np.nan_to_num(-np.inf))

    def test_hpe_likelihood_variance_small_enough_returns_expected(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model, maximum_uncertainty=0.1
        )
        like._get_selection_factor = lambda *args, **kwargs: (0, 0)
        like.parameters.update(self.params)
        self.assertEqual(like.log_likelihood_ratio(), 0.0)

    def test_hpe_likelihood_conversion_function_pops_parameters(self):
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            conversion_function=self.conversion_function,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences,
        )
        self.params["bar"] = None
        like.parameters.update(self.params)
        like.log_likelihood_ratio()
        self.assertFalse("bar" in like.parameters)

    def test_rate_likelihood_conversion_function_pops_parameters(self):
        like = RateLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            conversion_function=self.conversion_function,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences,
        )
        self.params["bar"] = None
        self.params["rate"] = 1
        like.parameters.update(self.params)
        like.log_likelihood_ratio()
        self.assertFalse("bar" in like.parameters)

    def test_rate_likelihood_requires_rate(self):
        like = RateLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences,
        )
        like.parameters.update(self.params)
        with self.assertRaises(KeyError):
            like.log_likelihood_ratio()

    def test_generate_extra_statistics(self):
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
            conversion_function=self.conversion_function,
            ln_evidences=self.ln_evidences,
        )
        self.params["bar"] = None
        new_params = like.generate_extra_statistics(sample=self.params.copy())
        expected = {
            "a": 1,
            "b": 1,
            "c": 1,
            "ln_bf_0": 0.0,
            "ln_bf_1": 0.0,
            "ln_bf_2": 0.0,
            "ln_bf_3": 0.0,
            "ln_bf_4": 0.0,
            "selection": 2.0,
            "selection_variance": 0.0,
            "var_0": 0.0,
            "var_1": 0.0,
            "var_2": 0.0,
            "var_3": 0.0,
            "var_4": 0.0,
            "variance": 0.0,
            "bar": None,
        }
        self.assertDictEqual(expected, new_params)

    def test_generate_rate_posterior_sample_returns_positive(self):
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences,
        )
        self.assertGreater(like.generate_rate_posterior_sample(), 0)

    def test_resampling_posteriors(self):
        priors = PriorDict(dict(a=Uniform(0, 2), b=Uniform(0, 2), c=Uniform(0, 2)))
        samples = priors.sample(100)
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences,
        )
        new_samples = like.posterior_predictive_resample(samples=samples)
        for key in new_samples:
            self.assertEqual(new_samples[key].shape, like.data[key].shape)

    def test_meta_data(self):
        model = Model([self.model, SinglePeakSmoothedMassDistribution()])
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences,
        )
        expected = dict(
            model=["<lambda>", "SinglePeakSmoothedMassDistribution"],
            data=dict(
                a=np.ones((5, 500)),
                b=np.ones((5, 500)),
                c=np.ones((5, 500)),
            ),
            n_events=5,
            sampling_prior=1,
            samples_per_posterior=500,
        )
        for key in like.meta_data:
            if key == "data":
                for key_2 in like.meta_data[key]:
                    self.assertTrue(
                        np.max(abs(like.meta_data[key][key_2] - expected[key][key_2]))
                        == 0
                    )
            else:
                self.assertEqual(like.meta_data[key], expected[key])
