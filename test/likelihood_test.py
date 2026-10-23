import unittest

import numpy as np
import pandas as pd
from bilby.core.prior import PriorDict, Uniform
from bilby.hyper.model import Model

import gwpopulation
from gwpopulation.hyperpe import (
    HyperparameterLikelihood,
    NullHyperparameterLikelihood,
    RateLikelihood,
)
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution

xp = np


class Likelihoods(unittest.TestCase):
    def setUp(self):
        gwpopulation.set_backend("numpy")
        self.params = dict(a=1, b=1, c=1)
        self.model = lambda dataset, a, b, c: dataset["a"]
        one_data = pd.DataFrame({key: xp.ones(500) for key in self.params})
        self.data = [one_data] * 5
        self.sample_lengths = [100, 123, 459, 43, 233]
        self.unequal_data = [one_data[:ns] for ns in self.sample_lengths]
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
            posteriors=self.data,
            hyper_prior=self.model,
            max_samples=10,
            require_equal_samples=True,
        )
        self.assertEqual(like.data["a"].shape, (5, 10))

    def test_hpe_likelihood_unequal_samples(self):
        like = HyperparameterLikelihood(
            posteriors=self.unequal_data,
            hyper_prior=self.model,
            require_equal_samples=False,
        )
        for value in like.data.values():
            self.assertEqual(value.shape, (sum(self.sample_lengths),))

    def test_hpe_likelihood_log_likelihood_ratio(self):
        like = HyperparameterLikelihood(posteriors=self.data, hyper_prior=self.model)
        self.assertEqual(like.log_likelihood_ratio(self.params), 0.0)

    def test_hpe_likelihood_converts_nan_to_neginf(self):
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=lambda *args, **kwargs: (np.nan, 0),
        )
        self.assertEqual(like.log_likelihood_ratio(self.params), np.nan_to_num(-np.inf))

    def test_hpe_likelihood_noise_likelihood_ratio(self):
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences,
        )
        self.assertEqual(like.noise_log_likelihood(), 0)

    def test_hpe_likelihood_log_likelihood_equal_ratio_zero_evidence(self):
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences,
        )
        self.assertEqual(
            like.log_likelihood_ratio(self.params), like.log_likelihood(self.params)
        )

    def test_hpe_likelihood_population_variance_too_large_returns_neginf(self):
        xp.random.seed(10)
        self.data[0]["a"] *= xp.random.uniform(0, 2, self.data[0]["a"].shape)
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model, maximum_uncertainty=1e-5
        )
        self.assertEqual(like.log_likelihood_ratio(self.params), np.nan_to_num(-np.inf))

    def test_hpe_likelihood_selection_variance_too_large_returns_neginf(self):
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            maximum_uncertainty=0.1,
            selection_function=lambda *args, **kwargs: (1e-5, 1),
        )
        self.assertEqual(like.log_likelihood_ratio(self.params), np.nan_to_num(-np.inf))

    def test_hpe_likelihood_variance_small_enough_returns_expected(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model, maximum_uncertainty=0.1
        )
        like._get_selection_factor = lambda *args, **kwargs: (0, 0)
        self.assertEqual(like.log_likelihood_ratio(self.params), 0.0)

    def test_rate_likelihood_requires_rate(self):
        like = RateLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences,
        )
        with self.assertRaises(KeyError):
            like.log_likelihood_ratio(self.params)

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
        self.assertGreater(like.generate_rate_posterior_sample(self.params), 0)

    def test_resampling_posteriors(self):
        priors = PriorDict(dict(a=Uniform(0, 2), b=Uniform(0, 2), c=Uniform(0, 2)))
        samples = priors.sample(100)
        like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences,
            require_equal_samples=True,
        )
        new_samples = like.posterior_predictive_resample(samples=samples)
        for key in new_samples:
            self.assertEqual(new_samples[key].shape, like.data[key].shape)

    def test_resampling_unequal_posteriors(self):
        priors = PriorDict(dict(a=Uniform(0, 2), b=Uniform(0, 2), c=Uniform(0, 2)))
        samples = priors.sample(100)
        like = HyperparameterLikelihood(
            posteriors=self.unequal_data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences,
            require_equal_samples=False,
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
            require_equal_samples=True,
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


class NullHyperparameterLikelihoodTests(unittest.TestCase):
    def setUp(self):
        gwpopulation.set_backend("numpy")
        self.params = dict(a=1, b=1, c=1)
        self.model = lambda dataset, a, b, c: dataset["a"]
        one_data = pd.DataFrame({key: xp.ones(500) for key in self.params})
        self.data = [one_data] * 5
        self.ln_evidences = [0] * 5
        self.selection_function = lambda args: 2.0

    def tearDown(self):
        pass

    def test_null_likelihood_variance_matches_hyperparameter_likelihood(self):
        """Test that NullHyperparameterLikelihood variance matches HyperparameterLikelihood"""
        # Create both likelihoods with the same parameters
        null_like = NullHyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
        )
        hyper_like = HyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
        )

        # Get the ln_likelihood and variance from both
        null_ln_l, null_variance = null_like.ln_likelihood_and_variance(self.params)
        hyper_ln_l, hyper_variance = hyper_like.ln_likelihood_and_variance(self.params)

        # Check that variances match
        self.assertEqual(null_variance, hyper_variance)

    def test_null_likelihood_returns_zero(self):
        """Test that NullHyperparameterLikelihood returns zero for the likelihood"""
        null_like = NullHyperparameterLikelihood(
            posteriors=self.data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
        )

        # Get the ln_likelihood
        null_ln_l, null_variance = null_like.ln_likelihood_and_variance(self.params)

        # Check that likelihood is zero
        self.assertEqual(null_ln_l, 0.0)

    def test_null_likelihood_returns_neginf_when_variance_exceeds_threshold(self):
        """Test that NullHyperparameterLikelihood returns -inf when variance > threshold"""
        # Create data with some variation to generate non-zero variance
        xp.random.seed(10)
        varied_data = [self.data[0].copy() for _ in range(5)]
        varied_data[0]["a"] *= xp.random.uniform(0, 2, varied_data[0]["a"].shape)

        # Create likelihood with small maximum_uncertainty threshold
        null_like = NullHyperparameterLikelihood(
            posteriors=varied_data,
            hyper_prior=self.model,
            selection_function=self.selection_function,
            maximum_uncertainty=1e-5,
        )

        # Get the log likelihood ratio (which incorporates the variance check)
        log_lr = null_like.log_likelihood_ratio(self.params)

        # Check that it returns -inf
        self.assertEqual(log_lr, np.nan_to_num(-np.inf))


class LogSpaceLikelihood(unittest.TestCase):
    """The per-event Monte Carlo integrals are computed from log weights."""

    def setUp(self):
        gwpopulation.set_backend("numpy")
        rng = np.random.default_rng(3)
        self.sample_lengths = [100, 123, 459, 43, 233]
        self.posteriors = [
            pd.DataFrame(dict(a=rng.uniform(0.1, 1, ns), prior=rng.uniform(0.5, 2, ns)))
            for ns in self.sample_lengths
        ]

    @staticmethod
    def model(dataset, b):
        return dataset["a"] ** b

    def expected(self, posteriors, b):
        """ln BF and variance from the linear weights."""
        ln_bfs, variances = [], []
        for post in posteriors:
            weights = post["a"].values ** b / post["prior"].values
            mean = np.mean(weights)
            ln_bfs.append(np.log(mean))
            variances.append((np.mean(weights**2) - mean**2) / (len(weights) * mean**2))
        return np.array(ln_bfs), np.array(variances)

    def check(self, likelihood, posteriors):
        ln_bfs, variances = likelihood._compute_per_event_ln_bayes_factors(dict(b=2))
        expected_ln_bfs, expected_variances = self.expected(posteriors, 2)
        np.testing.assert_allclose(ln_bfs, expected_ln_bfs, rtol=1e-10)
        np.testing.assert_allclose(variances, expected_variances, rtol=1e-8)

    def test_unequal_samples_match_linear(self):
        likelihood = HyperparameterLikelihood(self.posteriors, self.model)
        self.check(likelihood, self.posteriors)

    def test_equal_samples_match_linear(self):
        posteriors = [post[:40] for post in self.posteriors]
        likelihood = HyperparameterLikelihood(
            posteriors, self.model, require_equal_samples=True
        )
        self.check(likelihood, posteriors)

    def test_ln_prior_matches_prior(self):
        posteriors = [
            pd.DataFrame(dict(a=post["a"], ln_prior=np.log(post["prior"])))
            for post in self.posteriors
        ]
        likelihood = HyperparameterLikelihood(posteriors, self.model)
        self.check(likelihood, self.posteriors)
        self.assertNotIn("ln_prior", likelihood.data)

    def test_uses_ln_prob(self):
        """Weights of exp(-1000) underflow; their log does not."""

        def model(dataset, b):
            return np.exp(model.ln_prob(dataset, b))

        model.ln_prob = lambda dataset, b: -1000 + b * np.log(dataset["a"])
        likelihood = HyperparameterLikelihood(self.posteriors, model)
        ln_bfs = likelihood._compute_per_event_ln_bayes_factors(
            dict(b=2), return_uncertainty=False
        )
        expected_ln_bfs, _ = self.expected(self.posteriors, 2)
        np.testing.assert_allclose(ln_bfs, expected_ln_bfs - 1000, rtol=1e-10)

    def test_bilby_model_uses_ln_prob(self):
        """A bilby Model has no ln_prob; it is converted so that its
        components' log probabilities are used."""

        def model(dataset, b):
            return np.exp(model.ln_prob(dataset, b))

        model.ln_prob = lambda dataset, b: -1000 + b * np.log(dataset["a"])
        likelihood = HyperparameterLikelihood(self.posteriors, Model([model]))
        self.assertIsInstance(likelihood.hyper_prior, gwpopulation.hyperpe.Model)
        ln_bfs = likelihood._compute_per_event_ln_bayes_factors(
            dict(b=2), return_uncertainty=False
        )
        expected_ln_bfs, _ = self.expected(self.posteriors, 2)
        np.testing.assert_allclose(ln_bfs, expected_ln_bfs - 1000, rtol=1e-10)

    def test_events_on_different_scales(self):
        """One event's weights are e^500 times another's; each event's mean
        is unaffected by the others."""
        posteriors = [post.copy() for post in self.posteriors]
        posteriors[0]["prior"] *= np.exp(-500)
        likelihood = HyperparameterLikelihood(posteriors, self.model)
        ln_bfs, _ = likelihood._compute_per_event_ln_bayes_factors(dict(b=2))
        expected_ln_bfs, _ = self.expected(self.posteriors, 2)
        expected_ln_bfs[0] += 500
        np.testing.assert_allclose(ln_bfs, expected_ln_bfs, rtol=1e-10)
