from __future__ import division

import unittest

import numpy as np
import pandas as pd

try:
    import cupy as xp
except ImportError:
    xp = np

from gwpopulation.hyperpe import HyperparameterLikelihood, RateLikelihood


class Likelihoods(unittest.TestCase):

    def setUp(self):
        self.params = dict(a=1, b=1, c=1)
        self.model = lambda x, a, b, c: x['a']
        self.sampling_prior = lambda x: 1
        one_data = pd.DataFrame({key: xp.ones(500) for key in self.params})
        self.data = [one_data] * 5
        self.ln_evidences = [0] * 5
        self.selection_function = lambda args: 2
        self.conversion_function = lambda args: (args, ['bar'])

    def tearDown(self):
        pass

    def test_hpe_likelihood_requires_posteriors(self):
        with self.assertRaises(TypeError):
            _ = HyperparameterLikelihood(hyper_prior=self.model)

    def test_hpe_likelihood_requires_hyper_prior(self):
        with self.assertRaises(TypeError):
            _ = HyperparameterLikelihood(posteriors=self.data)

    def test_likelihood_pass_sampling_prior_works(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model,
            sampling_prior=lambda x: 5)
        self.assertEqual(like.sampling_prior, 5)

    def test_prior_in_posteriors(self):
        for frame in self.data:
            frame['prior'] = 1
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model)
        self.assertTrue(
            xp.array_equal(like.sampling_prior, xp.ones_like(like.data['a'])))

    def test_not_passing_prior(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model)
        self.assertEqual(like.sampling_prior, 1)

    def test_hpe_likelihood_set_evidences(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model,
            ln_evidences=self.ln_evidences)
        self.assertEqual(like.total_noise_evidence, 0)

    def test_hpe_likelihood_dont_set_evidences(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model)
        self.assertTrue(xp.isnan(like.total_noise_evidence))

    def test_hpe_likelihood_set_conversion(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model,
            conversion_function=self.conversion_function)
        self.assertEqual(like.conversion_function('foo'), ('foo', ['bar']))

    def test_hpe_likelihood_set_selection(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model,
            selection_function=self.selection_function)
        self.assertEqual(like.selection_function('foo'), 2.0)

    def test_hpe_likelihood_set_max_samples(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model, max_samples=10)
        self.assertEqual(like.data['a'].shape, (5, 10))

    def test_hpe_likelihood_log_likelihood_ratio(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model)
        like.parameters.update(self.params)
        self.assertEqual(like.log_likelihood_ratio(), 0.0)

    def test_hpe_likelihood_noise_likelihood_ratio(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences)
        like.parameters.update(self.params)
        self.assertEqual(like.noise_log_likelihood(), 0)

    def test_hpe_likelihood_log_likelihood_equal_ratio_zero_evidence(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences)
        like.parameters.update(self.params)
        self.assertEqual(like.log_likelihood_ratio(), like.log_likelihood())

    def test_hpe_likelihood_conversion_function_pops_parameters(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model,
            conversion_function=self.conversion_function,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences)
        self.params['bar'] = None
        like.parameters.update(self.params)
        like.log_likelihood_ratio()
        self.assertFalse('bar' in like.parameters)

    def test_rate_likelihood_conversion_function_pops_parameters(self):
        like = RateLikelihood(
            posteriors=self.data, hyper_prior=self.model,
            conversion_function=self.conversion_function,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences)
        self.params['bar'] = None
        self.params['rate'] = 1
        like.parameters.update(self.params)
        like.log_likelihood_ratio()
        self.assertFalse('bar' in like.parameters)

    def test_rate_likelihood_requires_rate(self):
        like = RateLikelihood(
            posteriors=self.data, hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences)
        like.parameters.update(self.params)
        with self.assertRaises(KeyError):
            like.log_likelihood_ratio()

    def test_generate_extra_statistics(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model,
            selection_function=self.selection_function,
            conversion_function=self.conversion_function,
            ln_evidences=self.ln_evidences)
        self.params['bar'] = None
        new_params = like.generate_extra_statistics(sample=self.params.copy())
        expected = {
            'a': 1, 'b': 1, 'c': 1,
            'ln_bf_0': 6.214608098422191, 'ln_bf_1': 6.214608098422191,
            'ln_bf_2': 6.214608098422191, 'ln_bf_3': 6.214608098422191,
            'ln_bf_4': 6.214608098422191, 'selection': 2.0, 'bar': None
        }
        self.assertDictEqual(expected, new_params)

    def test_generate_rate_posterior_sample_raises_error(self):
        like = HyperparameterLikelihood(
            posteriors=self.data, hyper_prior=self.model,
            selection_function=self.selection_function,
            ln_evidences=self.ln_evidences)
        with self.assertRaises(NotImplementedError):
            like.generate_rate_posterior_sample()
