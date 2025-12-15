"""Tests for gwpopulation.experimental.cosmo_models module."""

import numpy as np
import pytest
from wcosmo.utils import disable_units

from gwpopulation.experimental.cosmo_models import CosmoMixin, CosmoModel


class TestCosmoMixin:
    """Test suite for CosmoMixin class."""

    def test_init_planck15(self):
        """Test initialization with Planck15 (fixed cosmology)."""
        mixin = CosmoMixin(cosmo_model="Planck15")
        assert mixin.cosmo_model == "Planck15"
        assert mixin.cosmology_names == []

    def test_init_flatwcdm(self):
        """Test initialization with FlatwCDM (parametrized cosmology)."""
        mixin = CosmoMixin(cosmo_model="FlatwCDM")
        assert mixin.cosmo_model == "FlatwCDM"
        assert mixin.cosmology_names == ["H0", "Om0", "w0"]

    def test_init_flatlambdacdm(self):
        """Test initialization with FlatLambdaCDM (parametrized cosmology)."""
        mixin = CosmoMixin(cosmo_model="FlatLambdaCDM")
        assert mixin.cosmo_model == "FlatLambdaCDM"
        assert mixin.cosmology_names == ["H0", "Om0"]

    def test_cosmology_variables_flatwcdm(self):
        """Test extracting cosmology variables for FlatwCDM."""
        mixin = CosmoMixin(cosmo_model="FlatwCDM")
        parameters = {"H0": 70, "Om0": 0.3, "w0": -1.0, "other_param": 5}
        cosmo_vars = mixin.cosmology_variables(parameters)
        assert cosmo_vars == {"H0": 70, "Om0": 0.3, "w0": -1.0}
        assert "other_param" not in cosmo_vars

    def test_cosmology_variables_flatlambdacdm(self):
        """Test extracting cosmology variables for FlatLambdaCDM."""
        mixin = CosmoMixin(cosmo_model="FlatLambdaCDM")
        parameters = {"H0": 67.4, "Om0": 0.315, "other_param": 5}
        cosmo_vars = mixin.cosmology_variables(parameters)
        assert cosmo_vars == {"H0": 67.4, "Om0": 0.315}
        assert "other_param" not in cosmo_vars

    def test_cosmology_planck15(self):
        """Test cosmology method with Planck15 (no parameters needed)."""
        disable_units()
        mixin = CosmoMixin(cosmo_model="Planck15")
        parameters = {}
        cosmo = mixin.cosmology(parameters)
        # For Planck15, should return the same instance
        assert cosmo is not None
        # Verify it's a valid cosmology object with expected properties
        assert hasattr(cosmo, "luminosity_distance")
        assert hasattr(cosmo, "dDLdz")

    def test_cosmology_flatwcdm(self):
        """Test cosmology method with FlatwCDM (parametrized)."""
        disable_units()
        mixin = CosmoMixin(cosmo_model="FlatwCDM")
        parameters = {"H0": 70, "Om0": 0.3, "w0": -1.0}
        cosmo = mixin.cosmology(parameters)
        assert cosmo is not None
        assert hasattr(cosmo, "luminosity_distance")
        assert hasattr(cosmo, "dDLdz")

    def test_cosmology_flatlambdacdm(self):
        """Test cosmology method with FlatLambdaCDM (parametrized)."""
        disable_units()
        mixin = CosmoMixin(cosmo_model="FlatLambdaCDM")
        parameters = {"H0": 67.4, "Om0": 0.315}
        cosmo = mixin.cosmology(parameters)
        assert cosmo is not None
        assert hasattr(cosmo, "luminosity_distance")
        assert hasattr(cosmo, "dDLdz")

    def test_detector_frame_to_source_frame_with_redshift(self):
        """Test detector_frame_to_source_frame when redshift is provided."""
        disable_units()
        mixin = CosmoMixin(cosmo_model="Planck15")
        data = {
            "redshift": np.array([0.1, 0.2, 0.3]),
            "mass_1_detector": np.array([30.0, 40.0, 50.0]),
            "other_param": np.array([1.0, 2.0, 3.0]),
        }
        parameters = {}

        samples, jacobian = mixin.detector_frame_to_source_frame(data, **parameters)

        # Check that redshift is preserved
        np.testing.assert_array_equal(samples["redshift"], data["redshift"])

        # Check that detector frame mass is converted to source frame
        expected_mass_1 = data["mass_1_detector"] / (1 + data["redshift"])
        np.testing.assert_allclose(samples["mass_1"], expected_mass_1)

        # Check that other parameters are preserved
        np.testing.assert_array_equal(samples["other_param"], data["other_param"])

        # Check jacobian for detector frame quantities
        expected_jacobian = 1 + data["redshift"]
        np.testing.assert_allclose(jacobian, expected_jacobian)

    def test_detector_frame_to_source_frame_with_luminosity_distance_planck15(self):
        """Test detector_frame_to_source_frame with luminosity distance (Planck15)."""
        disable_units()
        mixin = CosmoMixin(cosmo_model="Planck15")
        # Use realistic luminosity distances
        data = {
            "luminosity_distance": np.array([400.0, 800.0, 1200.0]),
        }
        parameters = {}

        samples, jacobian = mixin.detector_frame_to_source_frame(data, **parameters)

        # Check that redshift is computed
        assert "redshift" in samples
        assert len(samples["redshift"]) == 3
        # Redshift should be positive for positive luminosity distances
        assert np.all(samples["redshift"] > 0)

        # Check that jacobian is computed
        assert len(jacobian) == 3
        assert np.all(jacobian > 0)

    def test_detector_frame_to_source_frame_with_luminosity_distance_flatwcdm(self):
        """Test detector_frame_to_source_frame with FlatwCDM and luminosity distance."""
        disable_units()
        mixin = CosmoMixin(cosmo_model="FlatwCDM")
        data = {
            "luminosity_distance": np.array([400.0, 800.0, 1200.0]),
            "mass_1_detector": np.array([30.0, 40.0, 50.0]),
        }
        # This is the critical test: parameters must be passed through
        parameters = {"H0": 70, "Om0": 0.3, "w0": -1.0}

        samples, jacobian = mixin.detector_frame_to_source_frame(data, **parameters)

        # Check that redshift is computed
        assert "redshift" in samples
        assert len(samples["redshift"]) == 3
        assert np.all(samples["redshift"] > 0)

        # Check that detector frame mass is converted
        expected_mass_1 = data["mass_1_detector"] / (1 + samples["redshift"])
        np.testing.assert_allclose(samples["mass_1"], expected_mass_1, rtol=1e-10)

        # Check that jacobian includes both dDLdz and (1+z) terms
        assert len(jacobian) == 3
        assert np.all(jacobian > 0)

    def test_detector_frame_to_source_frame_with_luminosity_distance_flatlambdacdm(
        self,
    ):
        """Test detector_frame_to_source_frame with FlatLambdaCDM and luminosity distance."""
        disable_units()
        mixin = CosmoMixin(cosmo_model="FlatLambdaCDM")
        data = {
            "luminosity_distance": np.array([400.0, 800.0, 1200.0]),
        }
        # This is the critical test: parameters must be passed through
        parameters = {"H0": 67.4, "Om0": 0.315}

        samples, jacobian = mixin.detector_frame_to_source_frame(data, **parameters)

        # Check that redshift is computed
        assert "redshift" in samples
        assert len(samples["redshift"]) == 3
        assert np.all(samples["redshift"] > 0)

        # Check that jacobian is computed
        assert len(jacobian) == 3
        assert np.all(jacobian > 0)

    def test_detector_frame_to_source_frame_missing_redshift_and_luminosity_distance(
        self,
    ):
        """Test that ValueError is raised when neither redshift nor luminosity distance is provided."""
        disable_units()
        mixin = CosmoMixin(cosmo_model="Planck15")
        data = {"mass_1_detector": np.array([30.0, 40.0, 50.0])}
        parameters = {}

        with pytest.raises(ValueError):
            mixin.detector_frame_to_source_frame(data, **parameters)

    def test_detector_frame_to_source_frame_preserves_non_detector_keys(self):
        """Test that non-detector frame keys are preserved correctly."""
        disable_units()
        mixin = CosmoMixin(cosmo_model="Planck15")
        data = {
            "redshift": np.array([0.1, 0.2, 0.3]),
            "mass_1_detector": np.array([30.0, 40.0, 50.0]),
            "mass_2_detector": np.array([20.0, 25.0, 30.0]),
            "spin_1": np.array([0.1, 0.2, 0.3]),
            "luminosity_distance": np.array([400.0, 800.0, 1200.0]),
        }
        parameters = {}

        samples, jacobian = mixin.detector_frame_to_source_frame(data, **parameters)

        # Check that redshift is preserved
        np.testing.assert_array_equal(samples["redshift"], data["redshift"])

        # Check that detector frame quantities are converted
        assert "mass_1" in samples
        assert "mass_2" in samples

        # Check that non-detector quantities are preserved
        np.testing.assert_array_equal(samples["spin_1"], data["spin_1"])

        # Check that luminosity_distance is not in samples (it's not copied)
        assert "luminosity_distance" not in samples

    def test_detector_frame_to_source_frame_multiple_detector_quantities(self):
        """Test conversion with multiple detector frame quantities."""
        disable_units()
        mixin = CosmoMixin(cosmo_model="Planck15")
        data = {
            "redshift": np.array([0.1, 0.2, 0.3]),
            "mass_1_detector": np.array([30.0, 40.0, 50.0]),
            "mass_2_detector": np.array([20.0, 25.0, 30.0]),
            "chirp_mass_detector": np.array([25.0, 30.0, 35.0]),
        }
        parameters = {}

        samples, jacobian = mixin.detector_frame_to_source_frame(data, **parameters)

        # Check all detector frame quantities are converted
        for key in ["mass_1", "mass_2", "chirp_mass"]:
            assert key in samples
            expected = data[key + "_detector"] / (1 + data["redshift"])
            np.testing.assert_allclose(samples[key], expected)

        # Jacobian should be (1+z)^3 for three detector frame quantities
        expected_jacobian = (1 + data["redshift"]) ** 3
        np.testing.assert_allclose(jacobian, expected_jacobian)

    def test_detector_frame_to_source_frame_parameters_different_cosmologies(self):
        """Test that different parameters lead to different redshifts for same luminosity distance."""
        disable_units()
        mixin = CosmoMixin(cosmo_model="FlatLambdaCDM")
        data = {
            "luminosity_distance": np.array([1000.0]),
        }

        # Test with different H0 values
        parameters1 = {"H0": 60.0, "Om0": 0.3}
        parameters2 = {"H0": 80.0, "Om0": 0.3}

        samples1, _ = mixin.detector_frame_to_source_frame(data, **parameters1)
        samples2, _ = mixin.detector_frame_to_source_frame(data, **parameters2)

        # Different H0 should lead to different redshifts
        assert samples1["redshift"][0] != samples2["redshift"][0]


class TestCosmoModel:
    """Test suite for CosmoModel class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        disable_units()
        model = CosmoModel(model_functions=[])
        assert model.cosmo_model == "Planck15"
        assert model.cosmology_names == []
        assert model.models == []

    def test_init_with_cosmo_model(self):
        """Test initialization with custom cosmology model."""
        disable_units()
        model = CosmoModel(model_functions=[], cosmo_model="FlatwCDM")
        assert model.cosmo_model == "FlatwCDM"
        assert model.cosmology_names == ["H0", "Om0", "w0"]

    def test_init_with_model_functions(self):
        """Test initialization with model functions."""
        disable_units()

        def simple_model(dataset, **parameters):
            """Simple test model that returns constant probability."""
            return np.ones(len(dataset.get("mass_1", [1])))

        model = CosmoModel(model_functions=[simple_model])
        assert len(model.models) == 1

    def test_prob_with_planck15_and_redshift(self):
        """Test prob method with Planck15 cosmology and redshift provided."""
        disable_units()

        def simple_model(dataset, **parameters):
            """Simple model that depends on mass_1."""
            return np.ones(len(dataset["mass_1"]))

        model = CosmoModel(model_functions=[simple_model], cosmo_model="Planck15")

        data = {
            "redshift": np.array([0.1, 0.2, 0.3]),
            "mass_1_detector": np.array([30.0, 40.0, 50.0]),
        }
        parameters = {}

        prob = model.prob(data, **parameters)

        # Check that probability is computed and jacobian is applied
        assert len(prob) == 3
        assert np.all(prob > 0)

        # Expected jacobian is (1 + z) for one detector frame quantity
        expected_jacobian = 1 + data["redshift"]
        # Probability should be 1 / jacobian since simple_model returns ones
        np.testing.assert_allclose(prob, 1.0 / expected_jacobian)

    def test_prob_with_planck15_and_luminosity_distance(self):
        """Test prob method with Planck15 cosmology and luminosity distance."""
        disable_units()

        def simple_model(dataset, **parameters):
            """Simple model that depends on mass_1."""
            return np.ones(len(dataset["mass_1"]))

        model = CosmoModel(model_functions=[simple_model], cosmo_model="Planck15")

        data = {
            "luminosity_distance": np.array([400.0, 800.0, 1200.0]),
            "mass_1_detector": np.array([30.0, 40.0, 50.0]),
        }
        parameters = {}

        prob = model.prob(data, **parameters)

        # Check that probability is computed
        assert len(prob) == 3
        assert np.all(prob > 0)

    def test_prob_with_flatwcdm_passes_kwargs(self):
        """Test that prob method passes kwargs to detector_frame_to_source_frame for FlatwCDM."""
        disable_units()

        def simple_model(dataset, **parameters):
            """Simple model that returns ones."""
            return np.ones(len(dataset["mass_1"]))

        model = CosmoModel(model_functions=[simple_model], cosmo_model="FlatwCDM")

        data = {
            "luminosity_distance": np.array([1000.0]),
            "mass_1_detector": np.array([30.0]),
        }
        # These parameters MUST be passed through for FlatwCDM to work
        parameters = {"H0": 70.0, "Om0": 0.3, "w0": -1.0}

        # This should NOT raise an error - the bug would cause it to fail
        # because kwargs were not passed to detector_frame_to_source_frame
        prob = model.prob(data, **parameters)

        assert len(prob) == 1
        assert prob[0] > 0

    def test_prob_with_flatlambdacdm_passes_kwargs(self):
        """Test that prob method passes kwargs to detector_frame_to_source_frame for FlatLambdaCDM."""
        disable_units()

        def simple_model(dataset, **parameters):
            """Simple model that returns ones."""
            return np.ones(len(dataset["mass_1"]))

        model = CosmoModel(model_functions=[simple_model], cosmo_model="FlatLambdaCDM")

        data = {
            "luminosity_distance": np.array([1000.0]),
            "mass_1_detector": np.array([30.0]),
        }
        # These parameters MUST be passed through for FlatLambdaCDM to work
        parameters = {"H0": 67.4, "Om0": 0.315}

        # This should NOT raise an error
        prob = model.prob(data, **parameters)

        assert len(prob) == 1
        assert prob[0] > 0

    def test_prob_with_flatwcdm_different_parameters(self):
        """Test that different cosmological parameters lead to different probabilities."""
        disable_units()

        def simple_model(dataset, **parameters):
            """Simple model that returns ones."""
            return np.ones(len(dataset["mass_1"]))

        model = CosmoModel(model_functions=[simple_model], cosmo_model="FlatwCDM")

        data = {
            "luminosity_distance": np.array([1000.0]),
            "mass_1_detector": np.array([30.0]),
        }

        # Test with different H0 values
        parameters1 = {"H0": 60.0, "Om0": 0.3, "w0": -1.0}
        parameters2 = {"H0": 80.0, "Om0": 0.3, "w0": -1.0}

        prob1 = model.prob(data, **parameters1)
        prob2 = model.prob(data, **parameters2)

        # Different cosmological parameters should lead to different probabilities
        assert prob1[0] != prob2[0]

    def test_prob_applies_jacobian_correctly(self):
        """Test that the jacobian is applied correctly for multiple detector frame quantities."""
        disable_units()

        def simple_model(dataset, **parameters):
            """Simple model that returns ones."""
            return np.ones(len(dataset["mass_1"]))

        model = CosmoModel(model_functions=[simple_model], cosmo_model="Planck15")

        data = {
            "redshift": np.array([0.1, 0.2, 0.3]),
            "mass_1_detector": np.array([30.0, 40.0, 50.0]),
            "mass_2_detector": np.array([20.0, 25.0, 30.0]),
        }
        parameters = {}

        prob = model.prob(data, **parameters)

        # Expected jacobian is (1 + z)^2 for two detector frame quantities
        expected_jacobian = (1 + data["redshift"]) ** 2
        # Probability should be 1 / jacobian since simple_model returns ones
        np.testing.assert_allclose(prob, 1.0 / expected_jacobian)

    def test_prob_with_complex_model(self):
        """Test prob method with a more complex model function."""
        disable_units()

        def mass_dependent_model(dataset, alpha=2.0, **parameters):
            """Model that depends on mass_1 with a power law."""
            return dataset["mass_1"] ** (-alpha)

        model = CosmoModel(
            model_functions=[mass_dependent_model], cosmo_model="Planck15"
        )

        data = {
            "redshift": np.array([0.1, 0.2]),
            "mass_1_detector": np.array([30.0, 40.0]),
        }
        parameters = {"alpha": 2.35}

        prob = model.prob(data, **parameters)

        # Check that probability is computed correctly
        assert len(prob) == 2
        assert np.all(prob > 0)

        # Verify the calculation:
        # mass_1 = mass_1_detector / (1 + z)
        mass_1_source = data["mass_1_detector"] / (1 + data["redshift"])
        # Model returns mass_1^(-alpha)
        model_prob = mass_1_source ** (-2.35)
        # Jacobian is (1 + z) for one detector frame quantity
        jacobian = 1 + data["redshift"]
        expected_prob = model_prob / jacobian

        np.testing.assert_allclose(prob, expected_prob)
