"""Tests for gwpopulation.experimental.cosmo_models module."""

import numpy as np
import pytest
from wcosmo.utils import disable_units

from gwpopulation.experimental.cosmo_models import CosmoMixin


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
