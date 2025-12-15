import bilby
import pandas as pd
import pytest
from bilby.hyper.model import Model
from numpyro import handlers

import gwpopulation
from gwpopulation.experimental.numpyro import gwpopulation_likelihood_model


@pytest.mark.parametrize("apply_selection", [True, False])
def test_gwpopulation_likelihood_model_matches_standard(apply_selection):
    """
    Test that gwpopulation_likelihood_model produces the same likelihood
    as the standard HyperparameterLikelihood.log_likelihood_ratio method.
    """
    # Skip if JAX is not available
    pytest.importorskip("jax")
    pytest.importorskip("numpyro")

    # Set backend to JAX for numpyro compatibility
    gwpopulation.set_backend("jax")
    xp = gwpopulation.models.mass.xp
    bilby.core.utils.random.seed(10)
    rng = bilby.core.utils.random.rng

    # Set up the model
    model = Model(
        [
            gwpopulation.models.mass.SinglePeakSmoothedMassDistribution(),
            gwpopulation.models.spin.independent_spin_magnitude_beta,
            gwpopulation.models.spin.independent_spin_orientation_gaussian_isotropic,
            gwpopulation.models.redshift.PowerLawRedshift(),
        ],
        cache=False,
    )
    vt_model = Model(
        [
            gwpopulation.models.mass.SinglePeakSmoothedMassDistribution(),
            gwpopulation.models.spin.independent_spin_magnitude_beta,
            gwpopulation.models.spin.independent_spin_orientation_gaussian_isotropic,
            gwpopulation.models.redshift.PowerLawRedshift(),
        ],
        cache=False,
    )

    # Create synthetic data
    bounds = dict(
        mass_1=(20, 25),
        mass_ratio=(0.9, 1),
        a_1=(0, 1),
        a_2=(0, 1),
        cos_tilt_1=(-1, -1),
        cos_tilt_2=(-1, -1),
        redshift=(0, 2),
        prior=(1, 1),
    )
    posteriors = [
        pd.DataFrame({key: rng.uniform(*bound, 100) for key, bound in bounds.items()})
        for _ in range(10)
    ]
    vt_data = {
        key: xp.asarray(rng.uniform(*bound, 10000)) for key, bound in bounds.items()
    }

    # Create selection function
    selection = gwpopulation.vt.ResamplingVT(vt_model, vt_data, len(posteriors))

    # Create likelihood
    likelihood = gwpopulation.hyperpe.HyperparameterLikelihood(
        hyper_prior=model,
        posteriors=posteriors,
        selection_function=selection if apply_selection else lambda args: 1,
    )

    # Get a prior sample
    priors = bilby.core.prior.PriorDict("priors/bbh_population.prior")
    prior_sample = priors.sample()

    # Get the standard likelihood value
    standard_ln_l = likelihood.log_likelihood_ratio(prior_sample)

    # Get the numpyro likelihood value using a seed handler
    # Note: predictive_resample=False to avoid the stochastic resampling
    with handlers.seed(rng_seed=0):
        numpyro_ln_l = gwpopulation_likelihood_model(
            likelihood,
            prior_sample,
            apply_selection=apply_selection,
            predictive_resample=False,
        )

    # Convert to number if needed
    if hasattr(numpyro_ln_l, "item"):
        numpyro_ln_l = float(numpyro_ln_l.item())
    else:
        numpyro_ln_l = float(numpyro_ln_l)

    # Check that they match
    assert abs(standard_ln_l - numpyro_ln_l) < 1e-6, (
        f"Numpyro likelihood {numpyro_ln_l} does not match "
        f"standard likelihood {standard_ln_l}"
    )
