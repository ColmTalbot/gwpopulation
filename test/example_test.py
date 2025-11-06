import glob

import bilby
import pandas as pd
import pytest
from bilby.hyper.model import Model

import gwpopulation
from gwpopulation.experimental.jax import JittedLikelihood


@pytest.mark.parametrize("jit", [True, False])
def test_likelihood_evaluation(backend, jit):
    if backend != "jax" and jit:
        pytest.skip(reason="JIT only works with JAX")

    gwpopulation.set_backend(backend)
    xp = gwpopulation.models.mass.xp
    bilby.core.utils.random.seed(10)
    rng = bilby.core.utils.random.rng

    if jit:
        cache = False
    else:
        cache = True

    model = Model(
        [
            gwpopulation.models.mass.SinglePeakSmoothedMassDistribution(),
            gwpopulation.models.spin.independent_spin_magnitude_beta,
            gwpopulation.models.spin.independent_spin_orientation_gaussian_isotropic,
            gwpopulation.models.redshift.PowerLawRedshift(),
        ],
        cache=cache,
    )
    vt_model = Model(
        [
            gwpopulation.models.mass.SinglePeakSmoothedMassDistribution(),
            gwpopulation.models.spin.independent_spin_magnitude_beta,
            gwpopulation.models.spin.independent_spin_orientation_gaussian_isotropic,
            gwpopulation.models.redshift.PowerLawRedshift(),
        ],
        cache=cache,
    )

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

    selection = gwpopulation.vt.ResamplingVT(vt_model, vt_data, len(posteriors))

    likelihood = gwpopulation.hyperpe.HyperparameterLikelihood(
        hyper_prior=model,
        posteriors=posteriors,
        selection_function=selection,
    )

    priors = bilby.core.prior.PriorDict("priors/bbh_population.prior")
    prior_sample = priors.sample()
    assert abs(likelihood.log_likelihood_ratio(prior_sample) + 7.037596674351107) < 0.01
    if jit:
        likelihood = JittedLikelihood(likelihood)
    assert abs(likelihood.log_likelihood_ratio(prior_sample) + 7.037596674351107) < 0.01
    likelihood.posterior_predictive_resample(pd.DataFrame(priors.sample(5)))


def test_prior_files_load():
    for fname in glob.glob("priors/*.prior"):
        print(fname)
        _ = bilby.core.prior.PriorDict(fname)
