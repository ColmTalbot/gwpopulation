import glob

import bilby
import numpy as np
import pandas as pd

import gwpopulation


def test_likelihood_evaluation():
    np.random.seed(10)

    model = bilby.hyper.model.Model(
        [
            gwpopulation.models.mass.SinglePeakSmoothedMassDistribution(),
            gwpopulation.models.spin.independent_spin_magnitude_beta,
            gwpopulation.models.spin.independent_spin_orientation_gaussian_isotropic,
            gwpopulation.models.redshift.PowerLawRedshift(),
        ]
    )
    vt_model = bilby.hyper.model.Model(
        [
            gwpopulation.models.mass.SinglePeakSmoothedMassDistribution(),
            gwpopulation.models.spin.independent_spin_magnitude_beta,
            gwpopulation.models.spin.independent_spin_orientation_gaussian_isotropic,
            gwpopulation.models.redshift.PowerLawRedshift(),
        ]
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
        pd.DataFrame(
            {key: np.random.uniform(*bound, 100) for key, bound in bounds.items()}
        )
        for _ in range(10)
    ]
    vt_data = {key: np.random.uniform(*bound, 10000) for key, bound in bounds.items()}

    selection = gwpopulation.vt.ResamplingVT(vt_model, vt_data, len(posteriors))

    likelihood = gwpopulation.hyperpe.HyperparameterLikelihood(
        hyper_prior=model, posteriors=posteriors, selection_function=selection
    )

    priors = bilby.core.prior.PriorDict("priors/bbh_population.prior")

    likelihood.parameters.update(priors.sample())
    assert abs(likelihood.log_likelihood_ratio() - 0.06141098844907589) < 0.01


def test_prior_files_load():
    for fname in glob.glob("priors/*.prior"):
        priors = bilby.core.prior.PriorDict(fname)
