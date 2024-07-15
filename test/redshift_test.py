import numpy as np
import pytest
from bilby.core.prior import PriorDict, Uniform
from wcosmo.astropy import Planck15
from wcosmo.utils import disable_units

import gwpopulation
from gwpopulation.models import redshift

from . import TEST_BACKENDS

N_TEST = 100


def _run_model_normalisation(model, priors, xp=np):
    zs = xp.linspace(1e-3, 2.3, 1000)
    test_data = dict(redshift=zs)
    norms = list()
    for _ in range(N_TEST):
        p_z = model(test_data, **priors.sample())
        norms.append(float(xp.trapz(p_z, zs)))
    assert np.max(np.abs(np.array(norms) - 1)) < 1e-3


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_powerlaw_normalised(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    model = redshift.PowerLawRedshift()
    priors = PriorDict()
    priors["lamb"] = Uniform(-15, 15)
    _run_model_normalisation(model=model, priors=priors, xp=xp)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_madau_dickinson_normalised(backend):
    gwpopulation.set_backend(backend)
    xp = gwpopulation.utils.xp
    model = redshift.MadauDickinsonRedshift()
    priors = PriorDict()
    priors["gamma"] = Uniform(-15, 15)
    priors["kappa"] = Uniform(-15, 15)
    priors["z_peak"] = Uniform(0, 5)
    _run_model_normalisation(model=model, priors=priors, xp=xp)


@pytest.mark.parametrize("backend", TEST_BACKENDS)
def test_powerlaw_volume(backend):
    """
    Test that the total volume matches astropy for a
    trivial case
    """
    gwpopulation.set_backend(backend)
    disable_units()
    xp = gwpopulation.utils.xp
    zs = xp.linspace(1e-3, 2.3, 1000)
    zs_numpy = gwpopulation.utils.to_numpy(zs)
    model = redshift.PowerLawRedshift()
    parameters = dict(lamb=1)
    total_volume = np.trapz(
        Planck15.differential_comoving_volume(zs_numpy) * 4 * np.pi,
        zs_numpy,
    )
    approximation = float(model.normalisation(parameters))
    assert abs(total_volume - approximation) / total_volume < 1e-2


def test_zero_outside_domain():
    model = redshift.PowerLawRedshift(z_max=1)
    assert model(dict(redshift=5), lamb=1) == 0


def test_four_volume():
    disable_units()
    assert (
        Planck15.comoving_volume(2.3) / 1e9
        - redshift.total_four_volume(lamb=1, analysis_time=1, max_redshift=2.3)
        < 1e-3
    )
