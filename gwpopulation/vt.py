from bilby.hyper.model import Model

import numpy as np

from .cupy_utils import trapz, xp
from .models.redshift import _Redshift


class _BaseVT(object):

    def __init__(self, model, data):
        self.data = data
        if isinstance(model, list):
            model = Model(model)
        elif not isinstance(model, Model):
            model = Model([model])
        self.model = model

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class GridVT(_BaseVT):

    def __init__(self, model, data):
        self.vts = data.pop("vt")
        super(GridVT, self).__init__(model=model, data=data)
        self.values = {key: xp.unique(self.data[key]) for key in self.data}
        shape = np.array(list(self.data.values())[0].shape)
        lens = {key: len(self.values[key]) for key in self.data}
        self.axes = {int(np.where(shape == lens[key])[0]): key for key in self.data}
        self.ndim = len(self.axes)

    def __call__(self, parameters):
        self.model.parameters.update(parameters)
        vt_fac = self.model.prob(self.data) * self.vts
        for ii in range(self.ndim):
            vt_fac = trapz(vt_fac, self.values[self.axes[self.ndim - ii - 1]], axis=-1)
        return vt_fac


class ResamplingVT(_BaseVT):

    def __init__(self, model, data, n_events=np.inf):
        super(ResamplingVT, self).__init__(model=model, data=data)
        self.n_events = n_events
        self.total_injections = data.get("total_generated", len(data["prior"]))
        self.analysis_time = data.get("analysis_time", 1)
        self.redshift_model = None
        for _model in self.model.models:
            if isinstance(_model, _Redshift):
                self.redshift_model = _model
        if self.redshift_model is None:
            self._surveyed_hypervolume = total_four_volume(
                lamb=0, analysis_time=self.analysis_time
            )

    def __call__(self, parameters):
        """
        Compute the expected number of detections given a set of injections.

        This should be implemented as in https://arxiv.org/abs/1904.10879

        If 4 * n_events < n_effective we return np.inf so that the sample
        is rejected.

        Parameters
        ----------
        parameters: dict
            The population parameters
        """
        mu, var = self.detection_efficiency(parameters)
        n_effective = mu ** 2 / var
        if n_effective < 4 * self.n_events:
            return np.inf
        vt_factor = mu / np.exp((3 + self.n_events) / 2 / n_effective)
        return vt_factor

    def detection_efficiency(self, parameters):
        self.model.parameters.update(parameters)
        weights = self.model.prob(self.data) / self.data["prior"]
        mu = float(xp.sum(weights) / self.total_injections)
        var = float(
            xp.sum(weights ** 2) / self.total_injections ** 2
            - mu ** 2 / self.total_injections
        )
        return mu, var

    def surveyed_hypervolume(self, parameters):
        if self.redshift_model is None:
            return self._surveyed_hypervolume
        else:
            return (
                self.redshift_model.total_spacetime_volume(**parameters)
                / 1e9
                * self.analysis_time
            )


def total_four_volume(lamb, analysis_time, max_redshift=2.3):
    from astropy.cosmology import Planck15

    redshifts = np.linspace(0, max_redshift, 1000)
    psi_of_z = (1 + redshifts) ** lamb
    normalization = 4 * np.pi / 1e9 * analysis_time
    total_volume = (
        np.trapz(
            Planck15.differential_comoving_volume(redshifts).value
            / (1 + redshifts)
            * psi_of_z,
            redshifts,
        )
        * normalization
    )
    return total_volume
