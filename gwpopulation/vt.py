"""
Sensitive volume estimation.
"""

from bilby.hyper.model import Model

import numpy as np

from .cupy_utils import trapz, xp
from .models.redshift import _Redshift, total_four_volume


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
    """
    Evaluate the sensitive volume on a grid.

    Parameters
    ----------
    model: callable
        Population model
    data: dict
        The sensitivity labelled `vt` and an entry for each parameter to be marginalized over.
    """

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
    """
    Evaluate the sensitive volume using a set of found injections.

    See https://arxiv.org/abs/1904.10879 for details of the formalism.

    Parameters
    ----------
    model: callable
        Population model
    data: dict
        The found injections and relevant meta data
    n_events: int
        The number of events observed
    """

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

        If n_effective < 4 * n_events we return np.inf so that the sample
        is rejected.

        Parameters
        ----------
        parameters: dict
            The population parameters
        """
        mu, var = self.detection_efficiency(parameters)
        if mu**2 <= 4 * self.n_events * var:
            return np.inf
        n_effective = mu**2 / var
        vt_factor = mu / np.exp((3 + self.n_events) / 2 / n_effective)
        return vt_factor

    def detection_efficiency(self, parameters):
        self.model.parameters.update(parameters)
        weights = self.model.prob(self.data) / self.data["prior"]
        mu = float(xp.sum(weights) / self.total_injections)
        var = float(
            xp.sum(weights**2) / self.total_injections**2
            - mu**2 / self.total_injections
        )
        return mu, var

    def surveyed_hypervolume(self, parameters):
        r"""
        The total surveyed 4-volume with units of :math:`Gpc^3yr`.

        .. math::
            \mathcal{V} = \int dz \frac{dV_c}{dz} \frac{\psi(z)}{1 + z}

        If no redshift model is specified, assume :math:`\psi(z)=1`.

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters to compute the volume at

        Returns
        -------
        float: The volume

        """
        if self.redshift_model is None:
            return self._surveyed_hypervolume
        else:
            return (
                self.redshift_model.normalisation(parameters) / 1e9 * self.analysis_time
            )
