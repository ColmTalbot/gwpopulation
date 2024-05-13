"""
Sensitive volume estimation.
"""

import numpy as np
from bilby.hyper.model import Model

from .experimental.cosmo_models import _BaseRedshift
from .models.redshift import _Redshift, total_four_volume
from .utils import to_number

xp = np


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
        self.axes = {int(np.where(shape == lens[key])[0][0]): key for key in self.data}
        self.ndim = len(self.axes)

    def __call__(self, parameters):
        self.model.parameters.update(parameters)
        vt_fac = self.model.prob(self.data) * self.vts
        for ii in range(self.ndim):
            vt_fac = xp.trapz(
                vt_fac, self.values[self.axes[self.ndim - ii - 1]], axis=-1
            )
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
    marginalize_uncertainty: bool (Default: False)
        Whether to return the uncertainty-marginalized pdet from Eq 11
        in https://arxiv.org/abs/1904.10879. Recommend not to use this
        as it is not completely understood if this uncertainty
        marginalization is correct.
    enforce_convergence: bool (Default: True)
        Whether to enforce the condition that n_effective > 4*n_obs.
        This flag only acts when marignalize_uncertainty is False.
    """

    def __init__(
        self,
        model,
        data,
        n_events=np.inf,
        marginalize_uncertainty=False,
        enforce_convergence=True,
    ):
        super(ResamplingVT, self).__init__(model=model, data=data)
        self.n_events = n_events
        self.total_injections = data.get("total_generated", len(data["prior"]))
        self.analysis_time = data.get("analysis_time", 1)
        self.redshift_model = None
        self.marginalize_uncertainty = marginalize_uncertainty
        self.enforce_convergence = enforce_convergence
        for _model in self.model.models:
            if isinstance(_model, _Redshift) or isinstance(_model, _BaseRedshift):
                self.redshift_model = _model
        if self.redshift_model is None:
            self._surveyed_hypervolume = total_four_volume(
                lamb=0, analysis_time=self.analysis_time
            )

    def __call__(self, parameters):
        """
        Compute the expected number of detections given a set of injections.

        Option to use the uncertainty-marginalized vt_factor from Equation 11
        in https://arxiv.org/abs/1904.10879 by setting `marginalize_uncertainty`
        to True, or use the estimator from Equation 8 (default behavior).

        Recommend not enabling marginalize_uncertainty and setting convergence
        criteria based on uncertainty in total likelihood in HyperparameterLikelihood.

        If using `marginalize_uncertainty` and n_effective < 4 * n_events we
        return np.inf so that the sample is rejected. This condition is also
        enforced if `enforce_convergence` is True.

        Returns either vt_factor or mu and var.

        Parameters
        ----------
        parameters: dict
            The population parameters
        """
        if not self.marginalize_uncertainty:
            mu, var = self.detection_efficiency(parameters)
            if self.enforce_convergence:
                _, correction = self.check_convergence(mu, var)
                mu += correction
            return mu, var
        else:
            vt_factor = self.vt_factor(parameters)
            return vt_factor

    def check_convergence(self, mu, var):
        converged = mu**2 > 4 * self.n_events * var
        return (
            converged,
            xp.nan_to_num(xp.inf * (1 - converged), nan=0, posinf=xp.inf),
        )

    def vt_factor(self, parameters):
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
        _, correction = self.check_convergence(mu, var)
        n_effective = mu**2 / var
        vt_factor = mu / xp.exp((3 + self.n_events) / 2 / n_effective)
        vt_factor += correction
        return vt_factor

    def detection_efficiency(self, parameters):
        self.model.parameters.update(parameters)
        weights = self.model.prob(self.data) / self.data["prior"]
        mu = to_number(xp.sum(weights) / self.total_injections, float)
        var = to_number(
            xp.sum(weights**2) / self.total_injections**2
            - mu**2 / self.total_injections,
            float,
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
        elif isinstance(self.redshift_model, _Redshift):
            return (
                self.redshift_model.normalisation(parameters) / 1e9 * self.analysis_time
            )
        elif (
            isinstance(self.redshift_model, _BaseRedshift)
            and self.redshift_model.cosmo_model is not None
        ):
            self.redshift_model.update_dvc_dz(**parameters)
            return (
                self.redshift_model.normalisation(parameters) / 1e9 * self.analysis_time
            )
