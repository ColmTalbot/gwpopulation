r"""
Searches for gravitational-wave transients are limited by the sensitivity of current detectors.
For this reason it is necessary to quantify the fraction of sources that are expected to be observed
to avoid biases in the population inference.

.. math::

    P_{\rm det}(\Lambda) = \int dd \int d\theta p(d, \theta | \lambda) \Theta(\rho(d) - \rho_{\rm th})

Here :math:`d` is the observed strain data, :math:`\theta` are the parameters of individual sources,
e.g., masses, spins, redshifts, etc., and :math:`\Lambda` are the population parameters.
The quantity :math:`\rho` is a detection statistic, e.g., the signal-to-noise ratio, and :math:`\rho_{\rm th}`
is the threshold for detection.

The most common method to estimate this quantity is to simulate a population of sources from
some reference distribution :math:`p(\theta | \varnothing)` and
compute the fraction of sources that are detected. Using a single reference set of such "injections"
one can estimate :math:`P_{\rm det}(\Lambda)` using Monte Carlo integration.

.. math::

    \hat{P}_{\rm det}(\Lambda) = \frac{1}{N} \sum_{i=1}^N \Theta(\rho_i - \rho_{\rm th})
    \frac{p(\theta_i | \Lambda)}{p(\theta_i | \varnothing)}

Since the detection statistic is independent of the population model, we can remove the :math:`\theta_i`
that don't pass the threshold yielding :math:`M` detected sources.

.. math::

    \hat{P}_{\rm det}(\Lambda) = \frac{1}{N} \sum_{i=1}^M
    \frac{p(\theta_i | \Lambda)}{p(\theta_i | \varnothing)}

This model is implemented in the :class:`gwpopulation.vt.ResamplingVT` class.

A simpler model is to interpolate some expression for

.. math::

    p_{\rm det}(\theta) = \int dd p(d, \theta) \Theta(\rho(d) - \rho_{\rm th})

The quantity :math:`P_{\rm det}(\Lambda)` can be computed by integrating over the
specified :math:`\theta`. This model is implemented in the :class:`gwpopulation.vt.GridVT` class.
Note that the computational cost of this approach scales exponentially with the number of parameters.
"""

import numpy as np
from bilby.hyper.model import Model

from .models.redshift import _Redshift, total_four_volume
from .utils import to_number

xp = np

__all__ = [
    "xp",
    "GridVT",
    "ResamplingVT",
]


class _BaseVT:
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
        The sensitivity labelled :code:`vt` and an entry for each
        parameter to be marginalized over.
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
    r"""
    Evaluate the sensitive volume using a set of found injections.

    See `Farr <https://arxiv.org/abs/1904.10879>`_ for details of the formalism.

    There is an option to use the uncertainty-marginalized vt_factor from
    Equation 11 in `Farr <https://arxiv.org/abs/1904.10879>` by setting
    :code:`marginalize_uncertainty = True`, or use the estimator from
    Equation 8 (default behavior).

    We recommend not enabling :code:`marginalize_uncertainty` and setting
    convergence criteria based on uncertainty in total likelihood in
    HyperparameterLikelihood.

    If using :code:`marginalize_uncertainty`: and
    :math:`n_{\rm eff} < 4 n_{\rm events}` we return :code:`np.inf`
    so that the sample is rejected. This condition is also
    enforced if :code:`enforce_convergence=True`.


    Parameters
    ----------
    model: callable
        Population model
    data: dict
        The found injections and relevant meta data
    n_events: int
        The number of events observed
    marginalize_uncertainty: bool (Default: :code:`False`)
        Whether to return the uncertainty-marginalized pdet from Eq 11
        in `Farr <https://arxiv.org/abs/1904.10879>`_. We recommend not to use
        this as it is not completely understood if this uncertainty
        marginalization is correct.
    enforce_convergence: bool (Default: :code:`True`)
        Whether to enforce the condition that :math:`n_{\rm eff} > 4 n_{\rm events}`.
        This flag only acts when marignalize_uncertainty is :code:`False`.
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
            if isinstance(_model, _Redshift):
                self.redshift_model = _model
        if self.redshift_model is None:
            self._surveyed_hypervolume = total_four_volume(
                lamb=0, analysis_time=self.analysis_time
            )

    def __call__(self, parameters):
        r"""
        Compute the expected fraction of detected sources given a
        set of injections for the specified population model.

        Parameters
        ----------
        parameters: dict
            The population parameters

        Returns
        -------
        (mu, vt_factor): float
            The expected number of detections if
            :code:`self.marginalize_uncertainty=False` or the uncertainty-marginalized
            vt_factor if :code:`self.marginalize_uncertainty=True`.
        var: float
            The variance in the estimate of :math:`P_{\rm det}` if
            :code:`self.marginalize_uncertainty=False`.
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
        r"""
        Check if the estimate of the detection efficiency has converged
        beyond the threshold of :math:`\frac{\mu^2}{\sigma^2} > 4 n_{\rm events}`.
        """
        converged = mu**2 > 4 * self.n_events * var
        return (
            converged,
            xp.nan_to_num(xp.inf * (1 - converged), nan=0, posinf=xp.inf),
        )

    def vt_factor(self, parameters):
        r"""
        Compute the expected number of detections given a set of injections.

        This is implemented as in `Farr <https://arxiv.org/abs/1904.10879>`_

        .. math::

            \text{vt_factor} = \frac{\mu}{\exp\left(\frac{3 + n_{\rm events}}{2 n_{\rm eff}}\right)}

        Parameters
        ----------
        parameters: dict
            The population parameters

        Returns
        -------
        vt_factor: float
            The uncertainty-marginalized vt_factor
        """
        mu, var = self.detection_efficiency(parameters)
        _, correction = self.check_convergence(mu, var)
        n_effective = mu**2 / var
        vt_factor = mu / xp.exp((3 + self.n_events) / 2 / n_effective)
        vt_factor += correction
        return vt_factor

    def detection_efficiency(self, parameters):
        r"""
        Compute the expected fraction of detections given a set of injections
        and the variance in the Monte Carlo estimate.

        Parameters
        ----------
        parameters: dict
            The population parameters

        Returns
        -------
        mu: float
            The expected fracion of detections :math:`P_{\rm det}`.
        var: float
            The variance in the estimate of :math:`P_{\rm det}`.
        """
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
        The total surveyed 4-volume with units of :math:`{\rm Gpc}^3{\rm yr}`.

        .. math::
            \mathcal{V} = \int dz \frac{dV_c}{dz} \frac{\psi(z)}{1 + z}

        If no redshift model is specified, assume :math:`\psi(z)=1`.

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters to compute the volume at

        Returns
        -------
        float
            The volume

        """
        if self.redshift_model is None:
            return self._surveyed_hypervolume
        elif isinstance(self.redshift_model, _Redshift):
            return (
                self.redshift_model.normalisation(parameters) / 1e9 * self.analysis_time
            )
