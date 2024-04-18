"""
Sensitive volume estimation for spectral siren (i.e., in detector frame)
"""
import numpy as np

from gwpopulation.experimental.cosmo_models import _CosmoRedshift
from gwpopulation.utils import to_number
from gwpopulation.vt import ResamplingVT

xp = np

class CosmoResamplingVT(ResamplingVT):
    """
    Evaluate the sensitive volume using a set of found injections.
    See https://arxiv.org/abs/1904.10879 for details of the formalism.
    Parameters
    ----------
    model: callable
        Population model
    data: dict
        The found injections and relevant meta data, samples should be in detector frame, (e.g., mass_1_detector, mass_ratio, luminosity_distance. Prior of samples should be in detector frame as well.)
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
    astropy_conv: boolean
            Wether luminosity distance - redshift conversions are done with astropy
    """

    def __init__(
        self,
        model,
        data,
        n_events=np.inf,
        marginalize_uncertainty=False,
        enforce_convergence=True,
        astropy_conv=False,
    ):
        super(CosmoResamplingVT, self).__init__(model=model, data=data, n_events=np.inf,marginalize_uncertainty=marginalize_uncertainty,enforce_convergence=enforce_convergence)
        for _model in self.model.models:
            if isinstance(_model, _CosmoRedshift):
                self.redshift_model = _model
        self.astropy_conv = astropy_conv

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

    def detection_efficiency(self, parameters):
        self.model.parameters.update(parameters)
        samples_in_source = self.redshift_model.detector_frame_to_source_frame(self.data, self.model.parameters['H0'], self.model.parameters['Om0'], self.model.parameters['w0'], self.astropy_conv)
        jac = self.redshift_model.detector_to_source_jacobian(samples_in_source['redshift'], self.model.parameters['H0'], self.model.parameters['Om0'], self.model.parameters['w0'], self.data['luminosity_distance'])
        jac *= (1+samples_in_source['redshift'])
        weights = self.model.prob(samples_in_source) / self.data["prior"] / jac
        mu = to_number(xp.sum(weights) / self.total_injections, float)
        var = to_number(
            xp.sum(weights**2) / self.total_injections**2
            - mu**2 / self.total_injections,
            float,
        )
        return mu, var