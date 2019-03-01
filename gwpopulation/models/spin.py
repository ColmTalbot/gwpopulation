from ..utils import beta_dist, truncnorm


def iid_spin(dataset, xi_spin, sigma_spin, amax, alpha_chi, beta_chi):
    """
    Independently and identically distributed spins.
    """
    prior = (
        iid_spin_orientation_gaussian_isotropic(dataset, xi_spin, sigma_spin) *
        iid_spin_magnitude_beta(dataset, amax, alpha_chi, beta_chi))
    return prior


def iid_spin_magnitude_beta(dataset, amax=1, alpha_chi=1, beta_chi=1):
    """
    Independently and identically distributed spin magnitudes.
    """
    return independent_spin_magnitude_beta(
        dataset, alpha_chi, alpha_chi, beta_chi, beta_chi, amax, amax)


def independent_spin_magnitude_beta(dataset, alpha_chi_1, alpha_chi_2,
                                    beta_chi_1, beta_chi_2, amax_1, amax_2):
    """ Independent beta distributions for both spin magnitudes.

    https://arxiv.org/abs/1805.06442 Eq. (10)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays containing 'a_1' and 'a_2'.
    alpha_chi_1, beta_chi_1: float
        Parameters of Beta distribution for more massive black hole.
    alpha_chi_2, beta_chi_2: float
        Parameters of Beta distribution for less massive black hole.
    amax_1, amax_2: float
        Maximum spin of the more/less massive black hole.
    """
    if alpha_chi_1 < 0 or beta_chi_1 < 0 or alpha_chi_2 < 0 or beta_chi_2 < 0:
        return 0
    prior = (beta_dist(dataset['a_1'], alpha_chi_1, beta_chi_1, scale=amax_1) *
             beta_dist(dataset['a_2'], alpha_chi_2, beta_chi_2, scale=amax_2))
    return prior


def iid_spin_orientation_gaussian_isotropic(dataset, xi_spin, sigma_spin):
    """
    Independently and identically distributed spin orientations.
    """
    return independent_spin_orientation_gaussian_isotropic(
        dataset, xi_spin, sigma_spin, sigma_spin)


def independent_spin_orientation_gaussian_isotropic(
        dataset, xi_spin, sigma_1, sigma_2):
    """A mixture model of spin orientations with isotropic and normally
    distributed components.

    https://arxiv.org/abs/1704.08370 Eq. (4)

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'cos_tilt_1' and 'cos_tilt_2'.
    xi_spin: float
        Fraction of black holes in preferentially aligned component.
    sigma_1: float
        Width of preferentially aligned component for the more
        massive black hole.
    sigma_2: float
        Width of preferentially aligned component for the less
        massive black hole.
    """
    prior = ((1 - xi_spin) / 4 + xi_spin *
             truncnorm(dataset['cos_tilt_1'], 1, sigma_1, 1, -1) *
             truncnorm(dataset['cos_tilt_2'], 1, sigma_2, 1, -1))
    return prior
