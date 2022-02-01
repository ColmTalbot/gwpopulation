"""
Helper functions for missing functionality in cupy.
"""

try:
    import cupy as xp
    from cupyx.scipy.special import erf, gammaln  # noqa

    CUPY_LOADED = True
except ImportError:
    import numpy as xp
    from scipy.special import erf, gammaln  # noqa

    CUPY_LOADED = False


def betaln(alpha, beta):
    r"""
    Logarithm of the Beta function

    .. math::
        \ln B(\alpha, \beta) = \frac{\ln\gamma(\alpha)\ln\gamma(\beta)}{\ln\gamma(\alpha + \beta)}

    Parameters
    ----------
    alpha: float
        The Beta alpha parameter (:math:`\alpha`)
    beta: float
        The Beta beta parameter (:math:`\beta`)

    Returns
    -------
    ln_beta: float, array-like
        The ln Beta function

    """
    ln_beta = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    return ln_beta


def to_numpy(array):
    """Cast any array to numpy"""
    if not CUPY_LOADED:
        return array
    else:
        return xp.asnumpy(array)


def trapz(y, x=None, dx=1.0, axis=-1):
    """
    Lifted from `numpy <https://github.com/numpy/numpy/blob/v1.15.1/numpy/lib/function_base.py#L3804-L3891>`_.

    Integrate along the given axis using the composite trapezoidal rule.
    Integrate `y` (`x`) along given axis.

    Parameters
    ==========
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    =======
    trapz : float
        Definite integral as approximated by trapezoidal rule.


    References
    ==========
    .. [1] Wikipedia page: http://en.wikipedia.org/wiki/Trapezoidal_rule

    Examples
    ========
    >>> trapz([1,2,3])
    4.0
    >>> trapz([1,2,3], x=[4,6,8])
    8.0
    >>> trapz([1,2,3], dx=2)
    8.0
    >>> a = xp.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> trapz(a, axis=0)
    array([ 1.5,  2.5,  3.5])
    >>> trapz(a, axis=1)
    array([ 2.,  8.])
    """
    y = xp.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = xp.asanyarray(x)
        if x.ndim == 1:
            d = xp.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = xp.diff(x, axis=axis)
    ndim = y.ndim
    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    product = d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0
    try:
        ret = product.sum(axis)
    except ValueError:
        ret = xp.add.reduce(product, axis)
    return ret
