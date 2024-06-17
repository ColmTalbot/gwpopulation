from functools import partial

import numpy as np

from ..utils import to_numpy

xp = np

__all__ = [
    "InterpolatedNoBaseModelIdentical",
    "_setup_interpolant",
]


def _setup_interpolant(nodes, values, kind="cubic", backend=None):
    """
    Create a caching spline interpolant.

    .. note::
        This function is not intended to be used directly but can overwrite
        the :code:`GWPopulation` backend.

    Parameters
    ==========
    nodes: array-like
        The positions of the nodes
    values: array-like
        The positions that the interpolant will be evaluated at.
    kind: str
        The interpolation order of the spline, default="cubic"
    backend: module
        The backend to use, default=:code:`numpy`.
    """
    from cached_interpolate import RegularCachingInterpolant as CachingInterpolant

    if backend is None:
        backend = xp

    nodes = to_numpy(nodes)
    interpolant = CachingInterpolant(nodes, nodes, kind=kind, backend=backend)
    interpolant.conversion = backend.array(interpolant.conversion)
    interpolant = partial(interpolant, backend.array(values))
    return interpolant


class InterpolatedNoBaseModelIdentical:
    """
    Base class for the Interpolated classes with no base model

    Parameters
    ==========
    parameters: list
        List of parameters to interpolate over, e.g., :code:`["a_1", "a_2"]`
    minimum: float
        Minimum value to normalize the spline over
    maximum: float
        Maximum value to normalize the spline over
    nodes: int
        Number of nodes to use in the spline, default=10
    kind: str
        The interpolation order of the spline, default=:code:`"cubic"`
    log_nodes: bool
        Whether to use log-spaced nodes, default=False
    regularize: bool
        Whether to regularize the spline node values to have root-mean-square value
        :code:`rms{name}`, default=False
    """

    def __init__(
        self,
        parameters,
        minimum,
        maximum,
        nodes=10,
        kind="cubic",
        log_nodes=False,
        regularize=False,
    ):
        self.nodes = nodes
        self._norm_spline = None
        self._data_spline = dict()
        self.kind = kind
        self._xs = xp.linspace(minimum, maximum, 10 * self.nodes)
        self.parameters = parameters
        self.min = minimum
        self.max = maximum
        self.log_nodes = log_nodes

        self.base = self.parameters[0].strip("_1")
        self.xkeys = [f"{self.base}{ii}" for ii in range(self.nodes)]
        self.fkeys = [f"f{self.base}{ii}" for ii in range(self.nodes)]
        self.regularize = regularize

    def __call__(self, dataset, **kwargs):
        """
        A wrapper to :func:`p_x_identical`
        """
        return self.p_x_identical(dataset, **kwargs)

    @property
    def variable_names(self):
        """
        The names of the hyperparameters of the model.

        These are the names of the parameters describing:

        - the node locations
        - the value of the model at the nodes
        - the root-mean-square value of the model if :code:`self.regularize=True`
        """
        keys = self.xkeys + self.fkeys
        if self.regularize:
            keys += [f"rms{self.base}"]
        return keys

    def setup_interpolant(self, nodes, values):
        if self.log_nodes:
            func = xp.log
        else:
            func = xp.array
        kwargs = dict(kind=self.kind, backend=xp)
        self._norm_spline = _setup_interpolant(func(nodes), func(self._xs), **kwargs)
        self._data_spline = {
            param: _setup_interpolant(func(nodes), func(values[param]), **kwargs)
            for param in self.parameters
        }

    def p_x_unnormed(self, dataset, parameter, x_splines, f_splines, **kwargs):
        """
        Calculate the unnormalized likelihood of the dataset given the model

        Parameters
        ==========
        dataset: dict
            Dictionary containing the data to evaluate the likelihood at
        parameter: str
            The parameter to evaluate the likelihood for
        x_splines: array-like
            The positions of the spline nodes
        f_splines: array-like
            The values at the spline nodes
        kwargs: dict, UNUSED

        Returns
        =======
        p_x: array-like
            The unnormalized likelihood of the dataset given the model at the
            provided points
        """
        if self._norm_spline is None:
            self.setup_interpolant(x_splines, dataset)

        perturbation = self._data_spline[parameter](y=f_splines)

        p_x = xp.exp(perturbation)
        p_x *= (dataset[f"{parameter}"] >= x_splines[0]) & (
            dataset[f"{parameter}"] <= x_splines[-1]
        )
        return p_x

    def norm_p_x(self, f_splines=None, x_splines=None, **kwargs):
        """
        Calculate the normalization of the spline

        Parameters
        ==========
        f_splines: array-like
            The values at the spline nodes
        x_splines: array-like
            The positions of the spline nodes
        kwargs: dict, UNUSED

        Returns
        =======
        norm: float
            The normalization of the spline
        """
        perturbation = self._norm_spline(y=f_splines)
        p_x = xp.exp(perturbation)
        p_x *= (self._xs >= x_splines[0]) & (self._xs <= x_splines[-1])
        norm = xp.trapz(p_x, self._xs)
        return norm

    def extract_spline_points(self, kwargs):
        """
        Extract the node positions and values from the dictionary of parameters

        Parameters
        ==========
        kwargs: dict
            Dictionary containing :code:`{self.base}_ii, f{self.base_ii}` and
            optionally :code`rms{self.base}`

        Returns
        =======
        f_splines: array-like
            The values at the spline nodes
        x_splines: array-like
            The positions of the spline nodes
        """
        f_splines = xp.array([kwargs[key] for key in self.fkeys])
        if self.regularize:
            f_splines *= kwargs[f"rms{self.base}"] / xp.mean(f_splines**2) ** 0.5
        x_splines = xp.array([kwargs[key] for key in self.xkeys])
        return f_splines, x_splines

    def p_x_identical(self, dataset, **kwargs):
        """
        Calculate the likelihood of the dataset given the model assuming
        that all the parameters are identically distributed.

        Parameters
        ==========
        dataset: dict
            Dictionary containing the data to evaluate the likelihood at
        kwargs: dict
            Dictionary containing the parameters of the model

        Returns
        =======
        p_x: array-like
            The likelihood of the dataset given the model at the provided points
        """
        self.infer_n_nodes(**kwargs)

        f_splines, x_splines = self.extract_spline_points(kwargs)

        p_x = xp.ones(xp.shape(dataset[self.parameters[0]]))

        for param in self.parameters:
            p_x *= self.p_x_unnormed(
                dataset, param, x_splines=x_splines, f_splines=f_splines, **kwargs
            )

        norm = self.norm_p_x(f_splines=f_splines, x_splines=x_splines, **kwargs)
        p_x /= norm ** len(self.parameters)
        return p_x

    def infer_n_nodes(self, **kwargs):
        """
        Infer the number of nodes from the dictionary of parameters.
        This method looks for the first missing parameter matching
        :code:`f{self.base}{nodes}` and updates the number of nodes
        to the number of found parameters.
        """
        nodes = self.nodes

        while True:
            if f"f{self.base}{nodes}" in kwargs:
                nodes += 1
            else:
                break

        if not nodes == self.nodes:
            print(
                f"Different number of nodes! Using {nodes} nodes instead of {self.nodes}"
            )
            self.__init__(nodes=nodes)
