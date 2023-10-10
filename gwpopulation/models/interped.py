import numpy as np

xp = np


class InterpolatedNoBaseModelIdentical(object):
    """
    Base class for the Interpolated classes with no base model
    """

    def __init__(self, parameters, minimum, maximum, nodes=10, kind="cubic"):
        """ """
        self.nodes = nodes
        self.norm_selector = None
        self.spline_selector = None
        self._norm_spline = None
        self._data_spline = dict()
        self.kind = kind
        self._xs = xp.linspace(minimum, maximum, 10 * self.nodes)
        self.parameters = parameters
        self.min = minimum
        self.max = maximum

        self.base = self.parameters[0].strip("_1")
        self.xkeys = [f"{self.base}{ii}" for ii in range(self.nodes)]
        self.fkeys = [f"f{self.base}{ii}" for ii in range(self.nodes)]

    def __call__(self, dataset, **kwargs):
        return self.p_x_identical(dataset, **kwargs)

    @property
    def variable_names(self):

        keys = self.xkeys + self.fkeys
        return keys

    def setup_interpolant(self, nodes, values):
        from cached_interpolate import CachingInterpolant

        kwargs = dict(x=nodes, y=values, kind=self.kind, backend=xp)
        self._norm_spline = CachingInterpolant(**kwargs)
        self._data_spline = {
            param: CachingInterpolant(**kwargs) for param in self.parameters
        }

    def p_x_unnormed(self, dataset, parameter, x_splines, f_splines, **kwargs):

        if self.spline_selector is None:
            if self._norm_spline is None:
                self.setup_interpolant(x_splines, f_splines)

            self.spline_selector = (dataset[f"{parameter}"] >= x_splines[0]) & (
                dataset[f"{parameter}"] <= x_splines[-1]
            )

        perturbation = self._data_spline[parameter](
            x=dataset[f"{parameter}"][self.spline_selector], y=f_splines
        )

        p_x = xp.zeros(xp.shape(dataset[self.parameters[0]]))
        p_x[self.spline_selector] = xp.exp(perturbation)
        return p_x

    def norm_p_x(self, f_splines=None, x_splines=None, **kwargs):
        if self.norm_selector is None:
            self.norm_selector = (self._xs >= x_splines[0]) & (
                self._xs <= x_splines[-1]
            )

        perturbation = self._norm_spline(x=self._xs[self.norm_selector], y=f_splines)
        p_x = xp.zeros(len(self._xs))
        p_x[self.norm_selector] = xp.exp(perturbation)
        norm = xp.trapz(p_x, self._xs)
        return norm

    def p_x_identical(self, dataset, **kwargs):

        self.infer_n_nodes(**kwargs)

        f_splines = np.array([kwargs[f"{key}"] for key in self.fkeys])
        x_splines = np.array([kwargs[f"{key}"] for key in self.xkeys])

        p_x = xp.ones(xp.shape(dataset[self.parameters[0]]))

        for param in self.parameters:
            p_x *= self.p_x_unnormed(
                dataset, param, x_splines=x_splines, f_splines=f_splines, **kwargs
            )

        norm = self.norm_p_x(f_splines=f_splines, x_splines=x_splines, **kwargs)
        p_x /= norm ** len(self.parameters)
        return p_x

    def infer_n_nodes(self, **kwargs):
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
