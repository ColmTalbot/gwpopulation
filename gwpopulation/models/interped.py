from ..cupy_utils import trapz, xp

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
        
    @property
    def variable_names(self):
        keys = [f"x{ii}" for ii in range(self.nodes)]
        keys += [f"f{ii}" for ii in range(self.nodes)]       
        return keys

    def setup_interpolant(self, nodes, values):
        from cached_interpolate import CachingInterpolant

        kwargs = dict(x=nodes, y=values, kind=self.kind, backend=xp)
        self._norm_spline = CachingInterpolant(**kwargs)
        self._data_spline = {param: CachingInterpolant(**kwargs) for param in self.parameters}
        #self._data_spline = CachingInterpolant(**kwargs)

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
        
        p_x = xp.exp(perturbation)
        
        return p_x

    def norm_p_x(self, f_splines=None, x_splines=None, **kwargs):
        if self.norm_selector is None:
            self.norm_selector = (self._xs >= x_splines[0]) & (
                self._xs <= x_splines[-1])
            
        perturbation = self._norm_spline(x=self._xs[self.norm_selector], y=f_splines)
        p_x = xp.zeros(len(self._xs))
        p_x[self.norm_selector] = xp.exp(perturbation)
        norm = trapz(p_x, self._xs)
        return norm
    
    def p_x_identical(self, dataset, **kwargs):

        f_splines = xp.array([kwargs.pop(f"f{i}") for i in range(self.nodes)])
        x_splines = xp.array([kwargs.pop(f"x{i}") for i in range(self.nodes)])
        
        
        
        p_x = xp.ones(len(dataset[self.parameters[0]]))
        
        for param in self.parameters:
            
            p_x *= self.p_x_unnormed(dataset, param, x_splines=x_splines, f_splines = f_splines, **kwargs)
        norm = self.norm_p_x(f_splines = f_splines, x_splines = x_splines, **kwargs)
        p_x /= norm ** len(self.parameters)
        return p_x
