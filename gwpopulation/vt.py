from bilby.hyper.model import Model

from . import trapz, xp


class GridVT(object):

    def __init__(self, model, data):
        self.vts = data.pop('vt')
        self.data = data
        if isinstance(model, list):
            model = Model(model)
        elif not isinstance(model, Model):
            model = Model([model])
        self.model = model
        self.values = {key: xp.unique(self.data[key]) for key in self.data}
        self.axes = {xp.where(self.data[key].shape == len(xp.unique)): key
                     for key in self.data}
        self.ndim = len(self.axes)

    def __call__(self, parameters):
        self.model.parameters.update(parameters)
        vt_fac = self.model.prob(self.data) * self.vts
        for ii in range(self.ndim):
            vt_fac = trapz(
                vt_fac, self.values[self.axes[self.ndim - ii - 1]], axis=-1)
        return vt_fac
