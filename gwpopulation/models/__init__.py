r"""
Population models for :code:`GWPopulation` follow a specific syntax which is
an extension of the :code:`bilby.hyper.model` syntax.

Functions
---------

The simplest definition is as function that take as input a dictionary of
arrays with sample points named :code:`dataset` and named hyperparmeters
and return an array containing the probability density of the model
evaluated at the sample points.

For example, the following function defines a simple power-law mass model:

.. code-block:: python

    def power_law_mass_model(dataset, alpha, minimum_mass, maximum_mass):
        mass_1 = dataset["mass_1"]
        normalization = (
            (1 - alpha)
            / (maximum_mass ** (1 - alpha) - minimum_mass ** (1 - alpha))
        )
        in_bounds = (mass_1 >= minimum_mass) & (mass_1 <= maximum_mass)
        return mass_1 ** -alpha * normalization * in_bounds

Classes
-------

The more complex definition is as a class that has a :code:`__call__` method
that has the same signature. In this case a :code:`variable_names` attribute
or property can be used to avoid having to explicitly name the hyperparameters.
For example, the following class defines the same power-law mass model:

.. code-block:: python

    class PowerLawMassModel:

        variable_names = ["alpha", "minimum_mass", "maximum_mass"]

        def __call__(self, dataset, **kwargs):
            return power_law_mass_model(
                dataset,
                kwargs["alpha"],
                kwargs["minimum_mass"],
                kwargs["maximum_mass"],
            )

The class definition allows for more complex models to be defined and to cache
values that are expensive to compute and don't depend on the hyperparameters.

Log densities
-------------

The likelihood and selection functions combine models with
:class:`gwpopulation.hyperpe.Model`, which sums the log densities of the
components (a :code:`bilby.hyper.model.Model` is converted to one). By
default this takes the log of each density, which is :code:`-inf` wherever the
density underflows: below :math:`\sim 10^{-308}` in double precision and
:math:`\sim 10^{-38}` in single precision, e.g., far from a narrow peak.
A model can avoid this by providing its log density, which is then used
instead. It is found, in order, as:

- an :code:`ln_prob` method of a class model, with the same signature as
  :code:`__call__`, or an :code:`ln_prob` attribute of a function model;
- a function :code:`ln_<name>` in the module that defines the function
  :code:`<name>`, if it takes the same arguments.

For example, defined next to :code:`power_law_mass_model`, the following is
used for its log density, where :code:`xp` is the array module, e.g.,
:code:`gwpopulation.utils.xp`:

.. code-block:: python

    def ln_power_law_mass_model(dataset, alpha, minimum_mass, maximum_mass):
        mass_1 = dataset["mass_1"]
        ln_normalization = xp.log(
            (1 - alpha)
            / (maximum_mass ** (1 - alpha) - minimum_mass ** (1 - alpha))
        )
        in_bounds = (mass_1 >= minimum_mass) & (mass_1 <= maximum_mass)
        return xp.where(in_bounds, ln_normalization - alpha * xp.log(mass_1), -xp.inf)

A function defined elsewhere can be attached instead, e.g.,
:code:`power_law_mass_model.ln_prob = ln_power_law_mass_model`.

We provide a number of pre-defined models along with standard base models that
can be extended. See the API documentation for more details on implemented models.
"""

from . import mass, redshift, spin
