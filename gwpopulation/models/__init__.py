"""
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

The more complex definition is as a class that has an :code:`__init__` method
that has the same signature. In this case a :code:`variable_names` attribute
or property can be used to avoid having to explicitly name the hyperparameters.
For example, the following class defines the same power-law mass model:

.. code-block:: python

    class PowerLawMassModel:

        variable_names = ["alpha", "minimum_mass", "maximum_mass"]

        def __init__(self, dataset, **kwargs):
            return power_law_mass_model(
                dataset,
                kwargs["alpha"],
                kwargs["minimum_mass"],
                kwargs["maximum_mass"],
            )

The class definition allows for more complex models to be defined and to cache
values that are expensive to compute and don't depend on the hyperparameters.

We provide a number of pre-defined models along with standard base models that
can be extended. See the API documentation for more details on implemented models.
"""

from . import mass, redshift, spin
