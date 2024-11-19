"""
When adding significant new functionality to the package, it is useful to have a place to test and develop.
This subpackage contains experimental new features and models that do not have a finalized API.
Any code in this subpackage is subject to change without warning in subsequent versions.

Current experimental features include:

- cosmology functionality using `wcosmo <https://wcosmo.readthedocs.io>`_.
- :code:`JAX` support for JIT-compiling population models and likelihoods.
- a :code:`numpyro` compatible definition of the :func:`gwpopulation.hyperpe.HyperparameterLikelihood`.
"""
