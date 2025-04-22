============
Installation
============

.. tabs::

   .. tab:: Conda

      .. code-block:: console

         $ conda install -c conda-forge gwpopulation

   .. tab:: Pip
      
      .. code-block:: console

         $ pip install gwpopulation

   .. tab:: Latest

      If you want to use the latest (unreleased) version of :code:`gwpopulation` you can
      install the current :code:`main` branch directly from :code:`GitHub`.

      .. code-block:: console

         $ pip install git+https://github.com/ColmTalbot/gwpopulation.git@main

      .. warning::

         While :code:`GWPopulation` has an extensive unit test suite, :code:`main` is
         more likely to contain bugs than released versions, especially when using GPUs.

   .. tab:: Development

      A development version of :code:`gwpopulation` can be installed from the
      source code in the usual way. Assuming a working python installation exists,
      this can be done by cloning the :code:`gwpopulation` repository and installing
      using :code:`pip`.

      .. code-block:: console

         $ git clone git@github.com:ColmTalbot/gwpopulation.git
         $ cd gwpopulation/
         $ pip install -e .

      .. note::
      
         Installing with :code:`-e` will produce an
         `editable installation <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_.
         Additional backends can also be specified via editable installations using
         :code:`pip install -e .[jax]`.

Supported python versions: 3.10-3.13.
This will install :code:`gwpopulation` and all required dependencies (e.g., :code:`bilby`)
if they are not already available in your system.

Additional backends
-------------------

The main power of :code:`gwpopulation` is the ability to use non-:code:`numpy` backends
for array operations.
If installing using :code:`pip` (in any method), the optional requirements for specify backends can
be installed by specifying, e.g.,

.. code-block:: console

   $ pip install gwpopulation[jax]
