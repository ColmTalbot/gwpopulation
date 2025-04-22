============
Installation
============

.. tabs::
   .. tab:: Conda

      .. code-block:: console
          $ conda install -c conda-forge gwpopulation
      
      Supported python versions: 3.10-3.13.

   .. tab:: Pip
      
      .. code-block:: console
          $ pip install gwpopulation
      
      Supported python versions: 3.10-3.13.

This will install the latest stable release of :code:`gwpopulation` into your
python package directory, including all required dependencies (e.g.,, :code:`bilby`
and :code:`dynesty`) if they are not already available in your system. Methods from
:code:`gwpopulation` should then be accessible for import into your program file.

Additional backends
-------------------

The main power of :code:`gwpopulation` is the ability to use non-:code:`numpy` backends
for array operations.
If installing using :code:`pip`, the optional requirements for specify backends can be installed
by specifying, e.g.,

.. code-block:: console

   $ pip install gwpopulation[jax]

:code:`gwpopulation` source installation
----------------------------------------

A development version of :code:`gwpopulation` can be installed from the source code
in the usual way.
Assuming a working python installation exists, this can be done by cloning the
:code:`gwpopulation` repository and installing using :code:`pip`.

.. code-block:: console

   $ git clone git@github.com:ColmTalbot/gwpopulation.git
   $ cd gwpopulation/
   $ pip install -e .

.. note::

   Installing with `-e` will produce an `editable installation <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_.
   Additional backends can also be specified via editable installations using :code:`pip install -e .[jax]`.

Once the installation is finished, you can check that the installation proceeded
successfully by running one of the examples outlined on the Examples page.
