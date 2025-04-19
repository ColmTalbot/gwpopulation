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

This will install the latest stable release of `gwpopulation` into your
python package directory, including all required dependencies (example, `bilby`
and `dynesty`) if they are not already available in your system. Methods from
`gwpopulation` should then be accessible for import into your program file.

`gwpopulation` source installation
--------------------------------

Since updates and bug corrections to `gwpopulation` are being constantly made,
it might be a good idea to install the latest available version from the source.
Assuming a working python installation exists, this can be done by cloning the
`gwpopulation` repository, installing the requirements and then installing the
main software.

.. code-block:: console

   $ git clone git@github.com:ColmTalbot/gwpopulation.git
   $ cd gwpopulation/
   $ pip install .

Once the installation is finished, you can check that the installation proceeded
successfully by running one of the examples outlined on the Examples page.


