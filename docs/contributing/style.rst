Code Style
----------

We have a number of linting rules for this repository which are enforced as part of CI testing.
The most notable is that we follow `black <https://black.readthedocs.io/en/stable/>`_ code style.
Please run the pre-commit checks before opening a pull request.
The simplest way to do this is to install the pre-commit package and run it on the repository:

.. code-block:: console

    mamba install pre-commit
    pre-commit install

.. note::

    :code:`pip` or :code:`conda` can be used instead of :code:`mamba` if you prefer.

This will ensure that the tests run before every commit.
If you want to additionally run the checks manually, you can do so with:

.. code-block:: bash

    pre-commit run --all-files