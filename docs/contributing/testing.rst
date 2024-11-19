Testing
-------

Testing is essential for any new contribution to :code:`GWPopulation`. We
have a suite of unit-tests that run on every push to the repository
and the example notebooks are run occasionally to ensure that they
are still working.

Unit Testing
~~~~~~~~~~~~

We use `pytest <https://docs.pytest.org/en/7.1.x/contents.html>`_ to run
our unit tests. These tests are located in the :code:`tests` directory and
are run automatically on every push to the repository.
Change will only be accepted without passing unit tests under very rare.
circumstances. To run the tests locally, you can use the following commands:

.. code-block:: bash

    python -m pip install .[test]
    python -m pytest --cov gwpopulation

If you are new to writing unit tests, you should begin with the existing
tests and see how they can be modified for your purposes.
Additionally, the :code:`pytest` website includes information on how to write
and using that framework.
If you have any additional questions, please open the pull request and ask
for assistance.

Testing with multiple backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since we support multiple backends, we have to test that the code works
with as many of them as possible. Unfortunately, we do not currently have
access to CI machines that have GPUs and so tests are performed with :code:`numpy`
and :code:`jax-cpu` backends. If you have access to an environment with a suitable
GPU we request that you run the tests with the :code:`cupy` and :code:`jax-gpu`
backends also and report the results in the pull request.
If you don't have access, you can request a maintainer runs the tests on a GPU.

Integration Testing
~~~~~~~~~~~~~~~~~~~

We do not have a formal process for full-scale integration testing.
However, when contributing a new feature, it is recommended to test
this feature in a real-world scenario. This can be done by creating
a modified version of one of the example notebooks. The result can
then either be added to the repository in your pull request, or linked
so that we have a record of the test. For testing in a GPU environment,
you can use `Google Colab <https://colab.research.google.com/>`_.