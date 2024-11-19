FAQ
---

.. grid::

    .. grid-item-card:: I found a typo in the documentation, is it worth fixing?

        Yes! Any help in improving the documentation is greatly appreciated. If you find a typo, please open a pull request with the fix.

.. grid::

    .. grid-item-card:: I think a significant feature is missing from the code, e.g., a way to estimate the likelihood, what should I do?

        If you think a significant feature is missing from the code, please check the `open issues <https://github.com/ColmTalbot/GWPopulation/issues>`_
        to see if it is already being discussed. If it is not, please open an `issue <https://github.com/ColmTalbot/GWPopulation/issues/new>`_
        describing the feature and any suggestions on implementation.

.. grid::

    ..  grid-item-card:: I think there is a bug in the code, what should I do?

        If you think you have found a bug in the code, please check the `open issues <https://github.com/ColmTalbot/GWPopulation/issues>`_
        to see if it is already being discussed. If it is not, please open an `issue <https://github.com/ColmTalbot/GWPopulation/issues/new>`_
        describing the bug. If you have a fix for the bug, please open a pull request with the fix.

.. grid::

    ..  grid-item-card:: I have a new model that I would like to share with the rest of the community, how can I do that?

        Rather than attempting to support a full curated model zoo, we recommend that you create a new repository
        and open an issue to add a link to it in the `External Examples` section of the documentation.
        If you're new to creating packages, you can use this `template repository <https://github.com/ColmTalbot/gwpopulation-additional-models>`_.

.. grid::

    ..  grid-item-card:: I would like to add a new backend to :code:`GWPopulation`, how can I do that?

        If you would like to add a new backend to :code:`GWPopulation`, please please check the `open issues <https://github.com/ColmTalbot/GWPopulation/issues>`_
        to see if it is already being discussed. If it is not, please open an `issue <https://github.com/ColmTalbot/GWPopulation/issues/new>`_
        to begin discussion. Features to consider are:

        - Does the new backend support all of the functionality needed for :code:`GWPopulation`?
        - If not, how much work would be required to add/support this functionality?

        One crucial capability is double precision support, for example, at the time of writing (June 2024) :code:`MLx` does not support
        double precision and so is not compatible with :code:`GWPopulation`.

.. grid::

    ..  grid-item-card:: I have a question that isn't answered here, what should I do?

        If you have a question that isn't answered in the documentation, please please check the `issues <https://github.com/ColmTalbot/GWPopulation/issues>`_
        to see if it is already being discussed. If it is not, please open an `issue <https://github.com/ColmTalbot/GWPopulation/issues/new>`_