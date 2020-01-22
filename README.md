[![Build Status](https://travis-ci.com/ColmTalbot/gwpopulation.svg?branch=master)](https://travis-ci.com/ColmTalbot/gwpopulation)
[![Maintainability](https://api.codeclimate.com/v1/badges/579536603e8e06466e63/maintainability)](https://codeclimate.com/github/ColmTalbot/gwpopulation/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/579536603e8e06466e63/test_coverage)](https://codeclimate.com/github/ColmTalbot/gwpopulation/test_coverage) [![Versions](https://img.shields.io/pypi/pyversions/gwpopulation.svg)](https://pypi.org/project/gwpopulation/)

A collection of parametric binary black hole mass/spin population models.

These are formatted to be consistent with the [Bilby](https://git.ligo.org/lscsoft/bilby) [hyper-parameter inference package](https://lscsoft.docs.ligo.org/bilby/hyperparameters.html).

For an example using this code to analyse the first gravitational-wave transient catalog (GWTC-1) see [here](https://colab.research.google.com/github/ColmTalbot/gwpopulation/blob/master/examples/GWTC1.ipynb).

Most of the models implemented are derived from models presented in one of:
- [Talbot & Thrane (2017)](https://arxiv.org/abs/1704.08370)
- [Talbot & Thrane (2018)](https://arxiv.org/abs/1801.02699)
- [Wysocki et al. (2018)](https://arxiv.org/abs/1805.06442)
- [Fishbach et al. (2019)](https://arxiv.org/abs/1805.10270)

Automatically generated docs can be found [here](https://colmtalbot.github.io/gwpopulation/).

As of v0.5.0 `GWPopulation` supports only `python >= 3.6`.
See [the python 3 statement](https://python3statement.org/) for more information.
