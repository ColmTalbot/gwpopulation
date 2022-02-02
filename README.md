[![Python package](https://github.com/ColmTalbot/gwpopulation/actions/workflows/python-package.yml/badge.svg)](https://github.com/ColmTalbot/gwpopulation/actions/workflows/python-package.yml)
[![GitHub Pages](https://github.com/ColmTalbot/gwpopulation/actions/workflows/pages.yml/badge.svg)](https://github.com/ColmTalbot/gwpopulation/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/ColmTalbot/gwpopulation/branch/master/graph/badge.svg?token=4K4V0HRDMI)](https://codecov.io/gh/ColmTalbot/gwpopulation)
[![Maintainability](https://api.codeclimate.com/v1/badges/579536603e8e06466e63/maintainability)](https://codeclimate.com/github/ColmTalbot/gwpopulation/maintainability)
[![Versions](https://img.shields.io/pypi/pyversions/gwpopulation.svg)](https://pypi.org/project/gwpopulation/)

# GWPopulation

A collection of parametric binary black hole mass/spin population models.

These are formatted to be consistent with the [Bilby](https://git.ligo.org/lscsoft/bilby) [hyper-parameter inference package](https://lscsoft.docs.ligo.org/bilby/hyperparameters.html).

For an example using this code to analyse the first gravitational-wave transient catalog (GWTC-1) see [here](https://colab.research.google.com/github/ColmTalbot/gwpopulation/blob/master/examples/GWTC1.ipynb).

Automatically generated docs can be found [here](https://colmtalbot.github.io/gwpopulation/).

If you're using this for production analyses, you may be interested in the associated pipeline code [gwpopulation_pipe](https://docs.ligo.org/RatesAndPopulations/gwpopulation_pipe/).

# Attribution

Please cite [Talbot _et al_ (2019)](https://doi.org/10.1103/PhysRevD.100.043030) if you find this package useful.

```
@ARTICLE{2019PhRvD.100d3030T,
       author = {{Talbot}, Colm and {Smith}, Rory and {Thrane}, Eric and
         {Poole}, Gregory B.},
        title = "{Parallelized inference for gravitational-wave astronomy}",
      journal = {\prd},
         year = 2019,
        month = aug,
       volume = {100},
       number = {4},
          eid = {043030},
        pages = {043030},
          doi = {10.1103/PhysRevD.100.043030},
archivePrefix = {arXiv},
       eprint = {1904.02863},
 primaryClass = {astro-ph.IM},
}
```

Additionally, please consider citing the original references for the implemented models which should be include in docstrings.

Most of the models implemented are derived from models presented in one of:
- [Talbot & Thrane (2017)](https://arxiv.org/abs/1704.08370)
- [Talbot & Thrane (2018)](https://arxiv.org/abs/1801.02699)
- [Wysocki et al. (2018)](https://arxiv.org/abs/1805.06442)
- [Fishbach et al. (2019)](https://arxiv.org/abs/1805.10270)
