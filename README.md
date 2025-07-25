<h1 align="center">
<img src="https://raw.githubusercontent.com/ColmTalbot/gwpopulation/main/docs/_static/logo-long.svg">
</h1>

------------------------------------------------------------------------------

[![DOI](https://joss.theoj.org/papers/10.21105/joss.07753/status.svg)](https://doi.org/10.21105/joss.07753)
[![Python package](https://github.com/ColmTalbot/gwpopulation/actions/workflows/python-package.yml/badge.svg)](https://github.com/ColmTalbot/gwpopulation/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/ColmTalbot/gwpopulation/branch/master/graph/badge.svg?token=4K4V0HRDMI)](https://codecov.io/gh/ColmTalbot/gwpopulation)
[![Versions](https://img.shields.io/pypi/pyversions/gwpopulation.svg)](https://pypi.org/project/gwpopulation/)
![Conda Downloads](https://img.shields.io/conda/d/conda-forge/gwpopulation)

Flexible, extensible, hardware-agnostic gravitational-wave population inference.

- [Documentation](https://ColmTalbot.github.io/gwpopulation)
- [Source Repository](https://github.com/ColmTalbot/GWPopulation)
- [Issues](https://github.com/ColmTalbot/GWPopulation/issues)
- [Contributing](https://colmtalbot.github.io/gwpopulation/contributing/index.html)

It provides:

- Simple use of GPU-acceleration via [JAX](https://jax.readthedocs.io/en/latest/) and [cupy](https://cupy.dev/).
- Implementations of widely used likelihood compatible with [Bilby](https://docs.ligo.org/lscsoft/bilby).
- A standard format for defining new population models.
- A collection of standard population models.

If you're using this on high-performance computing clusters, you may be interested in the associated pipeline code [gwpopulation_pipe](https://docs.ligo.org/RatesAndPopulations/gwpopulation_pipe/).

#### Attribution

------------------------------------------------------------------------------

Please cite [Talbot _et al._ (2025)](https://doi.org/10.21105/joss.07753) if you use `GWPopulation` in your research.

```bibtex
@article{Talbot2025,
  author = {Colm Talbot and Amanda Farah and Shanika Galaudage and Jacob Golomb and Hui Tong},
  title = {GWPopulation: Hardware agnostic population inference for compact binaries and beyond},
  journal = {Journal of Open Source Software},
  doi = {10.21105/joss.07753},
  url = {https://doi.org/10.21105/joss.07753},
  year = {2025},
  publisher = {The Open Journal},
  volume = {10},
  number = {109},
  pages = {7753},
  archivePrefix = {arXiv},
  eprint = {2409.14143},
  primaryClass = {astro-ph.IM},
}
```

The older citation can also be included for the initial proof-of-principle for the application of hardware acceleration [Talbot _et al._ (2019)](https://doi.org/10.1103/PhysRevD.100.043030).

```bibtex
@ARTICLE{2019PhRvD.100d3030T,
  author = {{Talbot}, Colm and {Smith}, Rory and {Thrane}, Eric and {Poole}, Gregory B.},
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
