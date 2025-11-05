---
title: 'GWPopulation: Hardware agnostic population inference for compact binaries and beyond'
tags:
  - Python
  - astronomy
  - gravitational waves
  - hierarchical inference
authors:
  -  name: Colm Talbot
     orcid: 0000-0003-2053-5582
     affiliation: 1
  -  name: Amanda Farah
     orcid: 0000-0002-6121-0285
     affiliation: 2
  -  name: Shanika Galaudage
     orcid: 0000-0002-1819-0215
     affiliation: 3, 4
  -  name: Jacob Golomb
     orcid: 0000-0002-6977-670X
     affiliation: 5, 6
  -  name: Hui Tong
     orcid: 0000-0002-4534-0485
     affiliation: 7, 8
affiliations:
  -  name: Kavli Institute for Cosmological Physics, University of Chicago, Chicago, IL 60637, USA
     index: 1
  -  name: Department of Physics, University of Chicago, Chicago, IL 60637, USA
     index: 2
  -  name: Laboratoire Lagrange, Université Côte d'Azur, Observatoire de la Côte d'Azur, CNRS, Bd de l'Observatoire, 06300, France
     index: 3
  -  name: Laboratoire Artemis, Université Côte d'Azur, Observatoire de la Côte d'Azur, CNRS, Bd de l'Observatoire, 06300, France
     index: 4
  -  name: Department of Physics, California Institute of Technology, Pasadena, CA 91125, USA
     index: 5
  -  name: LIGO Laboratory, California Institute of Technology, Pasadena, CA 91125, USA
     index: 6
  -  name: School of Physics and Astronomy, Monash University, VIC 3800, Australia
     index: 7
  -  name: "OzGrav: The ARC Centre of Excellence for Gravitational Wave Discovery, Clayton VIC 3800, Australia"
     index: 8
date: ---
bibliography: paper.bib

---

# Summary

Since the first direct detection of gravitational waves by the LIGO--Virgo collaboration in 2015 [@GW150914], the size of the gravitational-wave transient catalog has grown to nearly 100 events [R. Abbott et al., -@GWTC3], with the ongoing fourth observing run more than doubling the total number.
Extracting astrophysical or cosmological information from these observations is a hierarchical Bayesian inference problem.
`GWPopulation` is designed to provide simple-to-use, robust, and extensible tools for hierarchical inference in gravitational-wave astronomy or cosmology. It has been widely adopted for gravitational-wave astronomy, including producing flagship results for the LIGO-Virgo-KAGRA collaborations [R. Abbott et al., -@GWTC3Pop; @GW230529][^1].
While designed to work with observations of compact binary coalescences, `GWPopulation` may be available to a wider range of hierarchical Bayesian inference problems.

[^1]: For a full listing of papers using `GWPopulation`, see the [citations for the previous publication](https://ui.adsabs.harvard.edu/abs/2019PhRvD.100d3030T/citations).

Building on Bilby [@Bilby], `GWPopulation` can easily be used with a range of stochastic samplers through a standard interface.
By providing access to a range of array backends [currently NumPy, @numpy; JAX, @jax; and CuPy, @cupy], `GWPopulation` is hardware agnostic and can leverage hardware acceleration to meet the growing computational needs of these analyses.
The package includes:

- Implementations of the most commonly used likelihood functions in the field.
- Commonly used models for describing the astrophysical population of merging compact binaries,
  including the “PowerLaw+Peak” and “PowerLaw+Spline” mass models, “Default” spin model, and “PowerLaw” redshift models used in the latest LIGO-Virgo-KAGRA collaboration analysis of the compact binary population.[^2]
- Functionality to simultaneously infer the astrophysical distribution of sources and cosmic expansion history using the "spectral siren" method [@Ezquiaga2022].
- A standard specification allowing users to define additional models.

[^2]: See R. Abbott et al. [-@GWTC3Pop] for details of these models.

# Statement of need

Hierarchical Bayesian inference is the standard method for inferring parameters describing the astrophysical population of compact binaries and the cosmic expansion history [e.g., @Thrane2019; @Vitale2022].
The first step in the hierarchical inference process is drawing samples from the posterior distributions for the source parameters of each event under a fiducial prior distribution along with a set of simulated signals used to quantify the sensitivity of gravitational-wave searches.
Next, these samples are used to estimate the population likelihood using Monte Carlo integration with a computational cost that grows quadratically with the size of the observed population.
Since evaluating these Monte Carlo integrals is embarrassingly parallel, this is a prime candidate for hardware acceleration using graphics or tensor processing units.
`GWPopulation` provides functionality needed to perform this second step and is extensively used by members of the gravitational-wave astronomy community including the LIGO-Virgo-KAGRA collaborations.

Maximizing the information we can extract from the gravitational-wave transient catalog requires a framework where potential population models can be quickly constrained with the observed data with minimal boilerplate code.
Additionally, the availability of a standard open-source implementation improves the reliability and reproducibility of published results.
`GWPopulation` addresses all of these points by providing an open-source implementation of the functionality needed to perform population analyses while enabling user-defined models to be provided by a `Python` function/class definition.
The flexible backend system means hardware acceleration can be used with minimal coding effort.
Using `GWPopulation` on Google Colab, it is possible to perform an exploratory analysis with a new population model in minutes and produce production-quality results without needing high-performance or high-throughput computing clusters.
With access to high-throughput computing resources, a wide range of potential models can be easily explored using the associated `gwpopulation_pipe` [@gwpop_pipe] package.

# Related packages

Several other packages are actively used and maintained in the community that can be used for population inference that operate in complementary ways to `GWPopulation`.

- GWInferno [@gwinferno] is a package for hierarchical inference with gravitational-wave sources intended for use with NumPyro [@NumPyro] targeting high-dimensional models. GWInferno includes many population models initially adapted from `GWPopulation`.
- There is a wide range of packages designed for joint astrophysical and cosmological inference with gravitational-wave transients including `icarogw` [@icarogw], `gwcosmo` [@gwcosmo], `MGCosmoPop` [@mgcosmo], and `CHIMERA` [@chimera]. `icarogw` supports some hardware acceleration using CuPy but some cosmological calculations are limited to CPU support only. `chimera` is JAX-compatible and supports flat Lambda-CDM cosmologies along with analysis using galaxy catalogs.
- `vamana` [@vamana] models the compact binary distribution as a mixture of Gaussians and power-law distributions, and `popmodels` [@popmodels] implements a range of parametric models for the compact binary distribution and supports sampling via emcee [@emcee]. However, neither supports hardware acceleration at the time of writing.

# Acknowledgements

CT is supported by an Eric and Wendy Schmidt AI in Science Fellowship, a Schmidt Sciences program.
SG is supported by the ANR COSMERGE project, grant ANR-20-CE31-001 of the French Agence Nationale de la Recherche.
AF is supported by the NSF Research Traineeship program under grant No. DGE1735359, and by the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE-1746045.
HT is supported by Australian Research Council (ARC) Centre of Excellence CE230100016.

# References
