[![GitHub release (with filter)](https://img.shields.io/github/v/release/syp2001/spheroidal)](https://github.com/syp2001/spheroidal/releases)
[![Test Status](https://github.com/syp2001/spheroidal/actions/workflows/python-publish.yml/badge.svg)](https://github.com/syp2001/spheroidal/actions)
[![PyPI - Version](https://img.shields.io/pypi/v/spheroidal)](https://pypi.org/project/spheroidal/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/spheroidal.svg)](https://anaconda.org/conda-forge/spheroidal)
[![Documentation Status](https://readthedocs.org/projects/spheroidal/badge/?version=latest)](https://spheroidal.readthedocs.io/en/latest/?badge=latest)
[![GitHub License](https://img.shields.io/github/license/syp2001/spheroidal)](https://github.com/syp2001/spheroidal/blob/main/LICENSE)

# Spheroidal

`spheroidal` is a python library for computing spin weighted spheroidal
harmonics along with their eigenvalues and derivatives. It supports both integer
and half integer spin weights. The library provides implementations of the 
spherical expansion method described in [(Hughes, 2000)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.61.084004)
and the continued fraction method from [(Leaver, 1985)](https://www.edleaver.com/Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf)
Also included is code for computing spin weighted spherical harmonics and 
spherical-spheroidal mixing coefficients. See the [documenation](https://spheroidal.readthedocs.io)
for more information.

## Installation

```bash
   conda install -c conda-forge spheroidal
```

or using pip

```bash
   pip install spheroidal
```

## Basic Usage

```python
   import spheroidal
   from math import pi

   # Compute the spin weighted spheroidal eigenvalue with s = -2, l = 2, m = 2, gamma = 1.5
   spheroidal.eigenvalue(-2, 2, 2, 1.5)
   # -5.5776273646788255

   # Compute the corresponding spin weighted spheroidal harmonic
   S = spheroidal.harmonic(-2, 2, 2, 1.5)

   # Evaluate at theta = pi/2, phi = 0
   S(pi/2, 0)
   # (0.06692950919170575+0j)

   # Compute the derivative wrt theta at the same point
   spheroidal.harmonic_deriv(-2, 2, 2, 1.5)(pi/2, 0)
   # (-0.20852146386265577+0j)
```

## Authors

* Seyong Park
* Zach Nasipak