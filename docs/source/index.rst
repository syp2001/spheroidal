.. spheroidal documentation master file, created by
   sphinx-quickstart on Tue Nov  7 14:34:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========

.. |github-link| replace:: :code:`spheroidal`

.. _github-link: https://github.com/syp2001/spheroidal

|github-link|_ is a python library for computing spin weighted spheroidal
harmonics along with their eigenvalues and derivatives. It supports both integer
and half integer spin weights. The library provides implementations of the 
spherical expansion method described in `(Hughes, 2000) 
<https://journals.aps.org/prd/abstract/10.1103/PhysRevD.61.084004>`_
and the continued fraction method from `(Leaver, 1985) <https://www.edleaver.com/
Misc/EdLeaver/Publications/AnalyticRepresentationForQuasinormalModesOfKerrBlackHoles.pdf>`_.
Also included is code for computing spin weighted spherical harmonics and 
spherical-spheroidal mixing coefficients. Some example code for visualizing and
animating the harmonics is available on the `Visualization <notebooks/Visualization.html>`_ page.

.. image:: images/s-2.gif
   :align: center
   :width: 100%

Installation
------------
Install using Anaconda

.. code-block:: bash

   conda install -c conda-forge spheroidal

or using pip

.. code-block:: bash

   pip install spheroidal

Basic Usage
-----------

.. code-block:: python

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

.. toctree::
   :maxdepth: 2
   :caption: Contents

   notebooks/Background
   notebooks/Getting Started
   notebooks/Visualization

.. _API Reference:

API Reference
-------------

.. autosummary::
   :toctree: _autosummary
   :caption: API Reference
   :template: custom-module-template.rst
   :recursive:
   
   ~spheroidal.wrapper.eigenvalue
   ~spheroidal.wrapper.harmonic
   ~spheroidal.wrapper.harmonic_deriv
   ~spheroidal.spherical
   ~spheroidal.leaver
   ~spheroidal.spectral

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
