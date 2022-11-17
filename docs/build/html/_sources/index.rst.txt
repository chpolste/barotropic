Barotropic
==========

A framework for barotropic analysis and modelling of the atmosphere.

.. image:: https://zenodo.org/badge/213849638.svg
   :target: https://zenodo.org/badge/latestdoi/213849638


Features
========

- Integration of the barotropic PV equation on the sphere. 
  Fields are represented on a regular latitude-longitude grid while spatial derivatives are evaluated in spectral space (spherical harmonics).
  The model uses a leapfrog time integration scheme with Robert-Asselin filter, initialized with an Euler-forward step.
- A range of diagnostic functions to investigate properties of the flow and extract features, e.g. Rossby wave packets.
- Predefined plotting functions, initial condition presets and forcing terms.

The package is object-oriented, modular and easy to use.
The implementations are kept simple but sufficiently fast to allow interactive use at moderate spatial and temporal resolutions (e.g. 2.5° spatial resolution with 10 minute timesteps).
A simulation can be set up, run and visualized in under 10 lines of code, e.g.:

.. code::
   python

   import barotropic as bt
   
   grid    = bt.Grid(resolution=2.5)
   initial = bt.init.solid_body_rotation(grid, amplitude=15)
   forcing = bt.rhs.GaussianMountain(center=(30, 45), stdev=(10, 10), height=2000)
   model   = bt.BarotropicModel(forcing, diffusion_order=2)
   
   last, all_states = model.run(initial, 15*bt.MIN, 10*bt.DAY, save_every=6*bt.HOUR)
   
   last.plot.summary()



.. image::
   examples/example-summary-plot.png



Installation
============

Get the package from PyPI:

.. code::
    bash

    $ pip install barotropic

Alternatively, grab the source from GitHub: https://github.com/chpolste/barotropic.

Dependencies:

- :py:mod:`numpy`
- :py:mod:`scipy`
- :py:mod:`spharm`
- :py:mod:`matplotlib` (optional)
- :py:mod:`pywt` (optional)
- :py:mod:`hn2016_falwa` (optional)


Package Documentation
=====================

.. toctree::
   :maxdepth: 1

   grid
   state
   model
   init
   rhs
   plot
   diagnostics
   io
   tips
   references


License
=======

Copyright 2019-2022 Christopher Polster

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

