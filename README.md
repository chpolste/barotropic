# barotropic

A framework for barotropic analysis and modelling of the atmosphere.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7330469.svg)](https://doi.org/10.5281/zenodo.7330469)


## Features

- Integration of the barotropic PV equation on the sphere. 
  Fields are represented on a regular latitude-longitude grid while spatial derivatives are evaluated in spectral space (spherical harmonics).
  The model uses a leapfrog time integration scheme with Robert-Asselin filter, initialized with an Euler-forward step.
- A range of diagnostic functions to investigate properties of the flow and extract features, e.g. Rossby wave packets.
- Predefined plotting functions, initial condition presets and forcing terms.

The package is object-oriented, modular and easy to use.
The implementations are kept simple but sufficiently fast to allow interactive use at moderate spatial and temporal resolutions (e.g. 2.5° spatial resolution with 10 minute timesteps).
A simulation can be set up, run and visualized in under 10 lines of code, e.g.:

```python
import barotropic as bt

grid    = bt.Grid(resolution=2.5)
initial = bt.init.solid_body_rotation(grid, amplitude=15)
forcing = bt.rhs.GaussianMountain(center=(30, 45), stdev=(10, 10), height=2000)
model   = bt.BarotropicModel(forcing, diffusion_order=2)

last, all_states = model.run(initial, 15*bt.MIN, 10*bt.DAY, save_every=6*bt.HOUR)

last.plot.summary()
```

![example of the summary plot preset](examples/example-summary-plot.png)

Consult the [online documentation](https://chpolste.github.io/barotropic) to learn more about features and how to use the package.


## Installation

barotropic is available on PyPI:

    $ pip install barotropic

Dependencies:

- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [pyspharm](https://github.com/jswhit/pyspharm) ⚠️ 
- [matplotlib](https://github.com/matplotlib/matplotlib) (optional)
- [PyWavelets](https://github.com/PyWavelets/pywt) (optional)
- [falwa](https://github.com/csyhuang/hn2016_falwa) (optional; hn2016_falwa until 3.0.x)

Because the latest version of pyspharm (1.0.9, last accessed in September 2024) on PyPI is broken, no dependency on Pypharm is declared, but the package must be installed.
Please install the package manually from its GitHub repository after the installing barotropic with

    $ pip install git+https://github.com/jswhit/pyspharm


## License

Copyright 2019-2024 Christopher Polster

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

