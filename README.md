# barotropic

Integrate the barotropic PV equation on the sphere.
Fields are represented on a regular latitude-longitude grid while spatial derivatives are evaluated in spectral space (spherical harmonics).
The model uses a leapfrog time integration scheme with Robert-Asselin filter, initialized with an Euler-forward step.


## Features

- Initial condition and forcing presets for atmospheric science experiments
- A range of diagnostic functions to investigate properties of the flow
- Predefined plotting functions

The model code is object-oriented, modular and easy to use.
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

- Consult the [online documentation](https://chpolste.github.io/barotropic/docs/barotropic) to learn about the public interface.
- A few [example notebooks](examples) are provided in which the module is used to reproduce results from research papers.


## Dependencies

- [numpy](https://github.com/numpy/numpy)
- [pyspharm](https://github.com/jswhit/pyspharm)

Additional dependencies for plotting, diagnostics, etc.:

- [matplotlib](https://github.com/matplotlib/matplotlib)
- [scipy](https://github.com/scipy/scipy)
- [PyWavelets](https://github.com/PyWavelets/pywt) (>= 1.1)
- [hn2016_falwa](https://github.com/csyhuang/hn2016_falwa)

These are imported on demand and not required to just run the model.


## Installation

Installation as a package requires [setuptools](https://pypi.org/project/setuptools/).
From the root directory of the repository run

    pip install .

Optional dependencies are specified as an [extra](https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies) and can be turned into required dependencies by installing with

    pip install .[with-optional]


## License

Copyright 2019-2020 Christopher Polster

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

