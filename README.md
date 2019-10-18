# barotropic

A barotropic model on the sphere.

Integrates the barotropic PV equation using a pseudo-spectral spatial discretization and a leapfrog time-stepping scheme.
Includes initial condition and forcing presets as well as some diagnostic and plot functions.
The code is kept relatively simple but is still fast enough for interactive use at common spatial and temporal resolutions.


## Dependencies

- [numpy](https://github.com/numpy/numpy)
- [pyspharm](https://github.com/jswhit/pyspharm)

Additional dependencies for plotting and some diagnostics:

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


## Usage

Consult the docstrings and the provided [examples](examples).

