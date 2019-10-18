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
- [pywt](https://github.com/PyWavelets/pywt) (1.1 or newer)
- [hn2016_falwa](https://github.com/csyhuang/hn2016_falwa)

These are imported on demand and not required to just run the model.


## Usage

Consult the docstrings and the provided [examples](examples).

