Release Notes
=============

.. py:module:: barotropic


Barotropic 3.0.1
----------------

Patch release, 28 Mar 2022.

- Fix planetary vorticity in summary plot.


Barotropic 3.0
--------------

Major release, 9 Feb 2022.

.. note::
    This version is not backwards-compatible with v2.x.

New features:

- Add :py:meth:`Grid.mean`.
- Add :py:attr:`Grid.region` and :py:class:`grid.GridRegion`.
- Add :py:meth:`Grid.filter_meridional` and :py:meth:`Grid.filter_zonal`.
- Add :py:meth:`Grid.laplace` and :py:meth:`Grid.laplace_spectral`.
- Add :py:meth:`Grid.solve_poisson` and :py:meth:`Grid.solve_poisson_spectral`.
- Add :py:meth:`Grid.solve_diffusion` and :py:meth:`Grid.solve_diffusion_spectral`.
- Allow for sectoral zonalization with :py:meth:`Grid.zonalize`.
- Add :py:meth:`State.as_hn2016` and :py:meth:`State.from_hn2016`.
- Add informative `__repr__` methods to some classes.
- Add :py:func:`diagnostics.dominant_wavenumber_fourier`.

Changes:

- Make longitude the first dimension in all user-facing code, following plotting convention.
- Make `resolution` the first argument of the :py:class:`Grid` constructor.
- Rename :py:meth:`Grid.ddphi` to :py:meth:`Grid.derivative_meridional`.
- Rename :py:meth:`Grid.zonalize_eqlat` to :py:meth:`Grid.zonalize`.
- Fix :py:meth:`Grid.quad_boxcount`.
- Merge quadrature functions into :py:meth:`Grid.quad`.
- Add :py:attr:`Grid.fcor` and rename old `fcor` to `fcor2`.
- Perform more :py:class:`State` computations in spectral space to improve accuracy.
- Always send PV field through spectral space to ensure consistency in :py:class:`State` constructor.
- Rename attributes of :py:class:`State`: `lats` to `lat`, `lons` to `lon`, `lat` to `lat2`, `lon` to `lon2`
- Renamed :py:attr:`State.v_envelope_hilbert`.
- Remove :py:attr:`State.dominant_wavenumber`.
- Switch to much faster Fourier-based dominant wavenumber computation in :py:class:`State.falwa_filtered`.
- Fix double diffusion step in :py:meth:`BarotropicModel.euler`.
- Rename :py:mod:`barotropic.diagnostics` module and some function therein.
- Arguments of :py:func:`diagnostics.envelope_hilbert` now match :py:func:`diagnostics.dominant_wavenumber_fourier`.
- Remove :py:func:`diagnostics.falwa_hn2016` and :py:meth:`State.falwa_hn2016`.
- Reimplement interpolation in :py:func:`diagnostics.falwa` and :py:func:`diagnostics.fawa`.
- The wavenumber in :py:func:`diagnostics.filter_by_wavenumber` is now determined by FWHM.
  Previously the wavelength corresponding to the given wavenumber determined the full width of the window.
  The change aligns the implementation with the original intention of filtering like Ghinassi et al. (`2018 <https://doi.org/10.1175/MWR-D-18-0068.1>`_, `2020 <https://doi.org/10.1175/JAS-D-19-0149.1>`_).
  Things worked out before because the dominant wavenumber extracted from the meridional wind is half that of the corresponding FALWA field so the filter widths were sized as intended.
  Now the v-based wavenumber is doubled in :py:attr:`State.falwa_filtered` before it is given to :py:func:`diagnostics.filter_by_wavenumber` to compensate for the change.
- :py:func:`diagnostics.filter_by_wavenumber` now accepts zonally symmetric wavenumber fields as 1-dimensional nlat-sized arrays and is faster as it omits computation of smoothed fields that are not required.

Package and documentation:

- Move unit tests to `pytest`.
- Move documentation to `sphinx`.
- Add `scipy` as a non-optional dependency.
- General improvements to documentation.
- New example notebook that demonstrates wavenumber-based filtering.


Barotropic 2.0.1
----------------

Patch release, 12 Jul 2020.

- Fix import of `BarotropicField` from `hn2016_falwa <https://github.com/csyhuang/hn2016_falwa>`_.
- Declare the public interface explicitly with `__all__` in the `__init__.py`.
- Improve consistency of docstrings and add missing docstrings.
- Hide imported constants in submodules.
- :py:class:`rhs.RHSSum` and :py:class:`rhs.RHSProduct` have been renamed and are now hidden.
- :py:func:`plot.reduce_vectors` has different argument names.


Barotropic 2.0
--------------

Major release, 23 Apr 2020.

.. note::
    This version is not backwards-compatible with v1.x.

New features:

- :py:class:`Grid` exposes additional properties (`nlon`, `nlat`, `dlon`, `dlat`, `phis`, `lams`).
- :py:class:`Grid` is now able to compute gradients.
- The output of :py:meth:`Grid.zonalize_eqlat` should have improved in terms of contour value sampling and computation time for the default arguments.
- Stationary wavenumber diagnostic, waveguide detection and plot preset.
- New initial states :py:func:`init.motionless` and :py:func:`init.zonally_symmetric` have been added.
- Plot presets now allow selection of the latitude that is centered in maps.
- The pseudo-orography term has been completely reimplemented and now provides more configuration options and the possiblity to load a gridded orography.
- A new :py:mod:`io` submodule allows import of xarray datasets.
  However this functionality is not fully implemented yet.

Some functions and arguments have been renamed to achieve a more consistent naming throughout the code:

- Initial states are now accessible from a new :py:mod:`barotropic.init` submodule instead of being staticmethods of :py:class:`State`.
- :py:class:`BarotropicModel` now accepts ``None`` for the RHS forcing.
- The `diffusion_kappa` parameter of :py:class:`BarotropicModel` is now called `diffusion_coeff`.
- RHS PV tendencies are expected to be gridded instead of spectral.
- The resolution of a :py:class:`Grid` is now specified with the `resolution` argument (was named `latlon_resolution`).
- The properties `latitudes` and `longitudes` of :py:class:`Grid` are now called `lats` and `lons`.
- The parameters to specify the properties of the `gaussian_jet` initial state and orographic forcing have been changed.
- Plots are now accessible via a `plot` property of :py:class:`State` instead of methods whose names start with `plot_`.

Documentation:

- New example notebook recreating the experiments from `Wirth (2020) <https://doi.org/10.5194/wcd-1-111-2020>`_.


Barotropic 1.0
--------------

Initial release, 26 Nov 2019.


