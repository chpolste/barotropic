Barotropic
==========

Integrate the barotropic PV equation on the sphere. Fields are represented on a regular latitude-longitude grid while spatial derivatives are evaluated in spectral space (spherical harmonics). The model uses a leapfrog time integration scheme with Robert-Asselin filter, initialized with an Euler-forward step.

.. warning::
   :py:mod:`barotropic` version 3 is in development and will contain changes to the API.


Package Documentation
=====================

.. toctree::
   :maxdepth: 1

   module
   init
   rhs
   plot
   diagnostic
   io


Quickstart
==========

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


Examples
========

Jupyter notebooks utilizing :py:mod:`barotropic`:

- :doc:`examples/rwp-diagnostic-ghinassi-et-al-2018`
- :doc:`examples/waveguidability-wirth-2020`
- :doc:`examples/wavenumber-based-filtering`


.. toctree::
    :hidden:

    examples/rwp-diagnostic-ghinassi-et-al-2018
    examples/waveguidability-wirth-2020
    examples/wavenumber-based-filtering



Tips
====

- Functions with arguments that represent longitude and latitude (coordinates, etc.) will require them in this order (lon, lat), following the order used e.g. in plotting contexts or vector components (e.g. [X°N, Y°E] or [u, v]).
  Note that this is order is the reverse of the axis order in the underlying arrays, where latitude comes before longitude.
- Consider using the module-level **ZONAL** and **MERIDIONAL** constants as convenient and readable accessors for the grid dimensions when operating on arrays.

