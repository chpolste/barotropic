Tips and Examples
=================

- Functions with arguments that represent longitude and latitude (coordinates, etc.) will always take them in this order (first lon, then lat), following the order used e.g. in plotting contexts or vector components (e.g. [X°N, Y°E] or [u, v]).
  Note that this is order is the reverse of the axis order in the underlying arrays, where latitude comes before longitude.
- To ensure numerically robust and accurate results, :py:meth:`.Grid.quad` and :py:meth:`.Grid.quad_meridional` as well as functions relying on these routines, such as :py:meth:`.Grid.zonalize` and :py:func:`.diagnostics.falwa`, use the *sptrapz* integration scheme by default.
  Consider using the *boxcount* scheme to accelerate computations where highest accuracy is not required.
- Use the convenience constants :py:data:`ZONAL` and :py:data:`MERIDIONAL` from the top-level namespace to address field dimensions.
- Use the convenience factors :py:data:`MIN`, :py:data:`HOUR`, :py:data:`DAY`, :py:data:`WEEK` to create time intervals in a readable manner. Alternatively, both :py:class:`.State` and :py:class:`.BarotropicModel` can handle :py:class:`datetime.timedelta` and :py:class:`datetime.datetime` objects as input.


Example Notebooks
-----------------

Jupyter notebooks utilizing :py:mod:`barotropic`:

- :doc:`examples/rwp-diagnostics-ghinassi-et-al-2018`
- :doc:`examples/waveguidability-wirth-2020`
- :doc:`examples/wavenumber-based-filtering`


.. toctree::
    :hidden:

    examples/rwp-diagnostics-ghinassi-et-al-2018
    examples/waveguidability-wirth-2020
    examples/wavenumber-based-filtering


Transition from 2.x to 3.x
--------------------------

In the transition from version 2.x to 3.x, many methods, attributes, function arguments, etc. have been moved or renamed.
Some significant changes include:

- The **lat**, **lon**, etc. properties of :py:class:`.Grid` have been renamed.
- The former `diagnostic` module is now called `diagnostics`.
- Quadrature, derivatives and interpolation on :py:class:`.Grid` have been restructured.
- *scipy* is now a mandatory dependency.
- Longitude now appears consistently before latitude in argument lists (except for direct array access, where lat is the zeroth dimension).

Thus, it is very unlikely that code from version 2.x still works after upgrading to version 3.x.

Version 2.x continues to be available on the master branch, development of 3.x has moved to a new main branch (which is the default branch of the repository now).


