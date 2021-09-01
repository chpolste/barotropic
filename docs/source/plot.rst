Plotting
========

.. note::
    Plotting requires :py:mod:`matplotlib`. If :py:mod:`matplotlib` is not available, a warning will be emitted on import but some of the helper functions will still work.


Plot presets
------------

All plot presets are also accessible for interactive use as methods of the :py:attr:`.State.plot` interface.

.. autofunction:: barotropic.plot.rwp_diagnostic
.. autofunction:: barotropic.plot.summary
.. autofunction:: barotropic.plot.wave_activity
.. autofunction:: barotropic.plot.waveguides


Helper functions
----------------

.. automodule:: barotropic.plot
    :exclude-members: rwp_diagnostic, summary, wave_activity, waveguides

