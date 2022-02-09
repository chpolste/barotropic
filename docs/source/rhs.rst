Forcing Terms
=============

Wavemakers, sponges and other terms that may be part of the forcing of the barotropic PV equation.


Base Class Operations
---------------------

All predefined forcing terms inherit from :py:class:`.rhs.RHS`, which provides addition and multiplication operators to combine individual terms into a more complicated right hand side:

.. autoclass:: barotropic.rhs.RHS
    :undoc-members:
    :special-members:  __add__, __mul__


Orographic Forcing
------------------

While an actual bottom orography is not part of the barotropic framework, (pseudo-)orographic forcing can be approximated.
All orographic terms inherit from a common base class:

.. autoclass:: barotropic.rhs.Orography

Pre-defined orographic features:

.. autoclass:: barotropic.rhs.GaussianMountain
.. autoclass:: barotropic.rhs.ZonalSineMountains


Relaxation
----------

.. autoclass:: barotropic.rhs.LinearRelaxation


Other
-----

.. automodule:: barotropic.rhs
    :exclude-members: RHS, Orography, GaussianMountain, ZonalSineMountains, LinearRelaxation

