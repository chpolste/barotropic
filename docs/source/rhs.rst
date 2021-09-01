Forcing Terms
=============


Base Class Operations
---------------------

.. autoclass:: barotropic.rhs.RHS
    :undoc-members:
    :special-members:  __add__, __mul__


Relaxation
----------

.. autoclass:: barotropic.rhs.LinearRelaxation


Orographic Forcing
------------------

.. autoclass:: barotropic.rhs.GaussianMountain
.. autoclass:: barotropic.rhs.ZonalSineMountains

...

.. autoclass:: barotropic.rhs.Orography


Other
-----

.. automodule:: barotropic.rhs
    :exclude-members: RHS, LinearRelaxation, GaussianMountain, ZonalSineMountains, Orography

