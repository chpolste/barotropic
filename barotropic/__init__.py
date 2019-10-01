"""Barotropic model on the sphere (pseudospectral discretization)


References
----------

Huang, C. S., & Nakamura, N. (2016). Local finite-amplitude wave activity as
    a diagnostic of anomalous weather events. Journal of the Atmospheric
    Sciences, 73(1), 211-229.

Nakamura, N., & Zhu, D. (2010). Finite-amplitude wave activity and diffusive
    flux of potential vorticity in eddy–mean flow interaction. Journal of the
    Atmospheric Sciences, 67(9), 2701-2716.
    https://doi.org/10.1175/2010JAS3432.1

"""

from . import rhs, plot
from .constants import *
from .grid import Grid
from .model import BarotropicModel
from .state import State

