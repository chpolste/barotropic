from . import diagnostics, plot, rhs, init, io
from .constants import *
from .grid import Grid
from .model import BarotropicModel
from .state import State

__version__ = "3.0.0"

# Explicitly define the public interface
__all__ = [
    "diagnostics", "init", "plot", "rhs", "io",
    "Grid", "BarotropicModel", "State",
    "ZONAL", "MERIDIONAL",
    "MIN", "HOUR", "DAY", "WEEK"
]

