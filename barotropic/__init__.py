"""A barotropic model on the sphere.

- Obtain the latest source code from [GitHub](https://github.com/chpolste/barotropic).
- Licensed under the Apache License, Version 2.0.


Quick Start
-----------

Running the model generally involves these steps:

1. Create a `barotropic.Grid` instance.
2. Set up initial conditions (`barotropic.init`).
3. Set up forcing terms (`barotropic.rhs`).
4. Configure and run the model (`barotropic.BarotropicModel`).
5. Inspect and visualize the output (`barotropic.State`, `barotropic.diagnostic`, `barotropic.plot`).

A few example notebooks are provided in the code repository.


Convenience Constants
---------------------

- `MERIDIONAL`, `ZONAL`: Dimension indices for gridded fields.
- `MIN`, `HOUR`, `DAY`, `WEEK`: Time unit lengths in seconds.

These values can be used to improve readability of code and clarify meaning of
some operations, e.g. by using `axis=ZONAL` instead of `axis=-1` in numpy
operations or `5*DAY` instead of `432000`.


References
----------

These articles are referenced in the source code and documentation:

- [Ghinassi et al. (2018)](https://doi.org/10.1175/MWR-D-18-0068.1).
  Local finite-amplitude wave activity as a diagnostic for Rossby wave packets.
  Monthly Weather Review, 146(12), 4099-4114.
- [Held (1985)](https://doi.org/10.1175/1520-0469(1985)042<2280:PATOOM>2.0.CO;2).
  Pseudomomentum and the orthogonality of modes in shear flows
  Journal of the Atmospheric Sciences, 42(21), 2280-2288.
- [Held & Phillips (1987)](https://doi.org/10.1175/1520-0469(1987)044<0200:LANBDO>2.0.CO;2).
  Linear and nonlinear barotropic decay on the sphere
  Journal of the Atmospheric Sciences, 44(1), 200-207.
- [Hoskins & Karoly (1981)](https://doi.org/10.1175/1520-0469(1981)038<1179:TSLROA>2.0.CO;2).
  The steady linear response of a spherical atmosphere to thermal and orographic forcing.
  Journal of the Atmospheric Sciences, 38(6), 1179-1196.
- [Huang & Nakamura (2016)](https://doi.org/10.1175/JAS-D-15-0194.1).
  Local finite-amplitude wave activity as a diagnostic of anomalous weather events.
  Journal of the Atmospheric Sciences, 73(1), 211-229.
- [Nakamura & Zhu (2010)](https://doi.org/10.1175/2010JAS3432.1).
  Finite-amplitude wave activity and diffusive flux of potential vorticity in eddy–mean flow interaction.
  Journal of the Atmospheric Sciences, 67(9), 2701-2716.
- [Petoukhov et al. (2013)](https://doi.org/10.1073/pnas.1222000110).
  Quasiresonant amplification of planetary waves and recent Northern Hemisphere weather extremes.
  Proceedings of the National Academy of Sciences, 110(14), 5336-5341.
- [Wirth (2020)](https://doi.org/10.5194/wcd-1-111-2020).
  Waveguidability of idealized midlatitude jets and the  limitations of ray tracing theory.
  Weather Clim. Dynam., 1, 111–125.
- [Zimin et al. (2003)](https://doi.org/10.1175/1520-0493(2003)131<1011:EEORWP>2.0.CO;2).
  Extracting envelopes of Rossby wave packets.
  Monthly weather review, 131(5), 1011-1017.

"""

from . import diagnostic, plot, rhs, init, io
from .constants import *
from .grid import Grid
from .model import BarotropicModel
from .state import State

__version__ = "2.0.1"

# Explicitly define the public interface
__all__ = [
    "diagnostic", "init", "plot", "rhs", "io",
    "Grid", "BarotropicModel", "State",
    "ZONAL", "MERIDIONAL",
    "MIN", "HOUR", "DAY", "WEEK"
]

