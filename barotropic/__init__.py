"""Barotropic model on the sphere (pseudospectral discretization)


References
----------

Ghinassi, P., Fragkoulidis, G., & Wirth, V. (2018). Local finite-amplitude wave
    activity as a diagnostic for Rossby wave packets. Monthly Weather Review,
    146(12), 4099-4114. https://doi.org/10.1175/MWR-D-18-0068.1

Held, I. M. (1985). Pseudomomentum and the orthogonality of modes in shear
    flows. Journal of the Atmospheric Sciences, 42(21), 2280-2288.
    https://doi.org/10.1175/1520-0469(1985)042<2280:PATOOM>2.0.CO;2

Held, I. M., & Phillips, P. J. (1987). Linear and nonlinear barotropic decay on
    the sphere. Journal of the Atmospheric Sciences, 44(1), 200-207.
    https://doi.org/10.1175/1520-0469(1987)044<0200:LANBDO>2.0.CO;2

Huang, C. S., & Nakamura, N. (2016). Local finite-amplitude wave activity as
    a diagnostic of anomalous weather events. Journal of the Atmospheric
    Sciences, 73(1), 211-229.

Nakamura, N., & Zhu, D. (2010). Finite-amplitude wave activity and diffusive
    flux of potential vorticity in eddy–mean flow interaction. Journal of the
    Atmospheric Sciences, 67(9), 2701-2716.
    https://doi.org/10.1175/2010JAS3432.1

Zimin, A. V., Szunyogh, I., Patil, D. J., Hunt, B. R., & Ott, E. (2003).
    Extracting envelopes of Rossby wave packets. Monthly weather review,
    131(5), 1011-1017.
    https://doi.org/10.1175/1520-0493(2003)131<1011:EEORWP>2.0.CO;2

"""

from . import diagnostic, plot, rhs
from .constants import *
from .grid import Grid
from .model import BarotropicModel
from .state import State

