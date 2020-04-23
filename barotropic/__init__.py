"""A barotropic model on the sphere


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

Hoskins, B. J., & Karoly, D. J. (1981). The steady linear response of
    a spherical atmosphere to thermal and orographic forcing. Journal of the
    Atmospheric Sciences, 38(6), 1179-1196.
    https://doi.org/10.1175/1520-0469(1981)038<1179:TSLROA>2.0.CO;2

Huang, C. S., & Nakamura, N. (2016). Local finite-amplitude wave activity as
    a diagnostic of anomalous weather events. Journal of the Atmospheric
    Sciences, 73(1), 211-229.

Nakamura, N., & Zhu, D. (2010). Finite-amplitude wave activity and diffusive
    flux of potential vorticity in eddy–mean flow interaction. Journal of the
    Atmospheric Sciences, 67(9), 2701-2716.
    https://doi.org/10.1175/2010JAS3432.1

Petoukhov, V., Rahmstorf, S., Petri, S., & Schellnhuber, H. J. (2013).
    Quasiresonant amplification of planetary waves and recent Northern
    Hemisphere weather extremes. Proceedings of the National Academy of
    Sciences, 110(14), 5336-5341.
    https://doi.org/10.1073/pnas.1222000110

Wirth, V. (2020). Waveguidability of idealized midlatitude jets and the
    limitations of ray tracing theory. Weather Clim. Dynam., 1, 111–125.
    https://doi.org/10.5194/wcd-1-111-2020

Zimin, A. V., Szunyogh, I., Patil, D. J., Hunt, B. R., & Ott, E. (2003).
    Extracting envelopes of Rossby wave packets. Monthly weather review,
    131(5), 1011-1017.
    https://doi.org/10.1175/1520-0493(2003)131<1011:EEORWP>2.0.CO;2

"""

from . import diagnostic, plot, rhs, init, io
from .constants import *
from .grid import Grid
from .model import BarotropicModel
from .state import State

__version__ = "2.0.0"

