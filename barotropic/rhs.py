"""Predefined RHS forcing terms."""

import functools
import numbers
import numpy as np

from .constants import ZONAL as _ZONAL



class RHS:
    """RHS forcing term base class that implements arithmetic operators.
    
    Forcing terms should inherit from this class so they can be added and
    multiplied easily.
    """

    def __add__(self, other):
        return _RHSSum(self, other)

    def __radd__(self, other):
        return _RHSSum(other, self)

    def __mul__(self, other):
        return _RHSProduct(self, other)

    def __rmul__(self, other):
        return _RHSProduct(other, self)


class _RHSSum(RHS):

    def __init__(self, term1, term2):
        self._term1 = term1
        self._term2 = term2

    def __call__(self, state):
        return self._term1(state) + self._term2(state)

    def __repr__(self):
        return "({} + {})".format(repr(self._term1), repr(self._term2))


class _RHSProduct(RHS):

    def __init__(self, term1, term2):
        self._term1 = term1
        self._term2 = term2

    def __call__(self, state):
        return self._term1(state) * self._term2(state)

    def __repr__(self):
        return "({} * {})".format(repr(self._term1), repr(self._term2))



class TimedOffSwitch(RHS):
    """Turn off another forcing term after a specified amount of time.

    The forcing term returns 1 until the specified switch-off time is reached
    after which it returns 0 forever. Multiply with the term that is supposed
    to be switched off.
    """

    def __init__(self, tend):
        """The switch-off time is `tend`."""
        self._tend = tend

    def __call__(self, state):
        return 1 if state.time <= self._tend else 0


class LinearRelaxation(RHS):
    """Linear relaxation towards a reference PV state."""

    def __init__(self, rate, reference_pv, mask=None):
        """Specify the reference PV field and speed of relaxation.
        
        The forcing is calculated as
        
            rate * (reference_pv - pv),

        where `pv` is the PV field of the current timestep.

        The `mask` parameter does nothing at the moment.
        """
        self._rate = rate
        self._reference_pv = reference_pv
        # TODO: mask (allows implementation of sponge-like forcing)

    def __call__(self, state):
        return self._rate * (self._reference_pv - state.pv)


class Orography(RHS):
    """Pseudo-orographic forcing based on a given gridded orography.

    The forcing is calculated as

        -f/H * u·∇h,

    where `f` is the coriolis parameter in 1/s, `H` a scale height in m, `u`
    the horizontal wind and `h` is the height of the orography.

    This is class only supports time-invariant orography fields. The fields
    must be given on a lat-lon grid and will automatically be interpolated if
    required.

    Other orography forcing terms can inherit from this class and should only
    have to implement the `orography` method.
    """

    def __init__(self, lons, lats, orography, scale_height=10000., wind=("act", 1.), fcor_ref=None):
        """Set up an orographic forcing term for the specified orography.

        `orography` (in m), on a lat-lon grid given by `lats` and `lons`. The
        orography is linearly interpolated to the required grid when the
        forcing is evaluated by the model. `lons` and `orography` should be
        prepared such that the 0° column exists at both 0° and 360° to ensure
        that the interpolation does not yield NaN-values at the periodic
        boundary.
        
        `scale_height` should be given in m. The parameter `wind` controls
        which winds are used to evaluate the forcing, with these options:

        - `("act", factor)`: The actual 2D-wind is used, scaled by the given
          `factor`.
        - `("zon", factor)`: The zonal-mean zonal wind is used, scaled by the
          given `factor`. The meridional wind component is set to 0.
        - `("sbr", maxval)`: A constant solid body rotation wind profile is
          used with the given maximum wind speed at the equator. This can be
          useful to define a constant orography-based wavemaker for simple
          experiments.

        If `fcor_ref` is `None` (default), the actual coriolis parameter values
        are used to calculate the forcing, otherwise the given value (in 1/s)
        is used in the calculation of the forcing.
        """
        self._lons = lons
        self._lats = lats
        self._orography = orography
        self._scale_height = scale_height
        self._fcor_ref = fcor_ref
        # ...
        wind_kind, wind_fact = wind
        if wind_kind == "act":
            self._get_wind = lambda state: (wind_fact * state.u, wind_fact * state.v)
        elif wind_kind == "zon":
            self._get_wind = lambda state: (wind_fact * np.mean(state.u, axis=_ZONAL, keepdims=True), 0.)
        elif wind_kind == "sbr":
            self._get_wind = lambda state: (wind_fact * np.cos(state.grid.phis)[:,None], 0.)
        else:
            raise ValueError("wind specification error")

    def orography(self, grid):
        """The orography in m, interpolated to the given grid."""
        # Interpolate given orography to grid. Because np.interp requires
        # increasing x-values, flip sign of lats which range from +90° to -90°.
        lat_interp = lambda x: np.interp(-grid.lats, -self._lats, x)
        lon_interp = lambda x: np.interp( grid.lons,  self._lons, x)
        # 2D linear interpolation
        orog = np.apply_along_axis(lat_interp, 0, self._orography)
        return np.apply_along_axis(lon_interp, 1, orog)

    @functools.lru_cache(maxsize=8)
    def _orography_gradient(self, grid):
        """The vector-valued gradient of the orography.

        Values are cached since orography is time-invariant and only depends on
        the properties of the grid which do not change during a simulation.
        """
        return grid.gradient(self.orography(grid))

    def __call__(self, state):
        """Evaluate the forcing term for the given state."""
        grid = state.grid
        # Fixed coriolis parameter if given or use actual values from grid
        fcor = grid.fcor if self._fcor_ref is None else self._fcor_ref
        # Get wind and gradient of orography
        u, v = self._get_wind(state)
        gradx, grady = self._orography_gradient(grid)
        # Evaluate -f/H u·grad(orog)
        return -fcor / self._scale_height * (u * gradx + v * grady)


class GaussianMountain(Orography):
    """Gaussian-shaped pseudo-orography."""

    def __init__(self, height=1500, center=(30., 45.), stdev=(7.5, 20.), **orog_kwargs):
        """Gaussian mountain for pseudo-orographic forcing.

        The mountain is centered at the (lon, lat)-tuple `center` (in degrees)
        and decays with zonal and meridional standard deviations given by the
        tuple or value `stdev` (in degrees). `height` is the maximum height of
        the mountain in m. Additional `orog_kwargs` are given to the
        `Orography` base class.
        """
        super().__init__(None, None, None, **orog_kwargs)
        # Mountain properties for orography method
        self._height = height
        self._center_lon, self._center_lat = center
        if isinstance(stdev, numbers.Number):
            stdev = (stdev, stdev)
        self._stdev_lon_sq = stdev[0] ** 2
        self._stdev_lat_sq = stdev[1] ** 2

    def orography(self, grid):
        return (self._height
                # Decaying in zonal direction
                * ( np.exp(-0.5 * (grid.lon - self._center_lon)**2 / self._stdev_lon_sq) )
                # Decaying in meridional direction
                * ( np.exp(-0.5 * (grid.lat - self._center_lat)**2 / self._stdev_lat_sq) ))


class ZonalSineMountains(Orography):
    """Sinusoidal pseudo-orography in the zonal direction."""

    def __init__(self, height=1500, center_lat=45., stdev_lat=10., wavenumber=4, **orog_kwargs):
        """Sinusoidal mountain-chain for pseudo-orographic forcing.

        The mountains are centered at latitude `center_lat` (in degrees) and
        decay in meridional direction with a standard deviation of `stdev_lat`
        (in degrees). `wavenumber` specifies the number of crest-valley pairs
        in the zonal direction. `height` is the maximum height of the mountains
        in m. Additional `orog_kwargs` are given to the `Orography` base class.
        """
        super().__init__(None, None, None, **orog_kwargs)
        # Mountain chain properties for orography method
        self._height = height
        self._center_lat = center_lat
        self._stdev_lat_sq = stdev_lat ** 2
        self._wavenumber = wavenumber

    def orography(self, grid):
        return (self._height
                # Periodic in zonal direction
                * ( 0.5 * (1 + np.cos(self._wavenumber * grid.lam)) )
                # Decaying in meridional direction
                * ( np.exp(-0.5 * (grid.lat - self._center_lat)**2 / self._stdev_lat_sq) ))

