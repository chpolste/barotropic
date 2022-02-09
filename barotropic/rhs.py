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

    Parameters:
        tend (number | datetime): Return ``1`` until this switch-off time is
            reached after which return ``0`` forever.

    Returns:
        Term for multiplication with other RHS terms.
    """

    def __init__(self, tend):
        self._tend = tend

    def __call__(self, state):
        return 1 if state.time <= self._tend else 0


class LinearRelaxation(RHS):
    """Linear relaxation towards a reference PV state.

    Parameters:
        rate (number): Relaxation rate in 1/s.
        reference_pv (array): PV field that is relaxed toward.
        mask (array): Does nothing at the moment.

    Implements **rate** * (**reference_pv** - pv).
    """

    def __init__(self, rate, reference_pv, mask=None):
        self._rate = rate
        self._reference_pv = reference_pv
        # TODO: mask (allows implementation of sponge-like forcing)

    def __call__(self, state):
        return self._rate * (self._reference_pv - state.pv)


class Orography(RHS):
    """Pseudo-orographic forcing based on a given gridded orography.

    Parameters:
        lon (array): Longitudes of orography grid.
        lat (array): Latitudes of orography grid.
        orography (array): Height of the orography in m on the lon-lat grid
            defined by **lon** and **lat**.
        scale_height (number): Scale height in m.
        wind ((str, number)): Wind used to evaluate the
            forcing. Options:

            - `("act", factor)`: The actual 2D-wind is used, scaled by the
              given factor.
            - `("zon", factor)`: The zonal-mean zonal wind is used, scaled by
              the given factor. The meridional wind component is set to 0.
            - `("sbr", maxval)`: A constant solid body rotation wind profile
              is used with the given maximum wind speed at the equator. This
              can be useful to define a constant orography-based wavemaker for
              simple experiments.

        fcor_ref(None | number): If `None`, the actual coriolis parameter
            values are used to calculate the forcing, otherwise the given value
            (in 1/s) is used as a constant everywhere.

    The forcing is calculated as ``-f/H * u·∇h`` where ``f`` is the coriolis
    parameter in 1/s, ``H`` the scale height, ``u`` the horizontal wind and
    ``h`` is the height of the orography.

    This is class only supports time-invariant fields of orography. The height
    field must be given on a lon-lat grid. The orography is linearly
    interpolated to the required grid when the forcing is evaluated by the
    model. **lons** and **orography** should be prepared such that the 0°
    column exists at both 0° and 360° to ensure that the interpolation does not
    yield ``NaN``-values at the periodic boundary.

    Other orographic forcing terms can inherit from this class and should only
    have to implement the `orography` method.
    """

    def __init__(self, lon, lat, orography, scale_height=10000., wind=("act", 1.), fcor_ref=None):
        self._lon = lon
        self._lat = lat
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
            self._get_wind = lambda state: (wind_fact * np.cos(state.grid.phi)[:,None], 0.)
        else:
            raise ValueError("wind specification error")

    def orography(self, grid):
        """Interpolated orography.

        Parameters:
            grid (:py:class:`.Grid`): Interpolation target grid.

        Returns:
            Orography in m.
        """
        # Interpolate given orography to grid. Because np.interp requires
        # increasing x-values, flip sign of lat which range from +90° to -90°.
        lat_interp = lambda x: np.interp(-grid.lat, -self._lat, x)
        lon_interp = lambda x: np.interp( grid.lon,  self._lon, x)
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
        fcor = grid.fcor2 if self._fcor_ref is None else self._fcor_ref
        # Get wind and gradient of orography
        u, v = self._get_wind(state)
        gradx, grady = self._orography_gradient(grid)
        # Evaluate -f/H u·grad(orog)
        return -fcor / self._scale_height * (u * gradx + v * grady)


class GaussianMountain(Orography):
    """Gaussian-shaped pseudo-orography.

    Parameters:
        height (number): Height of the mountain in m.
        center ((number, number)): Center (λ,φ) of the mountain in degrees.
        stdev (number | (number, number)): Standard deviation in degrees,
            either as single value for both directions, or a tuple of values
            for different mountain widths in lon and lat.
        orog_kwargs: Arguments given to the :py:class:`Orography` base class
            constructor.
    """

    def __init__(self, height=1500, center=(30., 45.), stdev=(7.5, 20.), **orog_kwargs):
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
                * ( np.exp(-0.5 * (grid.lon2 - self._center_lon)**2 / self._stdev_lon_sq) )
                # Decaying in meridional direction
                * ( np.exp(-0.5 * (grid.lat2 - self._center_lat)**2 / self._stdev_lat_sq) ))


class ZonalSineMountains(Orography):
    """Sinusoidal pseudo-orography in the zonal direction.

    Parameters:
        height (number): Peak height of the mountains in m.
        center_lat (number): Central latitude of the mountain chain.
        stdev_lat (number): Standard deviation governing the meridional extent
            of the mountains in m.
        wavenumber (int): Number of crest-valley-pairs in the zonal direction.
        orog_kwargs: Arguments given to the :py:class:`Orography` base class
            constructor.
    """

    def __init__(self, height=1500, center_lat=45., stdev_lat=10., wavenumber=4, **orog_kwargs):
        super().__init__(None, None, None, **orog_kwargs)
        # Mountain chain properties for orography method
        self._height = height
        self._center_lat = center_lat
        self._stdev_lat_sq = stdev_lat ** 2
        self._wavenumber = wavenumber

    def orography(self, grid):
        return (self._height
                # Periodic in zonal direction
                * ( 0.5 * (1 + np.cos(self._wavenumber * grid.lam2)) )
                # Decaying in meridional direction
                * ( np.exp(-0.5 * (grid.lat2 - self._center_lat)**2 / self._stdev_lat_sq) ))

