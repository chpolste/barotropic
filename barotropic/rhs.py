"""RHS forcing terms in spectral space"""

import numpy as np



class RHS:
    """RHS term base class that implements arithmetic operators"""

    def __add__(self, other):
        return RHSSum(self, other)

    def __radd__(self, other):
        return RHSSum(other, self)

    def __mul__(self, other):
        return RHSProduct(self, other)

    def __rmul__(self, other):
        return RHSProduct(other, self)


class RHSSum(RHS):

    def __init__(self, term1, term2):
        self._term1 = term1
        self._term2 = term2

    def __call__(self, state):
        return self._term1(state) + self._term2(state)

    def __repr__(self):
        return "({} + {})".format(repr(self._term1), repr(self._term2))


class RHSProduct(RHS):

    def __init__(self, term1, term2):
        self._term1 = term1
        self._term2 = term2

    def __call__(self, state):
        return self._term1(state) * self._term2(state)

    def __repr__(self):
        return "({} * {})".format(repr(self._term1), repr(self._term2))


class TimedOffSwitch(RHS):
    """Turn off another forcing term after a specified amount of time
    
    The forcing term returns 1 until the specified switch-off time is reached
    after which it returns 0 forever. Multiply with the term that is supposed
    to be switched off.
    """

    def __init__(self, tend):
        """Set the switch-off time with `tend` in s"""
        self._tend = tend

    def __call__(self, state):
        return 1 if state.time <= self._tend else 0


class LinearRelaxation(RHS):
    """Linear relaxation towards a reference PV state"""

    def __init__(self, rate, reference_pv):
        """"""
        self._rate = rate
        self._reference_pv = reference_pv

    def __call__(self, state):
        return state.grid.to_spectral(self._rate * (self._reference_pv - state.pv))


class GaussianMountain(RHS):
    """Gaussian-shaped pseudo-orography"""

    def __init__(self, center=(30., 45.), half_value_width=(7.5, 20.), amplitude=0.15, fcor0=1.0e-4):
        """Gaussian mountain for pseudo-orographic forcing
        
        The mountain is centered at the (lon, lat)-tuple `center` (in degrees),
        decays with zonal and meridional standard deviations given by the tuple
        `half_value_width` (in degrees) and has an amplitude of `amplitude`
        (dimensionless). The forcing is calculated with a fixed coriolis
        parameter `fcor0` (in 1/s).
        """
        self._center_lon, self._center_lat = center
        self._hvw_lon_sq = half_value_width[0] ** 2
        self._hvw_lat_sq = half_value_width[1] ** 2
        self._const = fcor0 * amplitude

    def __call__(self, state):
        forcing = (self._const
                # Decaying in zonal direction
                * ( np.exp(-0.5 * (state.grid.lon - self._center_lon)**2 / self._hvw_lon_sq) )
                # Decaying in meridional direction
                * ( np.exp(-0.5 * (state.grid.lat - self._center_lat)**2 / self._hvw_lat_sq) ))
        # Use solid body rotation reference wind profile to calculate forcing
        u = 15. * np.cos(state.grid.phi)
        v = np.zeros_like(u)
        return -state.grid.divergence_spectral(u * forcing, v * forcing)


class ZonalSineMountains(RHS):
    """Sinusoidal pseudo-orography in the zonal direction"""

    def __init__(self, center_lat=45., half_value_width_lat=10., wavenumber=4, amplitude=0.3, fcor0=1.0e-4):
        """Sinusoidal mountain-chain for pseudo-orographic forcing

        The mountains are centered at latitude `center_lat` (in degrees) and
        decay in meridional direction with a standard deviation of
        `half_value_width_lat` (in degrees). Their `wavenumber` and `amplitude`
        are dimensionless parameters. The forcing is calculated with a fixed
        coriolis parameter `fcor0` (in 1/s).
        """
        self._center_lat = center_lat
        self._hvw_lat_sq = half_value_width_lat ** 2
        self._wavenumber = wavenumber
        self._const = fcor0 * amplitude

    def __call__(self, state):
        forcing = (self._const
                # Periodic in zonal direction
                * ( 0.5 * (1 + np.cos(self._wavenumber * np.deg2rad(state.grid.lon))) )
                # Decaying in meridional direction
                * ( np.exp(-0.5 * (state.grid.lat - self._center_lat)**2 / self._hvw_lat_sq) ))
        # Use solid body rotation reference wind profile to calculate forcing
        u = 15. * np.cos(state.grid.phi)
        v = np.zeros_like(u)
        return -state.grid.divergence_spectral(u * forcing, v * forcing)

