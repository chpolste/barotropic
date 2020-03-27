""" Preconfigured initial conditions"""

import numpy as np
from .state import State


def motionless(grid, time=0.):
    """A motionless state"""
    return State(grid, time, pv=grid.coriolis(grid.lat))


def zonally_symmetric(grid, time=0., u=None, pv=None):
    # Zonal profile of zonal wind given
    if u is not None and pv is None:
        u = np.asarray(u, dtype=float)
        assert u.ndim == 1 and u.size == grid.nlat, "Invalid dimensions for PV profile"
        u = np.repeat(u, grid.nlon).reshape(grid.shape)
        v = np.zeros_like(u)
        return State.from_wind(grid, time, u, v)
    # Zonal distribution of PV given
    if u is None and pv is not None:
        pv = np.asarray(pv, dtype=float)
        assert pv.ndim == 1 and pv.size == grid.nlat, "Invalid dimensions for PV profile"
        pv = np.repeat(pv, grid.nlon).reshape(grid.shape)
        return State(grid, time, pv=pv)
    raise ValueError("A zonal profile of u xor pv must be specified")


def solid_body_rotation(grid, time=0., amplitude=15.):
    """A cos(latitude) zonal wind profile"""
    u = amplitude * np.cos(grid.phi)
    v = np.zeros_like(u)
    return State.from_wind(grid, time, u, v)


def gaussian_jet(grid, time=0., amplitude=20., center_lat=45., stdev_lat=5.):
    """A bell-shaped zonal jet

    A linear wind profile in latitude is added to zero wind speeds at both
    poles.
    """
    u = amplitude * np.exp( -0.5 * (grid.lat - latitude)**2 / stdev**2 )
    # Subtract a linear function to set u=0 at the poles
    u_south = u[-1,0]
    u_north = u[ 0,0]
    u = u - 0.5 * (u_south + u_north) + (u_south - u_north) * grid.lat / 180.
    # No meridional wind
    v = np.zeros_like(u)
    return State.from_wind(grid, time, u, v)


def held_1985(grid, time=0., A=25., B=30., C=300.):
    """Zonal wind profile similar to that of the upper troposphere

    Introduced by Held (1985), also used by Held and Phillips (1987) and
    Ghinassi et al. (2018).
    """
    cosphi = np.cos(grid.phi)
    sinphi = np.sin(grid.phi)
    u = A * cosphi - B * cosphi**3 + C * cosphi**6 * sinphi**2
    v = np.zeros_like(u)
    return State.from_wind(grid, time, u, v)

