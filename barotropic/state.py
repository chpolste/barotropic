import numpy as np
from numbers import Number
from .constants import ZONAL


class State:
    """Model field container for a barotropic atmosphere"""

    def __init__(self, grid, time, pv=None, pv_spectral=None):
        self.grid = grid
        self.time = time
        # Lazily evaluated, memoized fields
        assert not (pv is None and pv_spectral is None)
        self._pv = pv
        self._pv_spectral = pv_spectral
        self._u = None
        self._v = None
        self._vorticity = None
        self._streamfunction = None
        self._fawa = None
        self._falwa = None

    @classmethod
    def from_wind(cls, grid, time, u, v):
        """Initialize a model state based on wind components"""
        pv = grid.fcor + grid.vorticity(u, v)
        return cls(grid, time, pv=pv)

    # Preconfigured wind profiles for initial conditions

    @classmethod
    def solid_body_rotation(cls, grid, time=0., amplitude=15.):
        """Predefined wind field using a cos(latitude) profile"""
        u = amplitude * np.cos(grid.phi)
        v = np.zeros_like(u)
        return cls.from_wind(grid, time, u, v)

    @classmethod
    def gaussian_jet(cls, grid, time=0., latitude=45., amplitude=20., stdev=5.):
        """Predefined wind field using a bell-shaped jet

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
        return cls.from_wind(grid, time, u, v)

    @classmethod
    def held_1985(cls, grid, time=0., A=25., B=30., C=300.):
        """Zonal wind profile similar to that of the upper troposphere

        Introduced by Held (1985), also used by Held and Phillips (1987) and
        Ghinassi et al. (2018).
        """
        cosphi = np.cos(grid.phi)
        sinphi = np.sin(grid.phi)
        u = A * cosphi - B * cosphi**3 + C * cosphi**6 * sinphi**2
        v = np.zeros_like(u)
        return cls.from_wind(grid, time, u, v)

    # Arithmetic operators

    def add_wind(self, other):
        """Wind-based addition of model fields"""
        if isinstance(other, State):
            assert self.grid is other.grid
            u = self.u + other.u
            v = self.v + other.v
        else:
            u, v = other
            assert self.grid.shape == u.shape
            assert self.grid.shape == v.shape
        return State.from_wind(self.grid, self.time, u, v)

    def add_vorticity(self, other):
        """Relative vorticity-based addition of model fields"""
        if isinstance(other, State):
            assert self.grid is other.grid
            # Add vorticity to PV so that planetary vorticity is not doubled
            pv = self.pv + other.vorticity
        else:
            assert self.grid.shape == other.shape
            pv = self.pv + other
        return State(self.grid, self.time, pv=pv)

    # Field accessors

    @property
    def pv(self):
        """Barotropic potential vorticity = absolute vorticity"""
        if self._pv is None:
            self._pv = self.grid.to_grid(self._pv_spectral)
        return self._pv

    @property
    def pv_spectral(self):
        """Barotropic potential vorticity in spectral space"""
        if self._pv_spectral is None:
            self._pv_spectral = self.grid.to_spectral(self._pv)
        return self._pv_spectral

    @property
    def u(self):
        """Zonal wind component"""
        if self._u is None:
            self._invert_pv()
        return self._u

    @property
    def v(self):
        """Meridional wind component"""
        if self._v is None:
            self._invert_pv()
        return self._v

    def _invert_pv(self):
        """Perform PV inversion to obtain wind components"""
        # Compute wind from vorticity using div = 0
        self._u, self._v = self.grid.wind(self.vorticity, np.zeros_like(self.vorticity))

    @property
    def vorticity(self):
        """Relative vorticity"""
        if self._vorticity is None:
            self._vorticity = self.pv - self.grid.fcor
        return self._vorticity

    @property
    def streamfunction(self):
        """Streamfunction of the wind"""
        if self._streamfunction is None:
            # Perform PV inversion to obtain wind
            vorticity_spectral = self.grid.to_spectral(self.vorticity)
            # Eigenvalues of the horizontal Laplacian
            eigenvalues = self.grid.laplacian_eigenvalues
            # Compute streamfunction
            psi_spectral = np.zeros_like(vorticity_spectral)
            psi_spectral[1:] = - vorticity_spectral[1:] / eigenvalues[1:]
            self._streamfunction = self.grid.to_grid(psi_spectral)
        return self._streamfunction

    @property
    def pv_flux_spectral(self):
        """PV flux in spectral space = [q*u, q*v]"""
        return self.grid.divergence_spectral(self.pv * self.u, self.pv * self.v)

    @property
    def enstrophy(self):
        return 0.5 * self.pv**2

    @property
    def eddy_enstrophy(self):
        return 0.5 * (self.pv - np.mean(self.pv, axis=ZONAL, keepdims=True))**2

    @property
    def energy(self):
        return 0.5 * (self.u * self.u + self.v * self.v)

    @property
    def fawa(self):
        """Finite-amplitude wave activity according to Nakamura and Zhu (2010)"""
        if self._fawa is None:
            grid = self.grid
            levels = 2 * grid.latitudes.size
            qq, yy = grid.zonalize_eqlat(self.pv, levels=levels, interpolate=None, quad="sptrapz")
            q_int = np.vectorize(lambda q: grid.quad_sptrapz(self.pv, self.pv - q))
            y_int = np.vectorize(lambda y: grid.quad_sptrapz(self.pv, grid.lat - y))
            wa = (q_int(qq) - y_int(yy)) / grid.circumference(yy)
            self._fawa = np.interp(grid.latitudes, yy, wa, left=0, right=0)
        return self._fawa

    @property
    def falwa(self):
        """Local Finite-amplitude wave activity according to Huang and Nakamura (2016)"""
        if self._falwa is None:
            grid = self.grid
            levels = 2 * grid.latitudes.size
            qq, yy = grid.zonalize_eqlat(self.pv, levels=levels, interpolate=grid.latitudes, quad="sptrapz")
            q_int = np.frompyfunc(lambda q, y: grid.quad_sptrapz_meridional(self.pv - q, self.pv - q), 2, 1)
            y_int = np.frompyfunc(lambda q, y: grid.quad_sptrapz_meridional(self.pv - q, grid.lat - y), 2, 1)
            icoslat = 1. / np.cos(np.deg2rad(yy))
            icoslat[ 0] = 0.
            icoslat[-1] = 0.
            self._falwa = np.stack(icoslat * (q_int(qq, yy) - y_int(qq, yy)))
        return self._falwa

    @property
    def falwa_hn2016(self):
        from hn2016_falwa.oopinterface import BarotropicField
        # hn2016_falwa expects latitudes to start at south pole
        xlon = self.grid.longitudes
        ylat = np.flip(self.grid.latitudes)
        bf = BarotropicField(xlon, ylat, pv_field=np.flipud(self.pv))
        # hn2016_falwa does not normalize with cosine of latitude
        icoslat = 1. / np.cos(self.grid.phi)
        icoslat[ 0,:] = 0.
        icoslat[-1,:] = 0.
        return np.flipud(icoslat * bf.lwa)

    # Shortcut to model integration

    def run(self, model, *args, **kwargs):
        return model.run(self, *args, **kwargs)

    # Shortcuts to plotting

    def plot_summary(self, *args, **kwargs):
        from . import plot
        return plot.summary(self, *args, **kwargs)

    def plot_wave_activity(self, *args, **kwargs):
        from . import plot
        return plot.wave_activity(self, *args, **kwargs)

