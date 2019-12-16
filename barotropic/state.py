import numpy as np
from . import diagnostic
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
        # Memoized diagnostics
        self._fawa = None
        self._falwa = None
        self._falwa_filtered = None
        self._dominant_wavenumber = None

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

    # Shortcuts to diagnostic fields

    @property
    def fawa(self):
        """Finite-amplitude wave activity according to Nakamura and Zhu (2010)"""
        if self._fawa is None:
            self._fawa, _ = diagnostic.fawa(self, interpolate=self.grid.latitudes)
        return self._fawa

    @property
    def falwa(self):
        """Finite-amplitude local wave activity according to Huang and Nakamura (2016)"""
        if self._falwa is None:
            self._falwa, _ = diagnostic.falwa(self, interpolate=self.grid.latitudes)
        return self._falwa

    @property
    def falwa_filtered(self):
        """Filtered Finite-amplitude local wave activity as in Ghinassi et al. (2018)"""
        if self._falwa_filtered is None:
            self._falwa_filtered = diagnostic.filter_by_wavenumber(self.falwa, self.dominant_wavenumber)
        return self._falwa_filtered

    @property
    def falwa_hn2016(self):
        """Finite-amplitude local wave activity according to Huang and Nakamura (2016)"""
        return diagnostic.falwa_hn2016(self, normalize_icos=True)

    @property
    def dominant_wavenumber(self):
        """Dominant zonal wavenumber at every gridpoint based on v"""
        if self._dominant_wavenumber is None:
            self._dominant_wavenumber = diagnostic.dominant_wavenumber(self.v, self.grid, smoothing=(21, 7))
        return self._dominant_wavenumber

    @property
    def rwp_envelope(self):
        """RWP envelope according to Zimin et al. (2003)"""
        return diagnostic.envelope_hilbert(self.v, 2, 10)

    # Shortcut to model integration

    def run(self, model, *args, **kwargs):
        """See barotropic.BarotropicModel.run"""
        return model.run(self, *args, **kwargs)

    # Shortcuts to plotting. Do not return figures here, so user does not have
    # to put semicolons in Jupyter notebooks to suppress double figure output.

    @property
    def plot(self):
        return StatePlotter(self)


class StatePlotter:

    def __init__(self, state):
        self._state = state

    def summary(self, *args, **kwargs):
        """See barotropic.plot.summary"""
        from . import plot
        plot.summary(self._state, *args, **kwargs)

    def wave_activity(self, *args, **kwargs):
        """See barotropic.plot.wave_activity"""
        from . import plot
        plot.wave_activity(self._state, *args, **kwargs)

    def rwp_diagnostic(self, *args, **kwargs):
        """See barotropic.plot.rwp_diagnostic"""
        from . import plot
        plot.rwp_diagnostic(self._state, *args, **kwargs)

