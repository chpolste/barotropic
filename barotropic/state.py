import numpy as np
from . import diagnostic, formatting
from .constants import ZONAL
from .grid import Grid


def _cached_property(f):
    cache = "_" + f.__name__
    def wrapper(self):
        if not hasattr(self, cache) or getattr(self, cache) is None:
            setattr(self, cache, f(self))
        return getattr(self, cache)
    wrapper.__doc__ = f.__doc__
    return property(wrapper)


class State:
    """Model field container for a barotropic atmosphere.
    
    Provides convenient access to fields, diagnostics and plots. Should
    generally be treated as an immutable object.
    """

    def __init__(self, grid, time, pv=None, pv_spectral=None):
        """`barotropic.State` constructor.
        
        - `grid` is a `barotropic.Grid` instance.
        - `time` is the valid time of the atmospheric state. To work properly
          with the time integration routines, this should be either a number
          representing seconds or a `datetime.datetime` instance.
        - Either `pv` xor `pv_spectral` must be given. If `pv` (gridded) is
          specified, it is converted to spectral representation first to ensure
          that the field can be represented by the spherical harmonics. An
          error is raised when both are given.

        Also see the alternative static method constructors and
        `barotropic.init`.
        """
        self.grid = grid
        self.time = time
        # Always send PV field through spectral representation to ensure
        # consistency between spherical harmonics and gridded representation
        assert pv is None or pv_spectral is None, "can only specify one of pv and pv_spectral args"
        self._pv_spectral = pv_spectral if pv is None else grid.to_spectral(pv)
        # Lazily evaluated, memoized fields
        self._pv = None
        self._u = None
        self._v = None
        self._streamfunction = None

    def __repr__(self):
        return formatting.state_repr(self)

    @classmethod
    def from_wind(cls, grid, time, u, v):
        """Initialize a model state based on wind components.
        
        This is an alternative constructor. `grid` and `time` are given to the
        default constructor and the state is initialized with PV derived from
        the given fields of wind components.
        """
        pv = grid.fcor + grid.vorticity(u, v)
        return cls(grid, time, pv=pv)

    @classmethod
    def from_vorticity(cls, grid, time, vorticity):
        """Initialize a model state based on relative vorticity.

        This is an alternative constructor. `grid` and `time` are given to the
        default constructor.
        """
        pv = grid.fcor + vorticity
        return cls(grid, time, pv=pv)

    # Arithmetic operators

    def add_wind(self, other):
        """Wind-based addition of atmospheric states.
        
        Combine this `barotropic.State` with `other` by adding their wind
        fields component-wise. `other` can either be another `barotropic.State`
        or a tuple containing the `u` and `v` component fields as arrays.
        Returns a new `barotropic.State`. The `time` parameter is taken from
        the object whose `add_wind` method is called.
        """
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
        """Relative vorticity-based addition of atmospheric states.
        
        Combine this `barotropic.State` with `other` by adding their relative
        vorticity fields. `other` can either be another `barotropic.State` or
        a vorticity field as an array. Returns a new `barotropic.State`. The
        `time` parameter is taken from the object whose `add_vorticity` method
        is called.
        """
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
        """Barotropic potential vorticity (i.e. absolute vorticity)."""
        if self._pv is None:
            self._pv = self.grid.to_grid(self._pv_spectral)
        return self._pv

    @property
    def pv_spectral(self):
        """Barotropic potential vorticity in spectral space."""
        if self._pv_spectral is None:
            self._pv_spectral = self.grid.to_spectral(self._pv)
        return self._pv_spectral

    @property
    def u(self):
        """Zonal wind component."""
        if self._u is None:
            self._invert_pv()
        return self._u

    @property
    def v(self):
        """Meridional wind component."""
        if self._v is None:
            self._invert_pv()
        return self._v

    def _invert_pv(self):
        """Perform PV inversion to obtain wind components."""
        # Compute wind from vorticity using div = 0
        vorticity = self.vorticity_spectral
        self._u, self._v = self.grid.wind(vorticity, np.zeros_like(vorticity))

    @property
    def streamfunction(self):
        """Streamfunction of the wind."""
        if self._streamfunction is None:
            psi_spec = self.grid.solve_poisson_spectral(self.vorticity_spectral)
            self._streamfunction = self.grid.to_grid(psi_spec)
        return self._streamfunction

    @property
    def pv_flux_spectral(self):
        """PV flux in spectral space `[q*u, q*v]`."""
        return self.grid.divergence_spectral(self.pv * self.u, self.pv * self.v)

    @property
    def vorticity(self):
        """Relative vorticity."""
        return self.pv - self.grid.fcor

    @property
    def vorticity_spectral(self):
        return self.pv_spectral - self.grid.fcor_spectral

    @property
    def enstrophy(self):
        """Enstropy = `0.5 * q²` where `q` is PV."""
        return 0.5 * self.pv**2

    @property
    def eddy_enstrophy(self):
        """Enstropy after removal of the zonal-mean PV."""
        return 0.5 * (self.pv - np.mean(self.pv, axis=ZONAL, keepdims=True))**2

    @property
    def energy(self):
        """Kinetic energy of the flow."""
        return 0.5 * (self.u * self.u + self.v * self.v)

    # Shortcuts to diagnostic fields

    @_cached_property
    def pv_zonalized(self):
        """Zonalized PV profile on the regular grid.
        
        See `barotropic.Grid.zonalize`.
        """
        return self.grid.zonalize(self.pv, interpolate=self.grid.lats)[0]

    @property
    def fawa(self):
        """Finite-Amplitude Wave Activity on the regular grid.
        
        See `barotropic.diagnostic.fawa`.
        """
        # Instead of an extra computation with diagnostic.fawa, use that
        # FAWA is the zonal average of FALWA (Huang and Nakamura 2016)
        return self.falwa.mean(axis=ZONAL)

    @_cached_property
    def falwa(self):
        """Finite-Amplitude Local Wave Activity on the regular grid.
        
        See `barotropic.diagnostic.falwa`.
        """
        return diagnostic.falwa(self, interpolate=self.grid.lats)[0]

    @property
    def falwa_filtered(self):
        """Finite-Amplitude Local Wave Activity, phase-filtered based on v.

        The FALWA field is filtered based on the doubled dominant wavenumber of
        the meridional wind obtained from Fourier analysis at each latitude as
        in Ghinassi et al. (2020).
        
        See `barotropic.diagnostic.falwa`,
        `barotropic.diagnostic.dominant_wavenumber_fourier` and
        `barotropic.diagnostic.filter_by_wavenumber`.
        """
        dominant_wavenumber = diagnostic.dominant_wavenumber_fourier(self.v, self.grid)
        return diagnostic.filter_by_wavenumber(self.falwa, 2*dominant_wavenumber)

    @property
    def falwa_hn2016(self):
        """Finite-Amplitude Local Wave Activity on the regular grid.
        
        See `barotropic.diagnostic.falwa_hn2016`.
        """
        return diagnostic.falwa_hn2016(self, normalize_icos=True)

    @property
    def v_envelope_hilbert(self):
        """Envelope of wave packets based on the Hilbert transform.

        See `barotropic.diagnostic.envelope_hilbert`.
        """
        return diagnostic.envelope_hilbert(self.v, (2, 10))

    @property
    def stationary_wavenumber(self):
        """Non-dimensionalised stationary (zonal) wavenumber (`Ks`, complex).

        See `barotropic.diagnostic.stationary_wavenumber`.
        """
        return diagnostic.stationary_wavenumber(self)

    def extract_waveguides(self, *args, **kwargs):
        """Shortcut to `barotropic.diagnostic.extract_waveguides`."""
        return diagnostic.extract_waveguides(self, *args, **kwargs)

    # Shortcut to model integration

    def run(self, model, *args, **kwargs):
        """Shortcut to `barotropic.BarotropicModel.run`"""
        return model.run(self, *args, **kwargs)

    # Shortcuts to plotting. Do not return figures here, so user does not have
    # to put semicolons in Jupyter notebooks to suppress double figure output.

    @property
    def plot(self):
        """Interface to plot presets from `barotropic.plot` for interactive use.

        Provides shortcuts to:

        - `barotropic.plot.rwp_diagnostic`
        - `barotropic.plot.summary`
        - `barotropic.plot.wave_activity`
        - `barotropic.plot.waveguides`

        Call like `state.plot.preset(...)`, where `preset` is the plot preset
        you want to see. The plotters accessed through this interface do not
        return the created `Figure` instance, which avoids that the created
        image is displayed twice in the output of a jupyter notebook cell.
        """
        return StatePlotter(self)


class StatePlotter:

    def __init__(self, state):
        self._state = state

    def summary(self, *args, **kwargs):
        from . import plot
        plot.summary(self._state, *args, **kwargs)

    def wave_activity(self, *args, **kwargs):
        from . import plot
        plot.wave_activity(self._state, *args, **kwargs)

    def rwp_diagnostic(self, *args, **kwargs):
        from . import plot
        plot.rwp_diagnostic(self._state, *args, **kwargs)

    def waveguides(self, *args, **kwargs):
        from . import plot
        plot.waveguides(self._state, *args, **kwargs)
