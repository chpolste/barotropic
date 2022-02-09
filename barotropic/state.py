import numpy as np
from . import diagnostics, formatting
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

    Parameters:
        grid (:py:class:`Grid`): Grid description for contained fields.
        time (number | datetime): Valid time. Specify either a number in
            seconds or a datetime-like object.
        pv (None | array): Gridded field of potential vorticity (PV, absolute
            vorticity in the barotropic framework).
        pv_spectral (None | array): Spectral representation of potential
            vorticity. Only specify if **pv** is ``None``.

    Note:
        There are alternative classmethod constructors of :py:class:`State` and
        the :py:mod:`init` module also provides functions to create
        :py:class:`State` objects for model initialization.

    Provides convenient access to fields, diagnostics and plots. Should
    generally be treated as an immutable object.
    """

    def __init__(self, grid, time, pv=None, pv_spectral=None):
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

        Parameters:
            grid (:py:class:`Grid`): Given to the default constructor.
            time (number | datetime): Given to the default constructor.
            u (array): Zonal wind component for initialization.
            v (array): Meridional wind component for initialization.

        Returns:
            New :py:class:`State` instance.
        """
        pv = grid.fcor2 + grid.vorticity(u, v)
        return cls(grid, time, pv=pv)

    @classmethod
    def from_vorticity(cls, grid, time, vorticity):
        """Initialize a model state based on relative vorticity.

        Parameters:
            grid (:py:class:`Grid`): Given to the default constructor.
            time (number | datetime): Given to the default constructor.
            vorticity (array): Relative vorticity field for initialization.

        Returns:
            New :py:class:`State` instance.
        """
        pv = grid.fcor2 + vorticity
        return cls(grid, time, pv=pv)

    # Arithmetic operators

    def add_wind(self, other):
        """Wind-based addition of atmospheric states.

        Parameters:
            other (tuple | :py:class:`State`): other wind field in the
                addition. Specify directly as (u,v)-tuple or extracted from
                a given :py:class:`State` object.

        Returns:
            New :py:class:`State` instance.
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

        Parameters:
            other (array | :py:class:`State`): other vorticity field in the
                addition. Specify directly as array or extracted from
                a given :py:class:`State` object.

        Returns:
            New :py:class:`State` instance.
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
        """PV flux in spectral space ``(q·u, q·v)``."""
        return self.grid.divergence_spectral(self.pv * self.u, self.pv * self.v)

    @property
    def vorticity(self):
        """Relative vorticity."""
        return self.pv - self.grid.fcor2

    @property
    def vorticity_spectral(self):
        return self.pv_spectral - self.grid.fcor2_spectral

    @property
    def enstrophy(self):
        """Enstropy = ``0.5 * q²`` where ``q`` is PV."""
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
        """Zonalized PV profile on the regular latitude grid.

        See :py:meth:`Grid.zonalize`.
        """
        return self.grid.zonalize(self.pv, interpolate=True)

    @property
    def fawa(self):
        """Finite-Amplitude Wave Activity on the regular grid.

        See :py:func:`diagnostics.fawa`.
        """
        # Instead of an extra computation with diagnostics.fawa, use that
        # FAWA is the zonal average of FALWA (Huang and Nakamura 2016)
        return self.falwa.mean(axis=ZONAL)

    @_cached_property
    def falwa(self):
        """Finite-Amplitude Local Wave Activity on the regular grid.

        See :py:func:`diagnostics.falwa`.
        """
        return diagnostics.falwa(self, interpolate=True)

    @property
    def falwa_filtered(self):
        """Finite-Amplitude Local Wave Activity, phase-filtered based on v.

        The FALWA field is filtered based on the doubled dominant wavenumber of
        the meridional wind obtained from Fourier analysis at each latitude as
        in Ghinassi et al. (2020).

        See :py:func:`diagnostics.falwa`,
        :py:func:`diagnostics.dominant_wavenumber_fourier` and
        :py:func:`diagnostics.filter_by_wavenumber`.
        """
        dominant_wavenumber = diagnostics.dominant_wavenumber_fourier(self.v, self.grid)
        return diagnostics.filter_by_wavenumber(self.falwa, 2*dominant_wavenumber)

    @property
    def v_envelope_hilbert(self):
        """Envelope of wave packets based on the Hilbert transform.

        See :py:func:`diagnostics.envelope_hilbert`.
        """
        return diagnostics.envelope_hilbert(self.v, (2, 10))

    @property
    def stationary_wavenumber(self):
        """Non-dimensionalised stationary (zonal) wavenumber (``Ks``, complex).

        See :py:func:`diagnostics.stationary_wavenumber`.
        """
        return diagnostics.stationary_wavenumber(self)

    def extract_ks_waveguides(self, *args, **kwargs):
        """Shortcut to :py:func:`diagnostics.extract_ks_waveguides`."""
        return diagnostics.extract_ks_waveguides(self, *args, **kwargs)

    # Shortcut to model integration

    def run(self, model, *args, **kwargs):
        """Shortcut to :py:meth:`BarotropicModel.run`"""
        return model.run(self, *args, **kwargs)

    # Shortcuts to plotting. Do not return figures here, so user does not have
    # to put semicolons in Jupyter notebooks to suppress double figure output.

    @property
    def plot(self):
        """Interface to plot presets from :py:mod:`plot` for interactive use.

        Provides shortcuts to:

        - :py:func:`plot.rwp_diagnostics`
        - :py:func:`plot.summary`
        - :py:func:`plot.wave_activity`
        - :py:func:`plot.waveguides`

        Plotters accessed through this interface do not return the **Figure**
        instance like their standalone versions from :py:mod:`plot`, which
        means that the created image is not displayed twice in the output of
        a jupyter notebook cell (once captured from within matplotlib and once
        as the returned object of the cell).
        """
        return StatePlotter(self)

    # Connection to other packages

    def as_hn2016(self, **barofield_kwargs):
        """Convert state to hn2016_falwa Barotropic state.

        Parameters:
            barofield_kwargs: Keyword arguments given to the constructor.

        Returns:
            :py:class:`hn2016_falwa.barotropic_field.BarotropicField` instance.

        .. note::
            In :py:mod:`hn2016_falwa` latitude starts at the South Pole, so
            array contents are meridionally flipped compared to :py:mod:`barotropic`.

        See :py:meth:`State.from_hn2016` for the inverse operation.

        Requires :py:mod:`hn2016_falwa`.
        """
        from hn2016_falwa.barotropic_field import BarotropicField
        pv = self.pv
        # hn2016_falwa expects latitudes to start at the South Pole
        xlon = self.grid.lon
        ylat = np.flip(self.grid.lat)
        pvud = np.flipud(self.pv)
        return BarotropicField(xlon, ylat, pv_field=pvud, **barofield_kwargs)

    @classmethod
    def from_hn2016(cls, grid, time, barofield):
        """Take PV from :py:mod:`hn2016_falwa` barotropic state object.

        Parameters:
            grid (:py:class:`Grid` | None): Grid used for instantiated
                :py:class:`State`. If ``None``, a new :py:class:`Grid` matching
                the input is created.
            time (number | datetime): Valid time as number in seconds or
                a datetime-like object.
            barofield (:py:class:`hn2016_falwa.barotropic_field.BarotropicField`):
                Instance from which to take PV field for instantiation.

        Returns:
            New :py:class:`State` instance.

        See :py:meth:`State.as_hn2016` for the inverse operation.
        """
        if grid is None:
            dlat = np.rad2deg(barofield.dphi[0])
            grid = Grid(resolution=dlat)
        # ...
        assert barofield.nlon == grid.nlon
        assert barofield.nlat == grid.nlat
        # TODO: check planet radius too??
        # Need to flip PV field so that latitude starts at the North Pole 
        pv = np.flipud(barofield.pv_field)
        return cls(grid, time, pv=pv)


class StatePlotter:

    def __init__(self, state):
        self._state = state

    def summary(self, *args, **kwargs):
        from . import plot
        plot.summary(self._state, *args, **kwargs)

    def wave_activity(self, *args, **kwargs):
        from . import plot
        plot.wave_activity(self._state, *args, **kwargs)

    def rwp_diagnostics(self, *args, **kwargs):
        from . import plot
        plot.rwp_diagnostics(self._state, *args, **kwargs)

    def waveguides(self, *args, **kwargs):
        from . import plot
        plot.waveguides(self._state, *args, **kwargs)
