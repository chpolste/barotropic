"""Plot presets and plotting helper functions.

Plotting requires `matplotlib`. If `matplotlib` is not available, a warning
will be emitted on import but some of the helper functions will still work.

All plot presets are also accessible for interactive use as methods of the
`barotropic.State.plot` interface.
"""

from functools import partial
from numbers import Number
import warnings
import numpy as np
from .constants import ZONAL as _ZONAL, MERIDIONAL as _MERIDIONAL
from .constants import HOUR as _HOUR
from . import diagnostic

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mpl_ticker
except ImportError:
    warnings.warn("matplotlib is not available to barotropic, plotting utilities will only work partially.")


# Data processing

def reduce_vectors(x, y, u, v, ny):
    """Reduce the density of points in the inputs by slicing.

    Use to slice a vector field with components `u` and `v` components and the
    corresponding coordinates `x` and `y` consistently, e.g. to reduce the
    number of arrows shown in a quiver plot. `ny` controls how many points are
    (approximately) kept in the y-dimension. The x-dimension is sliced with the
    same stride.

    Returns the reduced `x`, `y`, `u` and `v` in a tuple.
    """
    spacing = max(y.shape[_MERIDIONAL] // ny, 1)
    slicer = slice(spacing // 2, None, spacing), slice(spacing // 2, None, spacing)
    return x[slicer], y[slicer], u[slicer], v[slicer]


def hovmoellerify(states, f):
    """Prepare `states` for plotting as a Hovmöller diagram.

    - `states` is a list of `State` objects, which should be ordered
      chronologically
    - `f` is a function that is applied to each `State` in `states` and should
      return either a latitudinal or longitudinal 1D cross-section.

    Returns a tuple containing x-coordinates, y-coordinates and the Hovmöller
    field. If `f` returns a latitudinal cross-section (determined by the length
    of the vector), time is on the x-axis and latitude on the y-axis. Otherwise
    time is on the y-axis and longitude on the x-axis.
    """
    if len(states) == 0:
        raise ValueError("no states given")
    grid = states[0].grid
    times = np.array([state.time for state in states])
    fields = [f(state) for state in states]
    if fields[0].size == grid.nlon:
        return grid.lons, times, np.stack(fields)
    if fields[0].size == grid.nlat:
        return times, grid.lats, np.stack(fields).T
    raise ValueError("dimension mismatch: output of reduce has to match number of lons or lats")


def roll_lons(lons, center=180):
    """Center 2D fields around the given meridian.
    
    Returns a roll function that should be applied to 2D fields before plotting
    and a `configure_lon_x` function set up to label the longitudes axis
    correctly when using the unmodified lons from the grid during plotting.
    """
    center = center % 360
    cur_mid = lons.size // 2
    new_mid = np.argmin(np.abs(lons - center))
    shift = cur_mid - new_mid
    return partial(np.roll, shift=shift, axis=_ZONAL), partial(configure_lon_x, offset=lons[shift])


# Plot styling

def symmetric_levels(x, n=10, ext=None):
    """Generate `n` contour levels for field `x` that are symmetric around 0.
    
    `ext` is an override for the absolute value of the outermost levels.
    """
    if ext is None:
        ext = max(abs(np.min(x)), abs(np.max(x)))
    if ext == 0.:
        return np.array([-1., 1.])
    return np.linspace(-ext, ext, n)


def configure_lon_x(ax, offset=0):
    """Set up the x-axis ticks of `ax` to display longitude."""
    #ax.xaxis.set_major_formatter(mpl_ticker.StrMethodFormatter("{x:.0f}°"))
    ax.set_xticks(np.arange(0, 360, 30))
    ax.set_xticklabels(["{:.0f}°".format((x - offset) % 360) for x in ax.get_xticks()])


def configure_lat_y(ax, hemisphere):
    """Set up the y-axis ticks of `ax` to display latitude."""
    ax.yaxis.set_major_formatter(mpl_ticker.StrMethodFormatter("{x:.0f}°"))
    ax.set_ylim(0 if hemisphere == "N" else -90, 0 if hemisphere == "S" else 90)


def set_title_time(ax, time, loc="right"):
    """Put the formatted `time` into the title of `ax` at location `loc`.
    
    Shows `time` given as a number of seconds into hours and `time` given as
    a `datetime.datetime` in ISO format.
    """
    ax.set_title(
        "t = {:.1f} h".format(time / _HOUR) if isinstance(time, Number) else time.isoformat(), 
        loc=loc
    )


# Predefined figures

def summary(state, figsize=(11, 7), hemisphere="both", center_lon=180, pv_cmap="viridis",
        pv_max=None, v_max=None):
    """4-panel plot showing the model state in terms of vorticity and wind."""
    grid = state.grid
    roll, configure_lon_x = roll_lons(grid.lons, center_lon)
    # Scale PV to 10e-4 1/s
    pv = 10000 * state.pv
    if pv_max is not None:
        pv_max = pv_max * 10000
    # Plot 2 rows with 2 panels each
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=figsize, gridspec_kw={
        "width_ratios": (4, 10)
    })
    # Panel: zonal mean vorticity line plot
    ax11.vlines([0], -90, 90, linestyle="--", linewidth=0.5, color="#666666")
    # Planetary vorticity
    zmpv = np.mean(grid.fcor * 10000, axis=_ZONAL)
    ax11.plot(zmpv, grid.lats, color="#999999", label="pla.")
    # Zonal mean relative vorticity
    zmrv = np.mean(state.vorticity * 10000, axis=_ZONAL)
    ax11.plot(zmrv, grid.lats, color="#006699", label="rel.")
    # Zonal mean absolute (=potential) vorticity
    zmav = np.mean(pv, axis=_ZONAL)
    ax11.plot(zmav, grid.lats, color="#000000", label="pot.")
    configure_lat_y(ax11, hemisphere)
    ax11.legend(loc="upper left")
    ax11.set_title("zonal mean vort. [$10^{-4} \\mathrm{s}^{-1}$]", loc="left")
    # Panel: PV and wind vectors
    pv_levels = symmetric_levels(pv, 11 if hemisphere == "both" else 17, ext=pv_max)
    pvc = ax12.contourf(grid.lons, grid.lats, roll(pv), cmap=pv_cmap, levels=pv_levels, extend="both")
    fig.colorbar(pvc, ax=ax12)
    n_vectors = 13 if hemisphere == "both" else 21
    if not (np.linalg.norm(state.u) == 0. and np.linalg.norm(state.v) == 0.):
        ax12.quiver(*reduce_vectors(grid.lon, grid.lat, state.u, state.v, n_vectors))
    configure_lon_x(ax12)
    configure_lat_y(ax12, hemisphere)
    ax12.set_title("PV [$10^{-4} \\mathrm{s}^{-1}$] and wind vectors", loc="left")
    set_title_time(ax12, state.time)
    # Panel: Zonal mean zonal wind line plot
    ax21.vlines([0], -90, 90, linestyle="--", linewidth=0.5, color="#666666")
    zmu = np.mean(state.u, axis=_ZONAL)
    ax21.plot(zmu, grid.lats, color="#000000")
    ax21.set_title("mean zonal wind [$\\mathrm{m} \\mathrm{s}^{-1}$]", loc="left")
    configure_lat_y(ax21, hemisphere)
    # Panel: Meridional wind and streamfunction
    v_levels = symmetric_levels(state.v, 10, ext=v_max)
    pvc = ax22.contourf(grid.lons, grid.lats, roll(state.v), levels=v_levels, cmap="RdBu_r", extend="both")
    fig.colorbar(pvc, ax=ax22)
    psi = state.streamfunction
    psi_levels = np.linspace(np.min(psi), np.max(psi), 10 if hemisphere == "both" else 14)
    if psi_levels[0] != psi_levels[-1]:
        ax22.contour(grid.lons, grid.lats, roll(psi), levels=psi_levels, linestyles="-", colors="k")
    configure_lon_x(ax22)
    configure_lat_y(ax22, hemisphere)
    ax22.set_title("meridional wind [$\\mathrm{m} \\mathrm{s}^{-1}$] and streamfunction", loc="left")
    set_title_time(ax22, state.time)
    fig.tight_layout()
    return fig


def wave_activity(state, figsize=(11, 7), hemisphere="both", center_lon=180, falwa_cmap="YlOrRd"):
    """4-panel plot with Finite-Amplitude Wave Activity and PV diagnostics."""
    grid = state.grid
    roll, configure_lon_x = roll_lons(grid.lons, center_lon)
    # Scale PV to 10e-4 1/s
    pv = 10000 * state.pv
    # Plot 2 rows with 2 panels each
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=figsize, gridspec_kw={
        "width_ratios": (4, 10)
    })
    # Panel: zonal mean PV and zonalized PV
    ax11.vlines([0], -90, 90, linestyle="--", linewidth=0.5, color="#666666")
    # Zonalized PV
    eqlat_levels = 2 * grid.nlat
    zlpv, _ = grid.zonalize_eqlat(pv, levels=eqlat_levels, interpolate=grid.lats)
    ax11.plot(zlpv, grid.lats, color="#999999", label="equiv. lat.")
    # Zonal mean PV
    zmpv = np.mean(pv, axis=_ZONAL)
    ax11.plot(zmpv, grid.lats, color="#000000", label="mean")
    configure_lat_y(ax11, hemisphere)
    ax11.legend(loc="upper left")
    ax11.set_title("zonalized PV [$10^{-4} \\mathrm{s}^{-1}$]", loc="left")
    # Panel: PV and its deviation from zonalized PV
    devpv = pv - zlpv[:,None]
    devpvc = ax12.contourf(grid.lon, grid.lat, roll(devpv), levels=symmetric_levels(devpv, 10), cmap="RdBu")
    devpvb = fig.colorbar(devpvc, ax=ax12)
    pv_levels = np.linspace(np.min(pv), np.max(pv), 10 if hemisphere == "both" else 14)[1:-1]
    ax12.contour(grid.lon, grid.lat, roll(pv), colors="k", linestyles="-", levels=pv_levels)
    configure_lon_x(ax12)
    configure_lat_y(ax12, hemisphere)
    ax12.set_title("deviation from equiv. lat. PV [$10^{-4} \\mathrm{s}^{-1}$] and PV", loc="left")
    set_title_time(ax12, state.time)
    # Panel: FAWA
    ax21.vlines([0], -90, 90, linestyle="--", linewidth=0.5, color="#666666")
    ax21.plot(state.fawa, grid.lats, color="#000000")
    ax21.set_title("FAWA [$m s^{-1}$]", loc="left")
    configure_lat_y(ax21, hemisphere)
    # Panel: FALWA
    wac = ax22.contourf(grid.lon, grid.lat, roll(state.falwa), cmap=falwa_cmap)
    fig.colorbar(wac, ax=ax22)
    ax22.contour(grid.lon, grid.lat, roll(pv), linestyles="-", colors="k", levels=pv_levels)
    configure_lon_x(ax22)
    configure_lat_y(ax22, hemisphere)
    ax22.set_title("FALWA [$\\mathrm{m} \\mathrm{s}^{-1}$] and PV", loc="left")
    set_title_time(ax22, state.time)
    fig.tight_layout()
    return fig


def rwp_diagnostic(state, figsize=(8, 10.5), hemisphere="both", center_lon=180, v_max=None,
        rwp_max=None, rwp_cmap="YlOrRd"):
    """3-panel plot with Rossby wave packet diagnostics."""
    grid = state.grid
    roll, configure_lon_x = roll_lons(grid.lons, center_lon)
    # Plot 3 panels in 1 column
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    # Meridional wind and dominant wavenumber
    ax1.set_title("meridional wind [$\\mathrm{m} \\mathrm{s}^{-1}$] and dominant wavenumber", loc="left")
    v = state.v
    v_levels = symmetric_levels(v, 10, ext=v_max)
    ctf = ax1.contourf(grid.lon, grid.lat, roll(v), levels=v_levels, cmap="RdBu_r", extend="both")
    fig.colorbar(ctf, ax=ax1)
    dwn = state.dominant_wavenumber
    dwn_levels = np.arange(1, 11)
    if np.min(dwn) != np.max(dwn):
        ct = ax1.contour(grid.lon, grid.lat, roll(dwn), levels=dwn_levels, colors="k", linestyles="-", linewidths=1)
        ax1.clabel(ct, ct.levels, inline=True, fmt="%d")
    # Common colorbar range for envelope and filtered FALWA
    env = state.envelope_hilbert
    ffalwa = state.falwa_filtered
    if rwp_max is None:
        rwp_max = max(np.max(env), np.max(ffalwa))
    rwp_levels = np.linspace(0, rwp_max, 6)
    # RWP envelope and streamfunction
    ax2.set_title("RWP envelope [$\\mathrm{m} \\mathrm{s}^{-1}$] and streamfunction", loc="left")
    ctf = ax2.contourf(grid.lon, grid.lat, roll(env), cmap=rwp_cmap, levels=rwp_levels, extend="max")
    fig.colorbar(ctf, ax=ax2)
    psi = state.streamfunction
    psi_levels = np.linspace(np.min(psi), np.max(psi), 10 if hemisphere == "both" else 14)
    if psi_levels[0] != psi_levels[-1]:
        ax2.contour(grid.lon, grid.lat, roll(psi), levels=psi_levels, linestyles="-", colors="k")
    # Filtered FALWA and PV
    ax3.set_title("filtered FALWA [$\\mathrm{m} \\mathrm{s}^{-1}$] and PV", loc="left")
    ctf = ax3.contourf(grid.lon, grid.lat, roll(ffalwa), cmap=rwp_cmap, levels=rwp_levels, extend="max")
    fig.colorbar(ctf, ax=ax3)
    pv = state.pv
    pv_levels = np.linspace(np.min(pv), np.max(pv), 10 if hemisphere == "both" else 14)[1:-1]
    ax3.contour(grid.lon, grid.lat, roll(pv), colors="k", linestyles="-", levels=pv_levels)
    # Common styling
    for ax in (ax1, ax2, ax3):
        configure_lon_x(ax)
        configure_lat_y(ax, hemisphere)
        set_title_time(ax, state.time)
    fig.tight_layout()
    return fig


def waveguides(state, k_waveguides=None, hemisphere="both", xlim_bounds=(-5, 15),
        legend_loc=None):
    """2-panel plot showing stationary wavenumber and WKB waveguide diagnostics."""
    grid = state.grid
    # 2-Panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # Zonal-mean zonal wind
    ax1.set_title("zonal wind [$\\mathrm{m} \\mathrm{s}^{-1}$]", loc="left")
    ax1.plot(np.mean(state.u, axis=_ZONAL), grid.lats, color="#000000")
    # Stationary wavenumber and waveguides
    ks = state.stationary_wavenumber
    ax2.set_title("stationary wavenumber $K_s$", loc="left")
    # Wavenumbers for which waveguides are highlighted
    if k_waveguides is None:
        k_waveguides = tuple()
    if isinstance(k_waveguides, Number):
        k_waveguides = (k_waveguides,)
    for k in k_waveguides:
        # Collect boundary points of waveguides, intersperse with NaN values
        # which prevents connection between individual waveguides
        wg_coords = []
        for x, y in diagnostic.extract_waveguides(ks, k, grid=grid):
            wg_coords += [np.nan, x, y]
        # Let matplotlib choose the color
        ax2.plot([k]*len(wg_coords), wg_coords, label="k={}".format(k))
    # Display color-wavenumber labels if desired
    if legend_loc is not None:
        ax2.legend(loc=legend_loc)
    # Stationary wavenumber curve
    ax2.plot(np.real(ks) - np.imag(ks), grid.lats, color="#000000")
    # Keep wavenumber limits in given bounds
    xmin, xmax = ax2.get_xlim()
    xmin = max(xlim_bounds[0], xmin)
    xmax = min(xlim_bounds[1], xmax)
    ax2.set_xlim((xmin, xmax))
    # Make sure wavenumber ticks are integers
    xmin = np.ceil(xmin)
    xmax = np.ceil(xmax)
    step = np.ceil((xmax - xmin) / 10)
    xmin += xmin % step # guarantee that 0 is ticked
    ax2.set_xticks(np.arange(xmin, xmax, step))
    # Style y-axes as latitude and add a vertical line at 0
    for ax in (ax1, ax2):
        configure_lat_y(ax, hemisphere)
        ax.axvline(0, linestyle="--", linewidth=0.5, color="#666666")
    fig.tight_layout()
    return fig

