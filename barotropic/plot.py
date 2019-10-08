import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
from .constants import ZONAL, MERIDIONAL, HOUR


# Data processing

def reduce_vectors(lon, lat, x, y, ny):
    spacing = max(lat.shape[MERIDIONAL] // ny, 1)
    slicer = slice(spacing // 2, None, spacing), slice(spacing // 2, None, spacing)
    return lon[slicer], lat[slicer], x[slicer], y[slicer]

def hovmoellerify(states, f):
    if len(states) == 0:
        raise ValueError("no states given")
    grid = states[0].grid
    times = np.array([state.time for state in states])
    fields = [f(state) for state in states]
    if fields[0].size == grid.longitudes.size:
        return grid.longitudes, times, np.stack(fields)
    if fields[0].size == grid.latitudes.size:
        return times, grid.latitudes, np.stack(fields).T
    raise ValueError("dimension mismatch: output of reduce has to match number of lons or lats")


# Plot styling

def symmetric_levels(x, n=10, ext=None):
    if ext is None:
        ext = max(abs(np.min(x)), abs(np.max(x)))
    return np.linspace(-ext, ext, n)

def configure_lon_x(ax):
    ax.xaxis.set_major_formatter(mpl_ticker.StrMethodFormatter("{x:.0f}°"))
    ax.set_xticks(np.arange(0, 360, 30))

def configure_lat_y(ax, hemisphere):
    ax.yaxis.set_major_formatter(mpl_ticker.StrMethodFormatter("{x:.0f}°"))
    ax.set_ylim(0 if hemisphere == "N" else -90, 0 if hemisphere == "S" else 90)


# Predefined figures

def summary(state, figsize=(11, 7), hemisphere="both", pv_cmap="viridis", pv_max=None, v_max=None):
    """Plot a summary of the model state in terms of vorticity and wind"""
    grid = state.grid
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
    zmpv = np.mean(grid.fcor * 10000, axis=ZONAL)
    ax11.plot(zmpv, grid.latitudes, color="#999999", label="pla.")
    # Zonal mean relative vorticity
    zmrv = np.mean(state.vorticity * 10000, axis=ZONAL)
    ax11.plot(zmrv, grid.latitudes, color="#006699", label="rel.")
    # Zonal mean absolute (=potential) vorticity
    zmav = np.mean(pv, axis=ZONAL)
    ax11.plot(zmav, grid.latitudes, color="#000000", label="pot.")
    configure_lat_y(ax11, hemisphere)
    ax11.legend(loc="upper left")
    ax11.set_title("zonal mean vort. [$10^{-4} \\mathrm{s}^{-1}$]", loc="left")
    # Panel: PV and wind vectors
    pv_levels = symmetric_levels(pv, 11 if hemisphere == "both" else 17, ext=pv_max)
    pvc = ax12.contourf(grid.lon, grid.lat, pv, cmap=pv_cmap, levels=pv_levels, extend="both")
    fig.colorbar(pvc, ax=ax12)
    n_vectors = 13 if hemisphere == "both" else 21
    ax12.quiver(*reduce_vectors(grid.lon, grid.lat, state.u, state.v, n_vectors))
    configure_lon_x(ax12)
    configure_lat_y(ax12, hemisphere)
    ax12.set_title("PV [$10^{-4} \\mathrm{s}^{-1}$] and wind vectors", loc="left")
    ax12.set_title("t = {:.1f} h".format(state.time / HOUR), loc="right")
    # Panel: Zonal mean zonal wind line plot
    ax21.vlines([0], -90, 90, linestyle="--", linewidth=0.5, color="#666666")
    zmu = np.mean(state.u, axis=ZONAL)
    ax21.plot(zmu, grid.latitudes, color="#000000")
    ax21.set_title("mean zonal wind [$\\mathrm{m} \\mathrm{s}^{-1}$]", loc="left")
    configure_lat_y(ax21, hemisphere)
    # Panel: Meridional wind and streamfunction
    v_levels = symmetric_levels(state.v, 10, ext=v_max)
    pvc = ax22.contourf(grid.lon, grid.lat, state.v, levels=v_levels, cmap="RdBu_r", extend="both")
    fig.colorbar(pvc, ax=ax22)
    psi = state.streamfunction
    psi_levels = np.linspace(np.min(psi), np.max(psi), 10 if hemisphere == "both" else 14)
    ax22.contour(grid.lon, grid.lat, psi, levels=psi_levels, linestyles="-", colors="k")
    configure_lon_x(ax22)
    configure_lat_y(ax22, hemisphere)
    ax22.set_title("meridional wind [$\\mathrm{m} \\mathrm{s}^{-1}$] and streamfunction", loc="left")
    ax22.set_title("t = {:.1f} h".format(state.time / HOUR), loc="right")
    fig.tight_layout()
    return fig

def wave_activity(state, figsize=(11, 7), hemisphere="both", falwa_cmap="YlOrRd"):
    """Plot finite-amplitude wave activity and PV"""
    grid = state.grid
    # Scale PV to 10e-4 1/s
    pv = 10000 * state.pv
    # Plot 2 rows with 2 panels each
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=figsize, gridspec_kw={
        "width_ratios": (4, 10)
    })
    # Panel: zonal mean PV and zonalized PV
    ax11.vlines([0], -90, 90, linestyle="--", linewidth=0.5, color="#666666")
    # Zonalized PV
    eqlat_levels = 2 * grid.latitudes.size
    zlpv, _ = grid.zonalize_eqlat(pv, levels=eqlat_levels, interpolate=grid.latitudes)
    ax11.plot(zlpv, grid.latitudes, color="#999999", label="equiv. lat.")
    # Zonal mean PV
    zmpv = np.mean(pv, axis=ZONAL)
    ax11.plot(zmpv, grid.latitudes, color="#000000", label="mean")
    configure_lat_y(ax11, hemisphere)
    ax11.legend(loc="upper left")
    ax11.set_title("zonalized PV [$10^{-4} \\mathrm{s}^{-1}$]", loc="left")
    # Panel: PV and its deviation from zonalized PV
    devpv = pv - zlpv[:,None]
    devpvc = ax12.contourf(grid.lon, grid.lat, devpv, levels=symmetric_levels(devpv, 10), cmap="RdBu")
    devpvb = fig.colorbar(devpvc, ax=ax12)
    pv_levels = np.linspace(np.min(pv), np.max(pv), 10 if hemisphere == "both" else 14)[1:-1]
    ax12.contour(grid.lon, grid.lat, pv, colors="k", linestyles="-", levels=pv_levels)
    configure_lon_x(ax12)
    configure_lat_y(ax12, hemisphere)
    ax12.set_title("deviation from equiv. lat. PV [$10^{-4} \\mathrm{s}^{-1}$] and PV", loc="left")
    ax12.set_title("t = {:.1f} h".format(state.time / HOUR), loc="right")
    # Panel: FAWA
    ax21.vlines([0], -90, 90, linestyle="--", linewidth=0.5, color="#666666")
    ax21.plot(state.fawa, grid.latitudes, color="#000000")
    ax21.set_title("FAWA [$m s^{-1}$]", loc="left")
    configure_lat_y(ax21, hemisphere)
    # Panel: FALWA
    wac = ax22.contourf(grid.lon, grid.lat, state.falwa, cmap=falwa_cmap)
    fig.colorbar(wac, ax=ax22)
    ax22.contour(grid.lon, grid.lat, pv, linestyles="-", colors="k", levels=pv_levels)
    configure_lon_x(ax22)
    configure_lat_y(ax22, hemisphere)
    ax22.set_title("FALWA [$\\mathrm{m} \\mathrm{s}^{-1}$] and PV", loc="left")
    ax22.set_title("t = {:.1f} h".format(state.time / HOUR), loc="right")
    fig.tight_layout()
    return fig

def rwp_diagnostic(state, figsize=(8, 10.5), hemisphere="both", v_max=None, rwp_max=None,
        rwp_cmap="YlOrRd"):
    """Plot Rossby wave packet diagnostics"""
    grid = state.grid
    # Plot 3 panels in 1 column
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    # Meridional wind and dominant wavenumber
    v = state.v
    v_levels = symmetric_levels(v, 10, ext=v_max)
    ctf = ax1.contourf(grid.lon, grid.lat, v, levels=v_levels, cmap="RdBu_r", extend="both")
    fig.colorbar(ctf, ax=ax1)
    dwn = state.dominant_wavenumber
    dwn_levels = np.arange(1, 11)
    ax1.set_title("meridional wind [$\\mathrm{m} \\mathrm{s}^{-1}$] and dominant wavenumber", loc="left")
    ct = ax1.contour(grid.lon, grid.lat, dwn, levels=dwn_levels, colors="k", linestyles="-", linewidths=1)
    ax1.clabel(ct, ct.levels, inline=True, fmt="%d")
    # Common colorbar range for envelope and filtered FALWA
    env = state.rwp_envelope
    ffalwa = state.falwa_filtered
    if rwp_max is None:
        rwp_max = max(np.max(env), np.max(ffalwa))
    rwp_levels = np.linspace(0, rwp_max, 6)
    # RWP envelope and streamfunction
    ax2.set_title("RWP envelope [$\\mathrm{m} \\mathrm{s}^{-1}$] and streamfunction", loc="left")
    ctf = ax2.contourf(grid.lon, grid.lat, env, cmap=rwp_cmap, levels=rwp_levels, extend="max")
    fig.colorbar(ctf, ax=ax2)
    psi = state.streamfunction
    psi_levels = np.linspace(np.min(psi), np.max(psi), 10 if hemisphere == "both" else 14)
    ax2.contour(grid.lon, grid.lat, psi, levels=psi_levels, linestyles="-", colors="k")
    # Filtered FALWA and PV
    ax3.set_title("filtered FALWA [$\\mathrm{m} \\mathrm{s}^{-1}$] and PV", loc="left")
    ctf = ax3.contourf(grid.lon, grid.lat, ffalwa, cmap=rwp_cmap, levels=rwp_levels, extend="max")
    fig.colorbar(ctf, ax=ax3)
    pv = state.pv
    pv_levels = np.linspace(np.min(pv), np.max(pv), 10 if hemisphere == "both" else 14)[1:-1]
    ax3.contour(grid.lon, grid.lat, pv, colors="k", linestyles="-", levels=pv_levels)
    # Common styling
    for ax in (ax1, ax2, ax3):
        configure_lon_x(ax)
        configure_lat_y(ax, hemisphere)
        ax.set_title("t = {:.1f} h".format(state.time / HOUR), loc="right")
    fig.tight_layout()
    return fig

