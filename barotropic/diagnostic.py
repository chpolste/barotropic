"""Diagnostic fields for the barotropic model"""

import numpy as np
from .constants import ZONAL, MERIDIONAL
# To reduce non-optional dependencies, additional packages are loaded on demand
# in some methods. They include: pywt, scipy.signal, hn2016_falwa



def _get_grid_vars(which, grid, field):
    # If no grid is given, assume field is State-like and take its grid
    if grid is None:
        grid = field.grid
        outs = [getattr(field, attr) for attr in which]
    else:
        outs = [None] * len(which)
        outs[0] = field
    return (grid, *outs)

def _restrict_fourier_zonal(field, kmin, kmax):
    """Restrict the zonal Fourier spectrum to a range of wavenumbers
    
    Reduce the Fourier spectrum of real-valued `field` to contributions from
    wavenumbers `kmin` to `kmax` (both inclusive).
    """
    xfft = np.fft.rfft(field, axis=ZONAL)
    xfft[:,      :kmin] = 0.
    xfft[:,kmax+1:    ] = 0.
    return np.fft.irfft(xfft, axis=ZONAL)  


def fawa(pv_or_state, grid=None, levels=None, interpolate=None):
    """Finite-amplitude wave activity according to Nakamura and Zhu (2010)
    
    `levels` specifies the number of contours used in the equivalent latitude
    zonalization. By default, FAWA is returned on the computed equivalent
    latitudes. To obtain FAWA interpolated to a specific set of latitudes,
    specify these as an array with the `interpolate` argument.
    """
    grid, pv = _get_grid_vars(["pv"], grid, pv_or_state)
    # Compute zonalized background state of PV
    qq, yy = grid.zonalize_eqlat(pv, levels=levels, interpolate=None, quad="sptrapz")
    # Use formulation that integrates PV over areas north of PV
    # contour/equivalent latitude and then computes difference
    q_int = np.vectorize(lambda q: grid.quad_sptrapz(pv, pv - q))
    y_int = np.vectorize(lambda y: grid.quad_sptrapz(pv, grid.lat - y))
    # Normalize by zonal circumference at each latitude
    fawa = (q_int(qq) - y_int(yy)) / grid.circumference(yy)
    # Interpolate to a given set of latitudes if specified
    if interpolate is not None:
        fawa = np.interp(interpolate, yy, fawa, left=0, right=0)
        yy = interpolate
    return fawa, yy


def falwa(pv_or_state, grid=None, levels=None, interpolate=None):
    """Finite-amplitude local wave activity according to Huang and Nakamura (2016)
    
    `levels` specifies the number of contours used in the equivalent latitude
    zonalization. By default, FALWA is returned on a longitude/equivalent
    latitude grid. To obtain FALWA interpolated to a specific set of latitudes,
    specify these as an array with the `interpolate` argument.
    """
    grid, pv = _get_grid_vars(["pv"], grid, pv_or_state)
    # Compute zonalized background state of PV
    qq, yy = grid.zonalize_eqlat(pv, levels=levels, quad="sptrapz")
    # Use formulation that integrates PV over areas north of PV
    # contour/equivalent latitude and then computes difference
    q_int = np.frompyfunc(lambda q, y: grid.quad_sptrapz_meridional(pv - q, pv - q), 2, 1)
    y_int = np.frompyfunc(lambda q, y: grid.quad_sptrapz_meridional(pv - q, grid.lat - y), 2, 1)
    # Stack meridional integrals (normalized by cosine of latitude) along zonal
    # direction to obtain full field again
    falwa = np.stack((q_int(qq, yy) - y_int(qq, yy)) / np.cos(np.deg2rad(yy)))
    # Interpolate to a given set of latitudes if specified
    if interpolate is not None:
        interp = lambda _: np.interp(interpolate, yy, _, right=0., left=0.)
        falwa = np.apply_along_axis(interp, MERIDIONAL, falwa)
        yy = interpolate
    return falwa, yy


def falwa_hn2016(pv_or_state, grid=None, normalize_icos=True):
    """Finite-amplitude local wave activity according to Huang and Nakamura (2016)

    Values of FALWA are returned on the grid.
    
    Uses the implementation of package `hn2016_falwa`.
    See: https://github.com/csyhuang/hn2016_falwa
    """
    from hn2016_falwa.oopinterface import BarotropicField
    grid, pv = _get_grid_vars(["pv"], grid, pv_or_state)
    # hn2016_falwa expects latitudes to start at south pole
    xlon = grid.lons
    ylat = np.flip(grid.lats)
    bf = BarotropicField(xlon, ylat, pv_field=np.flipud(pv))
    # Extract local wave activity field and flip back
    lwa = np.flipud(bf.lwa)
    # hn2016_falwa does not normalize with cosine of latitude by default
    if normalize_icos:
        icoslat = 1. / np.cos(grid.phi)
        icoslat[ 0,:] = 0. # handle div/0 problem at 1 / cos( 90°)
        icoslat[-1,:] = 0. # handle div/0 problem at 1 / cos(-90°)
        lwa = icoslat * lwa
    return lwa


def dominant_wavenumber(field, grid, n_scales=120, smoothing=(21, 7)):
    """Compute the dominant zonal wavenumber at every gridpoint of field

    Implements the procedure of Ghinassi et al. (2018) based on wavelet
    analysis.

    `n_scales` determines the number of scales used in the continuous wavelet
    transform. The `smoothing` parameters determine the full width at half
    maximum of the Hann filter in zonal and meridional direction in degrees
    longitude/latitude.
    
    Requires `pywt` (version >= 1.1.0) and `scipy.signal`.
    """
    import pywt
    from scipy import signal
    # Truncate zonal fourier spectrum of meridional wind after wavenumber 20
    x = _restrict_fourier_zonal(field, 0, 20)
    # Triplicate field to avoid boundary issues (pywt.cwt does not support
    # periodic mode as of version 1.1)
    x = np.hstack([x, x, x])
    # Use the complex Morlet wavelet
    morlet = pywt.ContinuousWavelet("cmor1.0-1.0")
    # ...
    scales = 3 * 2**np.linspace(0, 6, n_scales)
    # Apply the continuous wavelet transform
    coef, freqs = pywt.cwt(x, scales=scales, wavelet=morlet, axis=ZONAL)
    # Extract the middle domain, throw away the periodic padding
    ii = coef.shape[-1] // 3
    coef = coef[:,:,ii:-ii]
    # Obtain the power spectrum
    power = np.abs(coef)**2
    # Determine wavenumbers from scales
    wavenum = pywt.scale2frequency(morlet, scales) * grid.shape[ZONAL]
    # Dominant wavenumber is that of maximum power in the spectrum
    dom_wavenum = wavenum[np.argmax(power, axis=0)]
    # Smooth dominant wavenumber with Hann windows. Window width is given as
    # full width at half maximum, which is half of the full width. Choose the
    # nearest odd number of gridpoints available to the desired value.
    smooth_lon, smooth_lat = smoothing
    hann_lon = signal.windows.hann(int(smooth_lon / 360 * grid.shape[ZONAL]) * 2 + 1)
    hann_lon = hann_lon / np.sum(hann_lon)
    hann_lat = signal.windows.hann(int(smooth_lat / 180 * grid.shape[MERIDIONAL]) * 2 + 1)
    hann_lat = hann_lat / np.sum(hann_lat)
    # Apply zonal filter first with periodic boundary
    dom_wavenum = signal.convolve2d(dom_wavenum, hann_lon[None,:], mode="same", boundary="wrap")
    # Then apply meridional filter with symmetrical boundary
    dom_wavenum = signal.convolve2d(dom_wavenum, hann_lat[:,None], mode="same", boundary="symm")
    return dom_wavenum


def filter_by_wavenumber(field, wavenumber):
    """Zonal Hann smoothing based on a space-dependent wavenumber field
    
    Apply Hann smoothing in zonal direction to the input `field` with the Hann
    window width governed at every gridpoint by the wavenumber at the same
    location in the `wavenumber` field.

    Used by Ghinassi et al. (2018) to filter the FALWA field, discounting phase
    information in order to diagnose wave packets as a whole.

    Requires `scipy.signal`.
    """
    from scipy import signal
    nlon = field.shape[ZONAL]
    # Limit wavenumbers to 1 and above. The Hann window at this point goes once
    # around the entire globe already.
    wavenumber = np.clip(wavenumber, 1., None)
    # Precompute the smoothed fields for all possible odd gridpoint-widths.
    # While this produces a lot of unused data, it allows to use more
    # numpy-accelerated operations which should outperform any pure-Python and
    # loop-based implementation. Widths and indices are mapped by 2n+1 -> n.
    convs = []
    for n in range(1, nlon + 2, 2):
        hann = signal.hann(n)
        # Normalize
        hann = hann / np.sum(hann)
        # Apply window along the zonal axis
        convs.append(signal.convolve2d(field, hann[None,:], mode="same", boundary="wrap"))
    # Stack into 3-dimensional array
    convs = np.stack(convs)
    # The window width in gridpoints is given by the total number of gridpoints in
    # zonal direction divided by the wavenumber. This width is transformed into
    # an index for the precomputed smoothed fields
    # TODO accept wavenumber as single number (find more general replacement
    #      for astype(int) method)
    idx = np.floor(nlon / wavenumber).astype(int) // 2
    ii, jj = np.indices(field.shape)
    return convs[idx,ii,jj]


def envelope_hilbert(field, wavenumber_min=2, wavenumber_max=10):
    """Compute envelope of wave packets using the Hilbert transform

    Applied to the merdional wind for RWP detection by Zimin et al. (2003).

    Requires `scipy.signal`.
    """
    from scipy import signal
    x = _restrict_fourier_zonal(field, wavenumber_min, wavenumber_max)
    return np.abs(signal.hilbert(x, axis=ZONAL))


def stationary_wavenumber(u_or_state, grid=None, order=None):
    """Non-dimensionalised stationary wavenumber a²Ks²"""
    grid, u, pv = _get_grid_vars(["u", "pv"], grid, u_or_state)
    # If u is a 2D field, calculate zonal mean
    if u.ndim != 1:
        u = np.mean(u, axis=ZONAL)
    # If no PV field is given, calculate zonal-mean PV from zonal-mean u
    if pv is None:
        rv = - grid.ddphi(u * np.cos(grid.phis), order=order) / grid.rsphere / np.cos(grid.phis)
        pv = grid.coriolis(grid.lats) + rv
    # If PV is a 2D field, calculate zonal mean
    elif pv.ndim != 1:
        pv = np.mean(pv, axis=ZONAL)
    # Calculate a²Ks²
    return np.cos(grid.phis)**2 * grid.rsphere * grid.ddphi(pv, order=order) / u
