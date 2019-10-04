import numpy as np
from .constants import ZONAL


# TODO take pv and grid as arguments, not state
def fawa(state, levels=None, interpolate=None):
    """Finite-amplitude wave activity according to Nakamura and Zhu (2010)
    
    TODO `levels`
    TODO `interpolate`
    """
    grid = state.grid
    # ...
    qq, yy = grid.zonalize_eqlat(state.pv, levels=levels, interpolate=None, quad="sptrapz")
    # ...
    q_int = np.vectorize(lambda q: grid.quad_sptrapz(state.pv, state.pv - q))
    y_int = np.vectorize(lambda y: grid.quad_sptrapz(state.pv, grid.lat - y))
    wa = (q_int(qq) - y_int(yy)) / grid.circumference(yy)
    # ...
    if interpolate is not None:
        qq = np.interp(interpolate, yy, wa, left=0, right=0)
        yy = interpolate
    return qq, yy


# TODO take pv and grid as arguments, not state
def falwa(state, levels=None, interpolate=None):
    """Local Finite-amplitude wave activity according to Huang and Nakamura (2016)
    
    TODO `levels`
    """
    grid = state.grid
    # ...
    # TODO interpolate later
    qq, yy = grid.zonalize_eqlat(state.pv, levels=levels, interpolate=interpolate, quad="sptrapz")
    # ...
    q_int = np.frompyfunc(lambda q, y: grid.quad_sptrapz_meridional(state.pv - q, state.pv - q), 2, 1)
    y_int = np.frompyfunc(lambda q, y: grid.quad_sptrapz_meridional(state.pv - q, grid.lat - y), 2, 1)
    # ...
    icoslat = 1. / np.cos(np.deg2rad(yy))
    icoslat[ 0] = 0.
    icoslat[-1] = 0.
    # ...
    return np.stack(icoslat * (q_int(qq, yy) - y_int(qq, yy)))


# TODO take pv and grid as arguments, not state
def falwa_hn2016(state, normalize_icos=True):
    """Finite-amplitude local wave activity according to Huang and Nakamura (2016)
    
    Uses implementation of package `hn2016_falwa`.
    See: https://github.com/csyhuang/hn2016_falwa
    """
    from hn2016_falwa.oopinterface import BarotropicField
    # hn2016_falwa expects latitudes to start at south pole
    xlon = state.grid.longitudes
    ylat = np.flip(state.grid.latitudes)
    bf = BarotropicField(xlon, ylat, pv_field=np.flipud(state.pv))
    # Extract local wave activity field and flip back
    lwa = np.flipud(bf.lwa)
    # hn2016_falwa does not normalize with cosine of latitude by default
    if normalize_icos:
        icoslat = 1. / np.cos(state.grid.phi)
        icoslat[ 0,:] = 0.
        icoslat[-1,:] = 0.
        lwa = icoslat * lwa
    return lwa


def dominant_wavenumber(field, grid, n_scales=120, smoothing=(9, 31)):
    """"""
    import pywt
    from scipy import signal
    # Truncate zonal fourier spectrum of meridional wind at wavenumber 20
    xfft = np.fft.rfft(field, axis=ZONAL)
    xfft[:,20:] = 0.
    x = np.fft.irfft(xfft, axis=ZONAL)  
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
    # Smooth dominant wavenumber with 2D-Hann window
    # TODO specify smoothing in °lat/lon instead of number of gridpoints
    smooth_lon, smooth_lat = smoothing
    hann2d = np.outer(signal.windows.hann(smooth_lat), signal.windows.hann(smooth_lon))
    hann2d = hann2d / np.sum(hann2d)
    # TODO boundary="wrap" is not correct in meridional direction as it
    # connects north- and south pole
    return signal.convolve2d(dom_wavenum, hann2d, mode="same", boundary="wrap")


def filter_by_wavenumber(field, wavenumber, grid):
    """"""
    from scipy import signal
    nlon = grid.shape[ZONAL]
    # ...
    wavenumber = np.clip(wavenumber, 1., None)
    # ...
    convs = []
    for n in range(1, nlon + 2, 2):
        hann = signal.hann(n)
        hann = hann / np.sum(hann)
        hann = hann[None,:]
        # ...
        convs.append(signal.convolve2d(field, hann, mode="same"))
    convs = np.stack(convs)
    # ...
    # TODO accept wavenumber as single number
    idx = np.floor(nlon / wavenumber).astype(int) // 2
    # ...
    ii, jj = np.indices(grid.shape)
    return convs[idx, ii, jj]

