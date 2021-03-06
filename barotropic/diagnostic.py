"""Diagnostic fields for the barotropic model."""

import numpy as np
from .constants import ZONAL as _ZONAL, MERIDIONAL as _MERIDIONAL
# To reduce non-optional dependencies, some required packages are imported in
# some of these functions on demand.



def _get_grid_vars(which, grid, field):
    # If no grid is given, assume field is State-like and take its grid
    if grid is None:
        grid = field.grid
        outs = [getattr(field, attr) for attr in which]
    # Otherwise return grid and field as they are and None for other variables
    else:
        outs = [None] * len(which)
        outs[0] = np.asarray(field)
    return (grid, *outs)

def _restrict_fourier_zonal(field, kmin, kmax):
    """Restrict the zonal Fourier spectrum to a range of wavenumbers

    Reduce the Fourier spectrum of real-valued `field` to contributions from
    wavenumbers `kmin` to `kmax` (both inclusive).
    """
    xfft = np.fft.rfft(field, axis=_ZONAL)
    xfft[:,      :kmin] = 0.
    xfft[:,kmax+1:    ] = 0.
    return np.fft.irfft(xfft, axis=_ZONAL)


def fawa(pv_or_state, grid=None, levels=None, interpolate=None):
    """Finite-Amplitude Wave Activity according to Nakamura and Zhu (2010).

    - If the first parameter is not a `barotropic.State`, `grid` must be
      specified.
    - `levels` specifies the number of contours generated for the equivalent
      latitude zonalization.
    - By default, FAWA is returned on the computed equivalent latitudes. To
      obtain FAWA interpolated to a specific set of latitudes, specify these
      with the `interpolate` parameter.

    Returns a tuple containing FAWA and its latitude coordinates.
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
    """Finite-Amplitude Local Wave Activity according to Huang and Nakamura (2016).

    - If the first parameter is not a `barotropic.State`, `grid` must be
      specified.
    - `levels` specifies the number of contours generated for the equivalent
      latitude zonalization.
    - By default, FALWA is returned on an equivalent latitude by longitude
      grid. To obtain FALWA interpolated to a specific set of latitudes,
      specify these with the `interpolate` parameter.

    Returns a tuple containing FALWA and its latitude coordinates.
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
        falwa = np.apply_along_axis(interp, _MERIDIONAL, falwa)
        yy = interpolate
    return falwa, yy


def falwa_hn2016(pv_or_state, grid=None, normalize_icos=True):
    """Finite-Amplitude Local Wave Activity according to Huang and Nakamura (2016).

    Uses the implementation of [`hn2016_falwa`](https://github.com/csyhuang/hn2016_falwa).

    - If the first parameter is not a `barotropic.State`, `grid` must be
      specified.
    - By default, FALWA is normalized with the inverse of the cosine of
      latitude. To disable this normalization, set `normalize_icos=False`.

    Returns FALWA on the regular latitude/longitude grid.
    """
    from hn2016_falwa.barotropic_field import BarotropicField
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
    """Dominant zonal wavenumber at every gridpoint of field.

    Implements the procedure of Ghinassi et al. (2018) based on a wavelet
    analysis of the input field.

    - `n_scales` determines the number of scales used in the continuous wavelet
      transform.
    - `smoothing` determines the full width at half maximum of the Hann filter
      in zonal and meridional direction in degrees longitude/latitude.

    Returns the gridded dominant zonal wavenumber.

    Requires `pywt` (version >= 1.1.0) and `scipy`.
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
    coef, freqs = pywt.cwt(x, scales=scales, wavelet=morlet, axis=_ZONAL)
    # Extract the middle domain, throw away the periodic padding
    ii = coef.shape[-1] // 3
    coef = coef[:,:,ii:-ii]
    # Obtain the power spectrum
    power = np.abs(coef)**2
    # Determine wavenumbers from scales
    wavenum = pywt.scale2frequency(morlet, scales) * grid.shape[_ZONAL]
    # Dominant wavenumber is that of maximum power in the spectrum
    dom_wavenum = wavenum[np.argmax(power, axis=0)]
    # Smooth dominant wavenumber with Hann windows. Window width is given as
    # full width at half maximum, which is half of the full width. Choose the
    # nearest odd number of gridpoints available to the desired value.
    smooth_lon, smooth_lat = smoothing
    hann_lon = signal.windows.hann(int(smooth_lon / 360 * grid.shape[_ZONAL]) * 2 + 1)
    hann_lon = hann_lon / np.sum(hann_lon)
    hann_lat = signal.windows.hann(int(smooth_lat / 180 * grid.shape[_MERIDIONAL]) * 2 + 1)
    hann_lat = hann_lat / np.sum(hann_lat)
    # Apply zonal filter first with periodic boundary
    dom_wavenum = signal.convolve2d(dom_wavenum, hann_lon[None,:], mode="same", boundary="wrap")
    # Then apply meridional filter with symmetrical boundary
    dom_wavenum = signal.convolve2d(dom_wavenum, hann_lat[:,None], mode="same", boundary="symm")
    return dom_wavenum


def filter_by_wavenumber(field, wavenumber):
    """Zonal Hann smoothing based on a space-dependent wavenumber field.
    
    Apply Hann smoothing in zonal direction to the input `field` with the Hann
    window width governed at every gridpoint by the wavenumber at the same
    location in the `wavenumber` field.

    Used by Ghinassi et al. (2018) to filter the FALWA field, discounting phase
    information in order to diagnose wave packets as a whole.

    Requires `scipy`.
    """
    from scipy import signal
    nlon = field.shape[_ZONAL]
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
    """Compute the envelope of wave packets with the Hilbert transform.

    Applied to the merdional wind for RWP detection by Zimin et al. (2003).

    Requires `scipy`.
    """
    from scipy import signal
    x = _restrict_fourier_zonal(field, wavenumber_min, wavenumber_max)
    return np.abs(signal.hilbert(x, axis=_ZONAL))


def stationary_wavenumber(u_or_state, grid=None, order=None, min_u=0.001, kind="complex"):
    """Non-dimensionalised stationary (zonal) wavenumber (`Ks`).

        Ks² = a²·βM/uM,

    where `a` is the radius of the sphere, `βM` is the Mercator projection
    gradient of zonal-mean PV and `uM` is the Mercator projection zonal-mean
    wind. See e.g. Hoskins and Karoly (1981), Wirth (2020).

    `Ks²` can be negative. It is set to positive or negative infinity where `u`
    is smaller than the threshold given by `min_u`. If `u` is given as a field
    or zonal profile and not via a `barotropic.State` in the first parameter,
    `βM` is calculated from the wind field using derivatives of order `order`.

    The `kind` parameter determines how the wavenumber is returned:

    - `kind="complex"` (default) returns the complex number `√(Ks²)`.
    - `kind="real"` returns the real number `Re(Ks) - Im(Ks)` where `Ks` is the
      complex number `√(Ks²)`. Since `Ks²` is real, the `Im(Ks)` is always zero
      when `Re(Ks)` is non-zero and vice versa, rendering this expression of
      `Ks` unique.
    - `kind="squared"` returns `Ks²`.
    """
    grid, u, pv = _get_grid_vars(["u", "pv"], grid, u_or_state)
    # If u is a 2D field, calculate zonal mean
    if u.ndim != 1:
        u = np.mean(u, axis=_ZONAL)
    # If no PV field is given, calculate zonal-mean PV from zonal-mean u
    if pv is None:
        rv = - grid.ddphi(u * np.cos(grid.phis), order=order) / grid.rsphere / np.cos(grid.phis)
        pv = grid.coriolis(grid.lats) + rv
    # If PV is a 2D field, calculate zonal mean
    elif pv.ndim != 1:
        pv = np.mean(pv, axis=_ZONAL)
    # Mercator projection zonal-mean PV gradient (βM) times cosine of latitude
    ks2 = np.cos(grid.phis)**2 * grid.rsphere * grid.ddphi(pv, order=order)
    # Divide by u, avoid latitudes with small u
    small_u = np.isclose(u, 0., atol=min_u)
    ks2[~small_u] /= u[~small_u]
    # Set to infinity where u is small, let betam determine the sign. Inf is
    # preferrable to NaN as does not require special treatment in WKB waveguide
    # extraction. Avoid 0*inf which is NaN.
    ks2[small_u & (ks2 != 0.)] *= np.inf
    # Return one of the 3 variants (see docstring):
    if kind == "squared":
        return ks2
    ks = np.sqrt(ks2.astype(complex))
    if kind == "complex":
        return ks
    if kind == "real":
        return np.real(ks) - np.imag(ks)
    raise ValueError("kind parameter must be one of 'squared', 'complex', 'real'")


def extract_waveguides(ks_or_state, k, grid=None):
    """Extract waveguide boundaries based on the stationary wavenumber.

    WKB and ray-tracing theory say that Rossby waves are refracted towards
    latitudes of higher stationary wavenumber. A local maximum in the zonal
    profile of stationary wavenumber Ks can therefore trap waves and
    constitutes a waveguide. See e.g. Petoukhov et al. (2013), Wirth (2020).

    - `k` is the wavenumber for which the waveguides are extracted.
    - If the first parameters is not a `barotropic.State` object, `ks` must be
      the complex variant of the stationary wavenumber and `grid` must be given
      (the grid is otherwise taken from the `barotropic.State` object.

    Returns a list of tuples that contain the boundary latitudes of the
    detected waveguides (one tuple per waveguide).
    """
    grid, ks = _get_grid_vars(["stationary_wavenumber"], grid, ks_or_state)
    # Convert stationary wavenumbers to "real" variant, so there is a proper
    # ordering (one could also square them, but then the linear interpolation
    # for the start/end latitudes is not as nice for the non-squared variants)
    ks = np.real(ks) - np.imag(ks)
    k  = np.real(k)  - np.imag(k)
    # There might be more than one waveguide
    waveguides = []
    # Scan from the North pole
    active = ks[0] >= k
    start = grid.lats[0]
    # Scan towards the South pole
    for lat_n, ks_n, ks_s in zip(grid.lats[:-1], ks[:-1], ks[1:]):
        # Waveguide ends if ks goes below k
        if active and ks_s < k:
            end = lat_n + grid.dlat if np.isinf(ks_n) else lat_n - grid.dlat * (ks_n - k) / (ks_s - ks_n)
            waveguides.append((start, end))
            active = False
        # Waveguide starts if ks exceeds k
        elif not active and ks_s > k:
            start = lat_n if np.isinf(ks_s) else lat_n - grid.dlat * (ks_n - k) / (ks_s - ks_n)
            active = True
    # Last waveguide must end at south pole
    if active:
        waveguides.append((start, grid.lats[-1]))
    return waveguides

