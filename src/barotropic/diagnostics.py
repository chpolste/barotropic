import numpy as np
import scipy.signal
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


def fawa(pv_or_state, grid=None, levels=None, interpolate=True, quad="sptrapz"):
    """Finite-Amplitude Wave Activity according to Nakamura and Zhu (2010).

    Computes FAWA as the zonal mean of :py:func:`falwa`.
    """
    grid, pv = _get_grid_vars(["pv"], grid, pv_or_state)
    # FAWA is zonal mean of FALWA
    lwa, yy = falwa(pv, grid, levels, interpolate=False, quad=quad)
    fawa = lwa.mean(axis=_ZONAL)
    # No interpolation: return contour values on equivalent latitudes from
    # zonalization and return the latitudes as well for reference
    if not interpolate:
        return fawa, yy
    # When interpolated, only return FAWA
    return grid.interpolate_meridional(fawa, yy, pole_values=(0., 0.))


def falwa(pv_or_state, grid=None, levels=None, interpolate=True, quad="sptrapz"):
    """Finite-Amplitude Local Wave Activity according to Huang and Nakamura (2016).

    Parameters:
        pv_or_state (:py:class:`.State` | array): Input state or 2D PV field.
        grid (None | :py:class:`.Grid`): Grid information only required if
            `pv_or_state` is not a :py:class:`.State`.
        levels (None | array | int): Parameter of :py:meth:`.Grid.zonalize`.
        interpolate (bool): Interpolate output onto the regular latitudes of
            the grid. If `False`, FALWA is returned on the equivalent latitudes
            obtained from the zonalization procedure. These may be irregular
            and unordered.
        quad (str): Quadrature rule used in the surface integrals of the
            computation. See :py:meth:`.Grid.quad`.

    Returns:
        If **interpolate** is `True`, return FALWA interpolated onto the
        regular latitudes as an array. Otherwise, return a tuple of FALWA and
        the associated equivalent latitudes (both arrays).

    To improve numerical stability if a boxcounting quadrature is selected, the
    zonalized PV profile is always interpolated to the dual latitude grid.
    Therefore, **levels** does not affect the output latitudes, even if
    **interpolate** is set to `False`.
    """
    grid, pv = _get_grid_vars(["pv"], grid, pv_or_state)
    # Compute zonalized background state of PV
    qq, yy = grid.zonalize(pv, levels=levels, interpolate=False, quad=quad)
    # Interpolate between gridpoints in case of the the boxcounting quadrature
    # to stabilize the computation (boxcounting does not interpolate the domain
    # of integration so being aligned with the gridbox boundaries helps)
    if quad == "boxcount":
        interp = grid.interp1d_meridional(qq, yy, pole_values=(np.max(pv), np.min(pv)))
        yy = 0.5 * (grid.lat[:-1] + grid.lat[1:])
        qq = interp(yy)
    # Use formulation that integrates PV over areas north of PV
    # contour/equivalent latitude and then computes difference
    q_int = np.frompyfunc(lambda q, y: grid.quad_meridional(pv-q, pv-q, method=quad), 2, 1)
    y_int = np.frompyfunc(lambda q, y: grid.quad_meridional(pv-q, grid.lat2-y, method=quad), 2, 1)
    # Stack meridional integrals (normalized by cosine of latitude) along zonal
    # direction to obtain full field again
    falwa = np.stack((q_int(qq, yy) - y_int(qq, yy)) / np.cos(np.deg2rad(yy)))
    # No interpolation: return contour values on equivalent latitudes from
    # zonalization and return the latitudes as well for reference
    if not interpolate:
        return falwa, yy
    # When interpolated, only return FALWA
    return grid.interpolate_meridional(falwa, yy, pole_values=(0., 0.))


def dominant_wavenumber_fourier(field, grid, smooth=("boxcar", 10), wavenumber_range=(1, 20)):
    """Dominant zonal wavenumber based on the Fourier power spectrum.

    Parameters:
        field (array): 2D input field.
        grid (:py:class:`.Grid`): Grid associated with input field.
        smooth (None | tuple): If not `None`, the output wavenumber profile is
            filtered with :py:meth:`.Grid.filter_meridional`, whose `window`
            and `width` arguments are set by `smooth`.
        wavenumber_range ((number, number)): Restrict the interval in which the
            dominant wavenumber is determined.

    Returns:
        Returns the dominant zonal wavenumber as a function of latitude.
    """
    assert len(wavenumber_range) == 2
    k_min, k_max = wavenumber_range
    # Compute Fourier power spectrum and determine wavenumber with maximum
    # power in the specified wavenumber range
    power = np.abs(np.fft.rfft(field, axis=_ZONAL)) ** 2
    k_dom = np.argmax(power[:,k_min:k_max+1], axis=_ZONAL).astype(float)
    # Apply smoothing if desired
    return k_dom if smooth is None else grid.filter_meridional(k_dom, *smooth)


def dominant_wavenumber_wavelet(field, grid, smooth=("hann", 40, 10)):
    """Dominant zonal wavenumber based on Wavelet Analysis.

    Parameters:
        field (array): 2D input field.
        grid (:py:class:`.Grid`): Grid associated with input field.
        smooth (None | tuple): The smoothing applied to the dominant wavenumber
            field. Set to `None` if no smoothing is desired. Otherwise provide
            a 3-tuple `(window, width_lon, width_lat)` used as input to
            :py:meth:`.Grid.filter_meridional` and :py:meth:`.Grid.filter_zonal`.

    Returns:
        Gridded dominant zonal wavenumber.

    Implements the procedure of Ghinassi et al. (2018) that determines
    a local dominant wavenumber at every gridpoint of the input field.

    Requires :py:mod:`pywt` (version >= 1.1.0).
    """
    import pywt
    # Truncate zonal fourier spectrum of meridional wind after wavenumber 20
    x = _restrict_fourier_zonal(field, 0, 20)
    # Triplicate field to avoid boundary issues (pywt.cwt does not support
    # periodic mode as of version 1.1)
    x = np.hstack([x, x, x])
    # Use the complex Morlet wavelet
    morlet = pywt.ContinuousWavelet("cmor1.0-1.0")
    # ... TODO: what's going on here?
    scales = 3 * 2**np.linspace(0, 6, 120)
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
    k_dom = wavenum[np.argmax(power, axis=0)]
    # Apply smoothing if desired
    if smooth is not None:
        window, wlon, wlat = smooth
        k_dom = grid.filter_meridional(k_dom, window, wlat)
        k_dom = grid.filter_zonal(k_dom, window, wlon)
    return k_dom


def filter_by_wavenumber(field, wavenumber):
    """Zonal Hann smoothing based on a space-dependent wavenumber field.

    Parameters:
        field (array): 2D input field.
        wavenumber (array): `nlat`-sized 1D array or `field`-shaped 2D
            array of wavenumbers.

    Returns:
        Filtered field.

    Apply Hann smoothing in zonal direction with the Hann window width governed
    at every gridpoint by the wavenumber given for the same location. The
    wavelength corresponding to each wavenumber determines the full width at
    half maximum of the Hann window.

    Used by Ghinassi et al. (2018) to filter the FALWA field, discounting phase
    information in order to diagnose wave packets as a whole.
    """
    nlon = field.shape[_ZONAL]
    # If the wavenumber field is provided as a function of latitude only,
    # extend it to a lon-lat-dependent 2D field
    if wavenumber.ndim == 1:
        assert wavenumber.shape[0] == field.shape[0], "1-dimensional input for wavenumber must be nlat-sized"
        wavenumber = np.repeat(wavenumber, nlon).reshape(field.shape)
    # Limit wavenumbers to values above 1. The Hann window at this point goes
    # twice around the entire globe already. For wavenumbers larger than nlon/2
    # the Hann window width is smaller than 4 gridpoints making it impossible
    # to adequately represent the window on the grid.
    wavenumber = np.clip(wavenumber, 1., nlon / 2)
    # scipy.signal.window.hann takes the full width of the window in gridpoints
    # as its argument. nlon / wavenumber is half the width of the window.  To
    # speed up computation, only consider odd window widths which can be
    # properly centered on a gridpoint.
    hann_width = ((nlon // wavenumber) * 2 + 1).astype(int)
    used_widths = set(np.unique(hann_width))
    # Precompute the smoothed fields for all required odd gridpoint-widths.
    # While this produces a lot of unused data, it allows to use more
    # numpy-accelerated operations which should outperform any pure-Python and
    # loop-based implementation. Widths and indices are mapped by 2n+1 -> n.
    convs = []
    for width in range(1, max(used_widths)+1, 2):
        # Skip smoothing for all widths that aren't used
        if width not in used_widths:
            convs.append(np.zeros_like(field))
            continue
        # Obtain the normalized Hann window for the current width
        hann = scipy.signal.hann(width)
        hann = hann / np.sum(hann)
        # Apply window along the zonal axis
        convs.append(scipy.signal.convolve2d(field, hann[None,:], mode="same", boundary="wrap"))
    # Stack into 3-dimensional array
    convs = np.stack(convs)
    # Extract the filtered value from the smoothed fields according the the
    # width-index mapping
    idx = (hann_width - 1) // 2
    ii, jj = np.indices(field.shape)
    return convs[idx,ii,jj]


def envelope_hilbert(field, wavenumber_range=(2, 10)):
    """Compute the envelope of wave packets with the Hilbert transform.

    Parameters:
        field (array): 2D input field.
        wavenumber_range ((number, number)): Restrict the interval in which the
            dominant wavenumber is determined.

    Returns:
        Filtered output field.

    Applied to the merdional wind for RWP detection by Zimin et al. (2003).
    """
    assert len(wavenumber_range) == 2
    k_min, k_max = wavenumber_range
    x = _restrict_fourier_zonal(field, k_min, k_max)
    return np.abs(scipy.signal.hilbert(x, axis=_ZONAL))


def stationary_wavenumber(u_or_state, grid=None, order=4, min_u=0.001, kind="complex"):
    """Non-dimensionalised stationary (zonal) wavenumber.

    Parameters:
        u_or_state (:py:class:`.State` | array): Input state or 2D field of
            zonal wind component.
        grid (None | :py:class:`.Grid`): Grid information only required if
            `u_or_state` is not a :py:class:`.State`.
        order (2 | 4): Order of the derivative used in the calculation.
            Parameter of :py:meth:`.Grid.derivative_meridional`.
        min_u (number): Threshold for the absolute value of the zonal wind
            below which it is considered to be zero.
        kind ("complex" | "real" | "squared"): ``Ks²`` can be negative.
            Determine how ``Ks`` is returned:

            - `"complex"`: return complex number `√(Ks²)`.
            - `"real"`: return real number ``Re(Ks) - Im(Ks)`` where ``Ks`` is the
              complex number ``√(Ks²)``. Since ``Ks²`` is real, ``Im(Ks)`` is
              always zero when ``Re(Ks)`` is non-zero and vice versa, rendering
              this expression of ``Ks`` unique.
            - `"squared"`: return ``Ks²``.

    Returns:
        Meridional profile of stationary wavenumber.

    Stationary zonal wavenumber::

        Ks² = r²·βM/uM,

    where ``r`` is the radius of the sphere, ``βM`` is the Mercator projection
    gradient of zonal-mean PV and ``uM`` is the Mercator projection zonal-mean
    wind. See e.g. Hoskins and Karoly (1981), Wirth (2020).
    """
    grid, u, pv = _get_grid_vars(["u", "pv"], grid, u_or_state)
    # If u is a 2D field, calculate zonal mean
    if u.ndim != 1:
        u = np.mean(u, axis=_ZONAL)
    # If no PV field is given, calculate zonal-mean PV from zonal-mean u
    if pv is None:
        rv = - grid.derivative_meridional(u * np.cos(grid.phi), order=order) / grid.rsphere / np.cos(grid.phi)
        pv = grid.fcor + rv
    # If PV is a 2D field, calculate zonal mean
    elif pv.ndim != 1:
        pv = np.mean(pv, axis=_ZONAL)
    # Mercator projection zonal-mean PV gradient (βM) times cosine of latitude
    ks2 = np.cos(grid.phi)**2 * grid.rsphere * grid.derivative_meridional(pv, order=order)
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


def extract_ks_waveguides(ks_or_state, k, grid=None):
    """Extract waveguide boundaries based on the stationary wavenumber.

    Parameters:
        ks_or_state (:py:class:`.State` | array): Input state or stationary
            wavenumber profile (complex variant).
        k (int): Wavenumber for which waveguides are extracted.
        grid (None | :py:class:`.Grid`): Grid information only required
            if `ks_or_state` is not a :py:class:`.State`.

    Returns:
        List of tuples that contain the boundary latitudes of the detected
        waveguides (one tuple per waveguide) in degrees.

    WKB and ray-tracing theory say that Rossby waves are refracted towards
    latitudes of higher stationary wavenumber. A local maximum in the
    meridional profile of stationary zonal wavenumber Ks can therefore trap
    waves and constitutes a waveguide. See e.g. Petoukhov et al. (2013), Wirth
    (2020).
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
    start = grid.lat[0]
    # Scan towards the South pole
    for lat_n, ks_n, ks_s in zip(grid.lat[:-1], ks[:-1], ks[1:]):
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
        waveguides.append((start, grid.lat[-1]))
    return waveguides

