import numpy as np


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

