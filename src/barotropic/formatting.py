from numbers import Number
import numpy as np


_sub = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")
_sup = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

def sub(num):
    return str(num).translate(_sub)

def sup(num):
    return str(num).translate(_sup)

def format_time(time):
    # Adjust unit depending on size of number
    if isinstance(time, Number):
        unit = "s"
        if time > 120:
            time, unit = time / 60, "min"
        if time > 120:
            time, unit = time / 60, "h"
        if time > 240:
            time, unit = time / 24, "d"
        return "t = {:.1f} {}".format(time, unit)
    # If time is not numeric assume it is datetime-like
    else:
        return time.isoformat()




def grid_repr(grid):
    return "\n".join([
        "<barotropic.Grid> with",
        "    (nlat, nlon) = {}".format(grid.shape),
        "    resolution   = {}°".format(grid.dlon),
        "    rsphere      = {} m".format(grid.rsphere),
        "    omega        = {} s⁻¹".format(grid.omega),
        "    ntrunc       = {}".format(grid._ntrunc),
        "    legfunc      = {}".format(grid._spharm.legfunc)
    ])

def grid_region_repr(region):
    # Draw a crude "map" with a box for the selection
    mask_lat = np.histogram(region.lat, np.linspace(-90, 90, 10))[0] > 0
    mask_lon = np.histogram(region.lon, np.linspace(0, 360., 37))[0] > 0
    mask = mask_lat[::-1,None] & mask_lon[None,:]
    out = ["    " + "".join(row) for row in np.where(mask, "X", "·")]
    out[0] += " 90°N"
    out[4] += " EQ"
    out[8] += " 90°S"
    out.append("    0°E              180°E")
    return "\n".join([
        "Region {:6.2f} to {:6.2f} °N (nlat = {})".format(*region.lat[[-1,0]], region.shape[0]),
        "       {:6.2f} to {:6.2f} °E (nlon = {})".format(*region.lon[[0,-1]], region.shape[1]),
        "", *out, "",
        "of {}".format(repr(region._grid))
    ])


def state_repr(state):
    return "<barotropic.State {}>".format(format_time(state.time))


def barotropic_model_repr(model):
    exp = sup(2 * model.diffusion_order)
    return "\n".join([
        "<barotropic.BarotropicModel> integrating",
        "    ∂q/∂t = -∇(u·q) - κ∇{}q + RHS ".format(exp),
        "    where κ   = {:.5e} m{}s⁻¹".format(model.diffusion_coeff, exp),
        "    and   RHS = {}".format(repr(model.rhs)), # TODO
    ])

