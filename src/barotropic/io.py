from collections import abc
import datetime as dt
import warnings

import numpy as np

from .state import State, StateList
from .grid import Grid


def as_dataset(states, fields=("pv", "u", "v"), concat_dim="time"):
    """Create an :py:class:`xarray.Dataset` from a collection of states.

    Parameters:
        states (iterable of :py:class:`.State`): states to include in the
            created dataset.
        fields (str | iterable | dict): fields extracted and placed in the
            dataset. If a name or list of names is specified, the
            correspondingly named attributes of the provided states are
            extracted. Alternatively, a mapping from names to attributes or
            callables can be specified. Fields are then extracted by accessing
            the attribute or calling the provided callable on all states and
            placed in the dataset under the names specified in the mapping.
        concat_dim (str): The dimension used to index the fields in the output
            dataset. By default, the time dimension is used as an index.
            Alternatively, a different dimension can be specified. If all
            states have a corresponding attribute, the attribute's values are
            used to construct a coordinate. Otherwise the coordinate is
            generated with :py:func:`numpy.arange`.

    Returns:
        :py:class:`xarray.Dataset`

    >>> states
    <StateList with 9 states>
    >>> bt.io.as_dataset(states, fields={ "pv": "pv", "ubar": lambda s: s.u.mean(axis=bt.ZONAL) })
    <xarray.Dataset>
    Dimensions:    (latitude: 73, longitude: 144, time: 9)
    Coordinates:
      * latitude   (latitude) float64 ...
      * longitude  (longitude) float64 ...
      * time       (time) float64 ...
    Data variables:
        pv         (time, latitude, longitude) float32 ...
        ubar       (time, latitude) float32 ...

    An empty list of states returns an empty dataset:

    >>> bt.io.as_dataset([])
    <xarray.Dataset>
    Dimensions:  ()
    Data variables:
        *empty*

    See also:
        :py:meth:`.StateList.as_dataset`.

    """
    import xarray as xr
    from . import __version__
    # Empty input
    if not states:
        return xr.Dataset()
    # Compatibility for single-state input
    if isinstance(states, State):
        states = [states]
    # Convert into state collection for accessors and basic integrity checks
    states = StateList(states)
    # Extract basic coordinates
    coords = dict()
    coords["latitude"] = xr.DataArray(states.grid.lat, dims=["latitude"], attrs={
        "units": "degrees_north"
    })
    coords["longitude"] = xr.DataArray(states.grid.lon, dims=["longitude"], attrs={
        "units": "degrees_east"
    })
    coords["time"] = xr.DataArray(states.time, dims=["time"])
    # Numeric time coordinate in seconds as in BarotropicModel integrator
    if coords["time"].dtype.kind in "fiu":
        coords["time"].attrs["units"] = "s"
    # Attach non-time index coordinate if specified
    if concat_dim != "time":
        try:
            values = getattr(states, concat_dim)
        except AttributeError:
            values = np.arange(len(states))
        coords[concat_dim] = values
    # Convert single-field and list-like field input
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, abc.Mapping):
        fields = { field: field for field in fields }
    # Collect all fields
    data_vars = dict()
    for name, getter in fields.items():
        # Extract fields from state collection
        values = states.map(getter) if callable(getter) else getattr(states, getter)
        # Attach coordinates according to dimensions of values
        if values.ndim == 3:
            dims = [concat_dim, "latitude", "longitude"]
        elif values.ndim == 2 and values.shape[1] == states.grid.nlat:
            dims = [concat_dim, "latitude"]
        elif values.ndim == 2 and values.shape[1] == states.grid.nlon:
            dims = [concat_dim, "longitude"]
        elif values.ndim == 1:
            dims = [concat_dim]
        else:
            raise NotImplementedError(
                f"field {name}: extracting fields with dimensions other "
                "than latitude or longitude is not (yet) supported"
            )
        data_vars[name] = xr.DataArray(values, dims=dims)
    # Assemble and return
    return xr.Dataset(data_vars, coords, attrs={
        "package": f"barotropic {__version__}"
    })


_VAR_NAMES = {
    "lon": ["longitude", "lon"],
    "lat": ["latitude", "lat"],
    "time": ["time"],
    "u": ["u", "ugrd", "eastward_wind"],
    "v": ["v", "vgrd", "northward_wind"],
    "rv": ["vo", "relv", "vorticity", "relative_vorticity"],
    # PV is usually not absolute vorticity and only considered as a last resort
    "pv": ["avo", "absv", "absolute_vorticity", "pv"],
}

def _extract_var(ds, names, override=None):
    if override is not None:
        return ds[name]
    for name in names:
        if name in ds:
            return ds[name]
    return None

def from_dataset(ds, names=None, grid_kwargs=None):
    """Load states from u and v components in an xarray Dataset

    Parameters:
        dataset (:py:class:`xarray.Dataset`): TODO
        names (dict | None): TODO
        grid_kwargs (dict | None): TODO

    Returns:
        :py:class:`.StateList`
    
    If automatic detection of grid and winds based on the coordinate
    attributes of the dataset fails, the association of `lat`, `lon`,
    `time`, `u` and `v` can be specified explicitly with a dict given to
    names. A `Grid` object shared by the returned states is created based
    on the dataset coordinates. Additional arguments for the `Grid`
    instanciation can be specified with the `grid_kwargs` argument.
    """
    # TODO allow path for ds, open here with xr.open_dataset

    # Extract coordinates, respect user override if provided
    if names is None:
        names = dict()
    var_map = {
        k: _extract_var(ds, ns, override=names.get(k, None))
        for k, ns in _VAR_NAMES.items()
    }

    # Verify longitude coordinate
    if var_map["lon"] is None:
        raise ValueError("required coordinate 'lon' not detected, please provide name")
    lon = var_map["lon"]
    if "units" in lon.attrs:
        units = lon.attrs["units"]
        if not "degree" in units and "east" in units:
            warnings.warn(f"expected 'degrees_east' as units of lon, but found '{units}'")
    assert lon.ndim == 1, "lon coordinate must be one-dimensional"
    assert lon.size > 0, "lon coordinate is empty"
    dlons = np.diff(lon)
    dlon = dlons[0]
    assert np.isclose(dlon, dlons).all(), "lon spacing is not regular"
    assert dlon > 0, "lon coordinate must be increasing (W to E)"
    # TODO test peridodicity

    # Verify latitude coordinate
    if var_map["lat"] is None:
        raise ValueError("required coordinate 'lat' not detected, please provide name")
    lat = var_map["lat"] # also ensures extracted var is a coordinate
    if "units" in lat.attrs:
        units = var_map["lat"].attrs["units"]
        if not "degree" in units and "north" in units:
            warnings.warn(f"expected 'degrees_north' as units of lat, but found '{units}'")
    assert lat.ndim == 1, "lat coordinate must be one-dimensional"
    assert lat.size > 0, "lat coordinate is empty"
    assert lat.size % 2 == 1, "lat coordinate must have odd number of grid points"
    assert lat.size - 1 == lon.size // 2, f"grid of shape [{lat.size}, {lon.size}] is not regular"
    dlats = np.diff(lat)
    dlat = dlats[0]
    assert np.isclose(dlat, dlats).all(), "lat spacing is not regular"
    assert dlat < 0, "lat coordinate must be decreasing (N to S)"
    assert np.isclose(dlat, -dlon), "grid is not regular: lat spacing {dlat}, lon spacing {dlon}"

    # Instanciate Grid that matches dataset coordinates
    grid_kwargs = {} if grid_kwargs is None else grid_kwargs
    grid = Grid(resolution=dlon, **grid_kwargs)

    # Consider multiple options to construct states:
    # States from horizontal wind components
    if var_map["u"] is not None and var_map["v"] is not None:
        # TODO verify u (NaN, plausibility, units, ...)
        # TODO verify v (NaN, plausibility, units, ...)
        # Verify that wind components are given in m/s
        #lname = var.attrs["long_name"].lower() if "long_name" in var.attrs else None
        #units = var.attrs["units"].lower() if "units" in var.attrs else None
        #if lname is not None and "wind" in lname:
        #    assert "m s**-1" in units or "m s^-1" in units or "m/s" in units
        #    if "u component" in lname or "zonal" in lname:
        #        var_map["u"] = name
        #    if "v component" in lname or "meridional" in lname:
        #        var_map["v"] = name
        as_state = lambda t, d: State.from_wind(grid, t, d[var_map["u"]], d[var_map["v"]])
    # States from relative vorticity
    elif var_map["rv"] is not None:
        # TODO verify rv (NaN, plausibility, units, ...)
        as_state = lambda t, d: State.from_vorticity(grid, t, d[var_map["vo"]])
    # States from absolute vorticity
    elif var_map["pv"] is not None:
        # TODO verify pv (NaN, plausibility, units, ...)
        as_state = lambda t, d: State(grid, t, pv=d[var_map["vo"]])
    else:
        raise ValueError(
            "no fields for state construction detected, please specify"
            " names of either 'u' and 'v', 'rv' or 'pv'."
        )

    # Dataset should have 3 dimensions: time, lat, lon
    assert len(ds.dims) == 3 # TODO not necessary, but warn of flattening
    # TODO rewrite with more construction options
    states = []
    for time in var_map["time"].values:
        data = ds.sel({ var_map["time"].name: time }, drop=True)
        # Verify that coordinates are in the right order
        assert tuple(data.coords) == (var_map["lon"].name, var_map["lat"].name)
        # Convert numpy datetime type into a regular datetime instance
        # https://stackoverflow.com/questions/13703720/
        if isinstance(time, np.datetime64):
            time = dt.datetime.utcfromtimestamp((time - np.datetime64(0, "s")) / np.timedelta64(1, "s"))
        states.append(
            State.from_wind(grid, time, data[var_map["u"].name].values, data[var_map["v"].name].values)
        )
    return StateList(states)

