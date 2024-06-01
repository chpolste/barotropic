from collections import abc
import datetime as dt
import itertools
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

def _select_var(ds, names, override=None):
    if override is not None:
        return ds[name]
    for name in names:
        if name in ds:
            return name
    return None

def from_dataset(ds, names=None, grid_kwargs=None, time_fill=0):
    """Load states from u and v components in an xarray Dataset

    Parameters:
        dataset (:py:class:`xarray.Dataset`): TODO
        names (dict | None): TODO
        grid_kwargs (dict | None): TODO
        time_fill (number | datetime): TODO

    Returns:
        :py:class:`.StateList`
    
    If automatic detection of grid and winds based on the coordinate
    attributes of the dataset fails, the association of `lat`, `lon`,
    `time`, `u` and `v` can be specified explicitly with a dict given to
    names. A `Grid` object shared by the returned states is created based
    on the dataset coordinates. Additional arguments for the `Grid`
    instanciation can be specified with the `grid_kwargs` argument.
    """
    import xarray as xr
    # If a path is given, read first
    if isinstance(ds, str):
        ds = xr.open_dataset(ds)
    # Processing below assumes input is a Dataset
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    ds = ds.squeeze()

    # Extract coordinates, respect user override if provided
    if names is None:
        names = dict()
    var_map = {
        ref: names.get(ref, _select_var(ds, ns)) # try override first, else pick from ds
        for ref, ns in _VAR_NAMES.items()
    }

    # Verify longitude coordinate
    if var_map["lon"] is None:
        raise ValueError("required coordinate 'lon' not detected, please provide name")
    lon = ds.coords[var_map["lon"]]
    assert lon.ndim == 1, "lon coordinate must be one-dimensional"
    assert lon.size > 0, "lon coordinate is empty"
    dlons = np.diff(lon.values)
    dlon = dlons[0]
    assert np.isclose(dlon, dlons).all(), "lon spacing is not regular"
    assert dlon > 0, "lon coordinate must be increasing (W to E)"
    # TODO test peridodicity

    # Verify latitude coordinate
    if var_map["lat"] is None:
        raise ValueError("required coordinate 'lat' not detected, please provide name")
    lat = ds.coords[var_map["lat"]] # also ensures extracted var is a coordinate
    assert lat.ndim == 1, "lat coordinate must be one-dimensional"
    assert lat.size > 0, "lat coordinate is empty"
    assert lat.size % 2 == 1, "lat coordinate must have odd number of grid points"
    assert lat.size - 1 == lon.size // 2, f"grid of shape [{lat.size}, {lon.size}] is not regular"
    dlats = np.diff(lat.values)
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
        if ds[var_map["u"]].dims != ds[var_map["v"]].dims:
            raise ValueError(f"dims of fields '{var_map['u']}' and '{var_map['v']}' not identical")
        data = ds[var_map["u"]]
        # Two fields required to initialize: proceed with DataArray of u for
        # data, and select matching v field during construction of the State
        def as_state(t, da):
            u = da.values
            # Select matching meridional wind field (dims_flatten assigned below)
            v = ds[var_map["v"]].sel({ dim: da.coords[dim] for dim in dims_flatten }).values
            return State.from_wind(grid, t, u, v)
    # States from relative vorticity
    elif var_map["rv"] is not None:
        data = ds[var_map["vo"]] # DataArray, no further selection necessary
        as_state = lambda t, da: State.from_vorticity(grid, t, da.values)
    # States from absolute vorticity
    elif var_map["pv"] is not None:
        data = ds[var_map["pv"]] # DataArray, no further selection necessary
        as_state = lambda t, da: State(grid, t, pv=da.values)
    else:
        raise ValueError(
            "no fields for state construction detected, please specify"
            " names of either 'u' and 'v', 'rv' or 'pv'."
        )

    # Make sure last two dimensions are lat and lon
    assert data.dims[-2] == lat.name, "lat dimension not in position -2, please transpose"
    assert data.dims[-1] == lon.name, "lon dimension not in position -1, please transpose"
    # Only 1-dimensional StateList output supported, so all dimensions other
    # than lon and lat are flattened
    dims_flatten = data.dims[:-2]
    dims_values = [data.coords[dim].values for dim in dims_flatten]
    if len(dims_flatten) > 0:
        _ = ", ".join(f"'{dim}'" for dim in dims_flatten)
        warnings.warn(f"dimension(s) {_} flattened in output StateList")

    states = []
    # Because itertools.product returns an empty tuple when called without any
    # arguments (here: when dims_values is empty), this also covers the case of
    # single field input with no flattened dimensions
    for selection in itertools.product(*dims_values):
        # Extract the current lon-lat field
        field = data.sel(dict(zip(dims_flatten, selection)))
        # Try to use time value from coordinates if possible, otherwise fall
        # back to fill value
        time = time_fill
        if var_map["time"] in field.coords:
            time = field.coords[var_map["time"]].values
            # Convert numpy datetime type into a regular datetime instance
            if isinstance(time, np.datetime64):
                # https://stackoverflow.com/questions/13703720/
                time = dt.datetime.utcfromtimestamp(
                    (time - np.datetime64(0, "s")) / np.timedelta64(1, "s")
                )
        # Create State and add to collection
        states.append(as_state(time, field))

    return StateList(states)

