"""Input/output functions (**work in progress**)."""

from collections import abc
import datetime as dt

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


def from_dataset(dataset, names=None, grid_kwargs=None):
    """Load states from u and v components in an xarray Dataset
    
    If automatic detection of grid and winds based on the coordinate
    attributes of the dataset fails, the association of `lat`, `lon`,
    `time`, `u` and `v` can be specified explicitly with a dict given to
    names. A `Grid` object shared by the returned states is created based
    on the dataset coordinates. Additional arguments for the `Grid`
    instanciation can be specified with the `grid_kwargs` argument.
    """
    # TODO also accept fields of absolute/relative vorticity
    # TODO write proper error messages
    # Dataset should have 3 dimensions: time, lat, lon
    assert len(dataset.dims) == 3
    # Initialize mapping of coordinate names
    var_map = { "lat": None, "lon": None, "time": None, "u": None, "v": None }
    # Iterate through all coordinates of the dataset and try to detect
    # usable fields based on metadata
    for name, var in dataset.variables.items():
        lname = var.attrs["long_name"].lower() if "long_name" in var.attrs else None
        units = var.attrs["units"].lower() if "units" in var.attrs else None
        if lname == "time":
            var_map["time"] = name
        # Verify that latitude and longitude is given in degrees
        elif lname == "longitude":
            assert "degree" in units and "east" in units
            var_map["lon"] = name
        elif lname == "latitude":
            assert "degree" in units and "north" in units
            var_map["lat"] = name
        # Verify that wind components are given in m/s
        elif lname is not None and "wind" in lname:
            assert "m s**-1" in units or "m s^-1" in units or "m/s" in units
            if "u component" in lname or "zonal" in lname:
                var_map["u"] = name
            if "v component" in lname or "meridional" in lname:
                var_map["v"] = name
    # Override mapping with given name map argument
    if names is not None:
        var_map.update(names)
    # Verify that every required variable has been found
    assert all(var is not None for var in var_map.values())
    # Extract latitude and longitude coordinates and verify basic
    # properties (non-emptyness, shape)
    lon = dataset[var_map["lon"]].values
    lat = dataset[var_map["lat"]].values
    assert lat.size > 0
    assert lon.size > 0
    assert lat.shape[0] % 2 == 1
    assert lat.shape[0] - 1 == lon.shape[-1] // 2
    # Verify that latitudes go from north pole to south pole, longitudes
    # from west to east and grid is regular
    dlats = np.diff(lat, axis= 0).flatten()
    dlons = np.diff(lon, axis=-1).flatten()
    dlat = dlats[0]
    dlon = dlons[0]
    assert dlat < 0
    assert dlon > 0
    assert np.isclose(-dlat, dlon)
    assert np.isclose(dlat, dlats).all()
    assert np.isclose(dlon, dlons).all()
    # Instanciate shared Grid that matches dataset coordinates
    grid_kwargs = {} if grid_kwargs is None else grid_kwargs
    grid = Grid(resolution=dlon, **grid_kwargs)
    # For every time in the dataset, extract the wind fields and create
    # a State from them
    states = []
    for time in dataset[var_map["time"]].values:
        data = dataset.sel({ var_map["time"]: time }, drop=True)
        # Verify that coordinates are in the right order
        assert tuple(data.coords) == (var_map["lon"], var_map["lat"])
        # Convert numpy datetime type into a regular datetime instance
        # https://stackoverflow.com/questions/13703720/
        if isinstance(time, np.datetime64):
            time = dt.datetime.utcfromtimestamp((time - np.datetime64(0, "s")) / np.timedelta64(1, "s"))
        states.append(
            State.from_wind(grid, time, data[var_map["u"]].values, data[var_map["v"]].values)
        )
    return states

