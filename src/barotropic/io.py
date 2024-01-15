"""Input/output functions (**work in progress**)."""

import numpy as np
import datetime as dt
from .state import State
from .grid import Grid


def to_dataset(states):
    # TODO
    raise NotImplementedError("This feature has not been implemented yet")


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

