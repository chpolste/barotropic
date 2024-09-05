from collections import abc
import datetime as dt
import itertools

import numpy as np

from .state import State, StateList
from .grid import Grid


def as_dataset(states, fields=("u", "v"), concat_dim="time"):
    """Export a collection of states to an xarray dataset.

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

    .. note::
        Export of spectral fields is currently not supported.

    Requrires :py:mod:`xarray`.

    >>> states
    <StateList with 9 states>
    >>> bt.io.as_dataset(states, fields={ "avo": "avo", "ubar": lambda s: s.u.mean(axis=bt.ZONAL) })
    <xarray.Dataset>
    Dimensions:    (latitude: 73, longitude: 144, time: 9)
    Coordinates:
      * latitude   (latitude) float64 ...
      * longitude  (longitude) float64 ...
      * time       (time) float64 ...
    Data variables:
        avo        (time, latitude, longitude) float32 ...
        ubar       (time, latitude) float32 ...

    An empty list of states returns an empty dataset:

    >>> bt.io.as_dataset([])
    <xarray.Dataset>
    Dimensions:  ()
    Data variables:
        *empty*

    .. versionadded:: 3.1

    .. seealso::
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
    "vo": ["vo", "relv", "vorticity", "relative_vorticity"],
    # PV is usually not absolute vorticity and only considered as a last resort
    "avo": ["avo", "absv", "absolute_vorticity", "pv"],
}

def _select_var(ds, names, override=None):
    if override is not None:
        return ds[name]
    for name in names:
        if name in ds:
            return name
    return None

def from_dataset(ds, names=None, grid_kwargs=None, time_fill=0):
    """Turn an xarray dataset into a collection of states.

    Parameters:
        ds (Dataset | DataArray): Input data containing the field(s) from
            which to construct the :py:class:`.State` objects.
        names (dict | None): Name overrides for variable detection. Expects
            a mapping of reference variable names to actual variable names as
            occurring in the datset. See below for the list of reference names.
        grid_kwargs (dict | None): Keyword arguments given to the
            :py:class:`.Grid` constructor.
        time_fill (number | datetime): Fill value for :py:attr:`.State.time` if
            a value cannot be extracted from the dataset.

    Returns:
        :py:class:`.StateList` with all dimensions (other than longitude and
        latitude) flattened.

    The built-in variable detection recognizes the following names:

    .. list-table::
        :header-rows: 1

        * - Init.
          - Ref.
          - Detected Variable Names
          - Description

        * -
          - ``lon``
          - ``longitude``, ``lon``
          - longitude coordinate (mandatory)

        * -
          - ``lat``
          - ``latitude``, ``lat``
          - latitude coordinate (mandatory)

        * -
          - ``time``
          - ``time``
          - time coordinate for :py:attr:`.State.time` (optional)

        * - A
          - ``u``
          - ``u``, ``ugrd``, ``eastward_wind``
          - zonal wind component

        * - A
          - ``v``
          - ``v``, ``vgrd``, ``northward_wind``
          - meridional wind component

        * - B
          - ``vo``
          - ``vo``, ``relv``, ``vorticity``, ``relative_vorticity``
          - relative vorticity

        * - C
          - ``avo``
          - ``avo``, ``absv``, ``absolute_vorticity``, ``pv``
          - absolute vorticity (barotropic PV)

    If the associated variable(s) are detected, the output :py:class:`State`
    objects are constructed with the following order of priority:

    A. From horizontal wind components (``u``, ``v``), using
       :py:meth:`.State.from_wind`.
    B. From relative vorticity (``vo``), using
       :py:meth:`.State.from_vorticity`.
    C. From absolute vorticity (``avo``), using the default :py:class:`.State`
       constructor.

    Only the first available construction is used. To enforce a specific
    initialization path, only supply the variable(s) required for that
    construction. The reference names can be mapped to custom variable names
    via the `names` argument.

    .. note::
        While ``pv`` is used throughout this package to refer to absolute
        vorticity, it may refer to Ertel or QG PV in other datasets. Therefore,
        names explicitly referring to absolute vorticity are given priority in
        the detection.

    If the dimensions of the initialization fields do not end with latitude and
    longitude, transpose the data first.

    >>> ds
    <xarray.Dataset>
    Dimensions:    (time: 6, latitude: 181, longitude: 360)
    Coordinates:
      * longitude  (longitude) float32 0.0 1.0 2.0 3.0 ... 356.0 357.0 358.0 359.0
      * latitude   (latitude) float32 90.0 89.0 88.0 87.0 ... -88.0 -89.0 -90.0
      * time       (time) datetime64[ns] ...
    Data variables:
        u          (time, latitude, longitude) float32 ...
        v          (time, latitude, longitude) float32 ...
    >>> bt.io.from_dataset(ds)
    <StateList with 6 states>

    .. versionadded:: 3.1
    """
    # Processing below assumes input is a Dataset, this converts DataArray (and
    # maybe other) input without having to import xarray for an isinstance check.
    if hasattr(ds, "to_dataset"):
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
    assert np.isclose(lon[-1] - lon[0], 360. - dlon), "lon coordinate irregular at periodic boundary"

    # Verify latitude coordinate
    if var_map["lat"] is None:
        raise ValueError("required coordinate 'lat' not detected, please provide name")
    lat = ds.coords[var_map["lat"]] # also ensures extracted var is a coordinate
    assert lat.ndim == 1, "lat coordinate must be one-dimensional"
    assert lat.size > 0, "lat coordinate is empty"
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
        assert ds[var_map["u"]].dims == ds[var_map["v"]].dims, \
                f"dims of fields '{var_map['u']}' and '{var_map['v']}' not identical"
        data = ds[var_map["u"]]
        # Two fields required to initialize: proceed with DataArray of u for
        # data, and select matching v field during construction of the State
        def as_state(t, da):
            u = da.values
            # Select matching meridional wind field (dims_flatten assigned below)
            v = ds[var_map["v"]].sel({ dim: da.coords[dim] for dim in dims_flatten }).values
            return State.from_wind(grid, t, u, v)
    # States from relative vorticity
    elif var_map["vo"] is not None:
        data = ds[var_map["vo"]] # DataArray, no further selection necessary
        as_state = lambda t, da: State.from_vorticity(grid, t, da.values)
    # States from absolute vorticity
    elif var_map["avo"] is not None:
        data = ds[var_map["avo"]] # DataArray, no further selection necessary
        as_state = lambda t, da: State(grid, t, pv=da.values)
    else:
        raise ValueError(
            "no fields for state construction detected, please specify"
            " names of either 'u' and 'v', 'vo' or 'avo'."
        )

    # Make sure last two dimensions are lat and lon
    assert data.dims[-2] == lat.name, "lat dimension not in position -2, please transpose"
    assert data.dims[-1] == lon.name, "lon dimension not in position -1, please transpose"
    # Only 1-dimensional StateList output supported, so all dimensions other
    # than lon and lat are flattened
    dims_flatten = data.dims[:-2]
    dims_values = [data.coords[dim].values for dim in dims_flatten]

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
                # https://stackoverflow.com/questions/13703720/, but
                # utcfromtimestamp is deprecated, start with UTC and drop the
                # timezone info for now
                timestamp = (time - np.datetime64(0, "s")) / np.timedelta64(1, "s")
                time = dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).replace(tzinfo=None)
        # Create State and add to collection
        states.append(as_state(time, field))

    return StateList(states)

