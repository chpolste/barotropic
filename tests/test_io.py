import importlib.util
import datetime as dt

import pytest

import numpy as np
import barotropic as bt

from . import ATOL_WIND, ATOL_VO, ATOL_PV


@pytest.mark.skipif(importlib.util.find_spec("xarray") is None, reason="unable to import xarray")
class TestAsDataset:

    def test_empty(self):
        assert not bt.io.as_dataset([])
        assert not bt.io.as_dataset(bt.StateList([]))

    def test_single(self):
        grid = bt.Grid()
        ds = bt.io.as_dataset(bt.init.motionless(grid))
        assert ds
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        assert "time" in ds.coords
        assert ds.coords["time"].size == 1

    def test_fields(self):
        grid = bt.Grid()
        ds = bt.io.as_dataset(bt.init.motionless(grid), fields={
            "uall": "u",
            "uzon": lambda s: s.u.mean(axis=bt.ZONAL),
            "umer": lambda s: s.u.mean(axis=bt.MERIDIONAL),
            "uavg": lambda s: s.u.mean(),
        })
        assert ds
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        assert "time" in ds.coords
        assert "v" not in ds
        assert ds["uall"].dims == ("time", "latitude", "longitude")
        assert ds["uzon"].dims == ("time", "latitude")
        assert ds["umer"].dims == ("time", "longitude")
        assert ds["uavg"].dims == ("time",)

    def test_concat_dim(self):
        grid = bt.Grid()
        s1 = bt.init.motionless(grid, time=0)
        s1.foobar = "foo"
        s2 = bt.init.motionless(grid, time=1) 
        s2.foobar = "bar"
        ds = bt.io.as_dataset([s1, s2], concat_dim="foobar")
        assert ds
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        assert "time" in ds.coords
        assert "foobar" in ds.coords
        assert ds.coords["time"].size == 2
        assert ds.coords["foobar"].size == 2
        assert ds.coords["foobar"][0] == "foo"
        assert ds.coords["foobar"][1] == "bar"


@pytest.mark.skipif(importlib.util.find_spec("xarray") is None, reason="unable to import xarray")
class TestFromDataset:

    time = [dt.datetime(2024, 1, 1, 0), dt.datetime(2024, 1, 1, 12), dt.datetime(2024, 1, 2, 0)]

    @pytest.fixture
    def ds(self):
        import xarray as xr
        grid = bt.Grid(10.)
        u = np.zeros((3, 4) + grid.shape)
        v = np.zeros((3, 4) + grid.shape)
        vo = np.zeros((3, 4) + grid.shape)
        avo = grid.fcor2 + vo
        return xr.Dataset(
            data_vars={
                "u": (["time", "level", "latitude", "longitude"], u),
                "v": (["time", "level", "latitude", "longitude"], v),
                "vo": (["time", "level", "latitude", "longitude"], vo),
                "avo": (["time", "level", "latitude", "longitude"], avo),
            },
            coords={
                "time": ("time", self.time),
                "level": ("level", np.arange(4)),
                "latitude": ("latitude", grid.lat),
                "longitude": ("longitude", grid.lon),
            }
        )

    def test_from_wind(self, ds):
        states = bt.io.from_dataset(ds[["u", "v"]])
        assert len(states) == 3*4
        assert np.allclose(states.u, 0., atol=ATOL_WIND)
        assert np.allclose(states.v, 0., atol=ATOL_WIND)
        assert np.isclose(states.grid.dlon, 10.)

    def test_from_vorticity(self, ds):
        states = bt.io.from_dataset(ds[["vo"]])
        assert len(states) == 3*4
        assert np.allclose(states.u, 0., atol=ATOL_VO)
        assert np.allclose(states.v, 0., atol=ATOL_VO)
        assert np.isclose(states.grid.dlon, 10.)

    def test_from_pv(self, ds):
        states = bt.io.from_dataset(ds[["avo"]])
        assert len(states) == 3*4
        assert np.allclose(states.u, 0., atol=ATOL_PV)
        assert np.allclose(states.v, 0., atol=ATOL_PV)
        assert np.isclose(states.grid.dlon, 10.)

    def test_single(self, ds):
        states = bt.io.from_dataset(ds.isel({ "time": 0, "level": 0 }))
        assert len(states) == 1

    def test_dataarray(self, ds):
        states = bt.io.from_dataset(ds["vo"])
        assert len(states) == 3*4

    def test_time_detection(self, ds):
        states = bt.io.from_dataset(ds.isel({ "level": 0 }))
        assert len(states) == 3
        assert states[0].time == self.time[0]
        assert states[1].time == self.time[1]
        assert states[2].time == self.time[2]

    def test_time_fill(self, ds):
        states = bt.io.from_dataset(ds.rename({ "time": "foo" }), time_fill=42)
        assert np.all(states.time == 42)

    def test_irregular_grid(self, ds):
        pytest.raises(AssertionError, bt.io.from_dataset, ds.isel({ "longitude": slice(0, -4) }))
        pytest.raises(AssertionError, bt.io.from_dataset, ds.isel({ "latitude": slice(0, -4) }))
        # Grid could be regular by shape, but coords are only a region
        pytest.raises(AssertionError, bt.io.from_dataset, ds.isel({
            "latitude": slice(0, 10), "longitude": slice(0, 18)
        }))
        # Coordinate values with irregular spacing
        pytest.raises(AssertionError, bt.io.from_dataset, ds.assign_coords({
            "longitude": [0., 10., 25., 30., *np.arange(40., 360., 10.)]
        }))
        pytest.raises(AssertionError, bt.io.from_dataset, ds.assign_coords({
            "latitude": [90., 80., 71., 60., *np.arange(50., -91., -10.)]
        }))

    def test_dims_order(self, ds):
        pytest.raises(AssertionError, bt.io.from_dataset, ds.transpose("time", "longitude", "latitude", "level"))
        pytest.raises(AssertionError, bt.io.from_dataset, ds.transpose("time", "latitude", "level", "longitude"))

    def test_mismatched_u_and_v(self, ds):
        import xarray as xr
        pytest.raises(AssertionError, bt.io.from_dataset, xr.merge([
            ds["u"],
            ds["v"].isel({ "time": 0 }, drop=True)
        ]))

    def test_with_missing(self, ds):
        pytest.raises(ValueError, bt.io.from_dataset, ds.isel({ "longitude": 0 }, drop=True))
        pytest.raises(ValueError, bt.io.from_dataset, ds.isel({ "latitude": 0 }, drop=True))
        pytest.raises(ValueError, bt.io.from_dataset, ds.drop_vars(["u", "vo", "avo"]))
        pytest.raises(ValueError, bt.io.from_dataset, ds.drop_vars(["v", "vo", "avo"]))
        pytest.raises(ValueError, bt.io.from_dataset, ds.drop_vars(["u", "v", "vo", "avo"]))

