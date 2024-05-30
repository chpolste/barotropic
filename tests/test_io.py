import pytest

import numpy as np
import barotropic as bt


class TestDataset:

    def test_as_dataset_empty(self):
        assert not bt.io.as_dataset([])
        assert not bt.io.as_dataset(bt.StateList([]))

    def test_as_dataset_single(self):
        grid = bt.Grid()
        ds = bt.io.as_dataset(bt.init.motionless(grid))
        assert ds
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        assert "time" in ds.coords
        assert ds.coords["time"].size == 1

    def test_as_dataset_fields(self):
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

    def test_as_dataset_concat_dim(self):
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

