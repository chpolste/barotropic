import pytest

import numpy as np
from barotropic import Grid


class TestGrid:

    @pytest.mark.parametrize("resolution", [2.5, 5, 10., 30., 90.])
    @pytest.mark.parametrize("rsphere", [1000., 10000, 100000.])
    def test_gridpoint_area(self, resolution, rsphere):
        grid = Grid(resolution, rsphere=rsphere)
        # Area of sphere must be reproduced exactly
        area_sum = np.sum(grid.gridpoint_area)
        area_ref = 4 * np.pi * rsphere * rsphere
        assert np.allclose(area_sum, area_ref)

    @pytest.mark.parametrize("resolution", [2.5, 5, 10., 30., 90.])
    @pytest.mark.parametrize("rsphere", [1000., 10000, 100000.])
    def test_quad_sptrapz(self, resolution, rsphere):
        grid = Grid(resolution, rsphere=rsphere)
        y = np.ones_like(grid.lat)
        # Integration mus reproduce area of sphere exactly
        area_sum = grid.quad_sptrapz(y)
        area_ref = 4 * np.pi * rsphere * rsphere
        assert np.allclose(area_sum, area_ref)

    def test_derivative_meridional_1D_with_phi(self):
        grid = Grid()
        ddphi2 = grid.derivative_meridional(grid.phis, order=2)
        ddphi4 = grid.derivative_meridional(grid.phis, order=4)
        # For 1D input, polar points should be set to 0
        assert np.allclose(ddphi2[ 0], 0.)
        assert np.allclose(ddphi2[-1], 0.)
        assert np.allclose(ddphi4[ 0], 0.)
        assert np.allclose(ddphi4[-1], 0.)
        # Other points are normal
        assert np.allclose(ddphi2[1:-1], 1.)
        assert np.allclose(ddphi4[1:-1], 1.)

    def test_derivative_meridional_1D_with_lats(self):
        grid = Grid()
        ddphi2 = grid.derivative_meridional(grid.lats, order=2)
        ddphi4 = grid.derivative_meridional(grid.lats, order=4)
        assert np.allclose(ddphi2[ 0], 0.)
        assert np.allclose(ddphi2[-1], 0.)
        assert np.allclose(ddphi4[ 0], 0.)
        assert np.allclose(ddphi4[-1], 0.)
        assert np.allclose(ddphi2[1:-1], 180/np.pi)
        assert np.allclose(ddphi4[1:-1], 180/np.pi)


class TestRegion:

    def test_global(self):
        grid = Grid(resolution=10.)
        region = grid.region[:,:]
        assert region.shape == grid.shape
        assert np.allclose(region.lats, grid.lats)
        assert np.allclose(region.lons, grid.lons)
        assert np.allclose(region.extract(grid.lons), grid.lons)
        assert np.allclose(region.extract(grid.lats), grid.lats)
        assert np.allclose(region.extract(grid.lon), grid.lon)

    @pytest.mark.parametrize("lo,hi", [(30, 60), (60, 30)])
    def test_latitude(self, lo, hi):
        grid = Grid(resolution=10.)
        region = grid.region[lo:hi]
        assert region.shape == (4, 36)
        assert np.allclose(region.lats, [60., 50., 40., 30.])
        assert np.allclose(region.lons, grid.lons)
        assert np.all(region.extract(grid.lats) <= 60.)
        assert np.all(30. <= region.extract(grid.lats))
        assert np.all(region.extract(grid.lat) <= 60.)
        assert np.all(30. <= region.extract(grid.lat))
        assert np.allclose(region.extract(grid.lons), grid.lons)

    def test_longitude1(self):
        grid = Grid(resolution=10.)
        region = grid.region[:,265:]
        assert region.shape == (19, 9)
        assert np.allclose(region.lats, grid.lats)
        assert np.allclose(region.lons, [270, 280, 290, 300, 310, 320, 330, 340, 350])
        assert np.all(region.extract(grid.lons) < 360.)
        assert np.all(265 <= region.extract(grid.lons))
        assert np.all(region.extract(grid.lon) < 360.)
        assert np.all(265. <= region.extract(grid.lon))
        assert np.allclose(region.extract(grid.lats), grid.lats)

    def test_longitude2(self):
        grid = Grid(resolution=10.)
        region = grid.region[:,310:40]
        assert region.shape == (19, 10)
        assert np.allclose(region.lats, grid.lats)
        assert np.allclose(region.lons, [310, 320, 330, 340, 350, 0, 10, 20, 30, 40])
        assert region.extract(grid.lons).shape == (10,)
        assert region.extract(grid.lon).shape == (19, 10)
        assert np.allclose(region.extract(grid.lats), grid.lats)

    def test_combined(self):
        grid = Grid(resolution=10.)
        region = grid.region[:-50,20:50]
        assert region.shape == (5, 4)
        assert np.allclose(region.lats, [-50, -60, -70, -80, -90])
        assert np.allclose(region.lons, [20, 30, 40, 50])
        assert np.allclose(region.extract(grid.lats), [-50, -60, -70, -80, -90])
        assert np.allclose(region.extract(grid.lons), [20, 30, 40, 50])
        assert region.extract(grid.gridpoint_area).shape == (5, 4)

    def test_single(self):
        grid = Grid(resolution=10.)
        region = grid.region[40:40,80:80]
        assert region.shape == (1, 1)
        assert np.allclose(region.lats, [40])
        assert np.allclose(region.lons, [80])
        assert region.extract(grid.fcor).shape == (1, 1)

    def test_empty(self):
        grid = Grid(resolution=10.)
        region = grid.region[-200:-300,500:-200]
        assert region.shape == (0, 0)
        assert region.lats.size == 0
        assert region.lons.size == 0
        assert region.extract(grid.lats).size == 0

