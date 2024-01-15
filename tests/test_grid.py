import pytest

import numpy as np
from barotropic import Grid


class TestGrid:

    @pytest.mark.parametrize("resolution", [2.5, 5, 10., 30., 90.])
    @pytest.mark.parametrize("rsphere", [1000., 10000, 100000.])
    def test_gridpoint_area(self, resolution, rsphere):
        grid = Grid(resolution, rsphere=rsphere)
        # Area of sphere must be reproduced exactly
        area_sum = np.sum(grid.gridpoint_area) * grid.nlon
        area_ref = 4 * np.pi * rsphere * rsphere
        assert np.allclose(area_sum, area_ref)

    @pytest.mark.parametrize("resolution", [2.5, 5, 10., 30., 90.])
    @pytest.mark.parametrize("rsphere", [1000., 10000, 100000.])
    @pytest.mark.parametrize("method", ["boxcount", "sptrapz"])
    def test_quad(self, resolution, rsphere, method):
        grid = Grid(resolution, rsphere=rsphere)
        y = np.ones_like(grid.lat2)
        # Integration mus reproduce area of sphere exactly
        area_sum = grid.quad(y, method=method)
        area_ref = 4 * np.pi * rsphere * rsphere
        assert np.allclose(area_sum, area_ref)
        # Method must fail for other inputs
        with pytest.raises(Exception):
            grid.quad_sptrapz(1.)
        with pytest.raises(Exception):
            grid.quad_sptrapz(np.ones(grid.nlat, dtype=float))
        with pytest.raises(Exception):
            grid.quad_sptrapz(np.ones(grid.nlon, dtype=float))

    def test_derivative_meridional_1D_with_phi(self):
        grid = Grid()
        ddphi2 = grid.derivative_meridional(grid.phi, order=2)
        ddphi4 = grid.derivative_meridional(grid.phi, order=4)
        # For 1D input, polar points should be set to 0
        assert np.allclose(ddphi2[ 0], 0.)
        assert np.allclose(ddphi2[-1], 0.)
        assert np.allclose(ddphi4[ 0], 0.)
        assert np.allclose(ddphi4[-1], 0.)
        # Other points are normal
        assert np.allclose(ddphi2[1:-1], 1.)
        assert np.allclose(ddphi4[1:-1], 1.)

    def test_derivative_meridional_1D_with_lat(self):
        grid = Grid()
        ddphi2 = grid.derivative_meridional(grid.lat, order=2)
        ddphi4 = grid.derivative_meridional(grid.lat, order=4)
        assert np.allclose(ddphi2[ 0], 0.)
        assert np.allclose(ddphi2[-1], 0.)
        assert np.allclose(ddphi4[ 0], 0.)
        assert np.allclose(ddphi4[-1], 0.)
        assert np.allclose(ddphi2[1:-1], 180/np.pi)
        assert np.allclose(ddphi4[1:-1], 180/np.pi)

    @pytest.mark.parametrize("resolution", [2.5, 5, 10., 30., 90.])
    def test_mean_ones(self, resolution):
        grid = Grid(resolution=resolution)
        field = np.ones(grid.shape)
        assert np.isclose(grid.mean(field), 1.)
        assert np.allclose(grid.mean(field, axis="zonal"), np.ones(grid.nlat))
        assert np.allclose(grid.mean(field, axis="meridional"), np.ones(grid.nlon))
        # Test region sizes
        region = grid.region[30:280,-30:70]
        assert grid.mean(field, region=region).size == 1
        assert grid.mean(field, region=region, axis="zonal").shape == (region.shape[0],)
        assert grid.mean(field, region=region, axis="meridional").shape == (region.shape[1],)

    @pytest.mark.parametrize("resolution", [2.5, 5, 10., 30., 90.])
    def test_mean_east_west_split(self, resolution):
        grid = Grid(resolution=resolution)
        halflon = grid.nlon // 2
        field = np.ones(grid.shape)
        field[:,halflon:] = 0.
        assert np.isclose(grid.mean(field), 0.5)
        assert np.allclose(grid.mean(field, axis="zonal"), 0.5)
        assert np.allclose(grid.mean(field, axis="meridional"), ([1.] * halflon) + ([0.] * halflon))

    @pytest.mark.parametrize("resolution", [2.5, 5, 10., 30., 90.])
    def test_mean_north_south_split(self, resolution):
        grid = Grid(resolution=resolution)
        halflat = grid.nlat // 2
        field = np.ones(grid.shape)
        field[halflat,:] = 0.5
        field[halflat+1:,:] = 0.
        assert np.isclose(grid.mean(field), 0.5)
        assert np.allclose(grid.mean(field, axis="zonal"), ([1.] * halflat) + [0.5] + ([0.] * halflat))
        assert np.allclose(grid.mean(field, axis="meridional"), 0.5)
        assert np.allclose(grid.mean(field, region=grid.region[:,20:]), 1.) # Northern region
        assert np.allclose(grid.mean(field, region=grid.region[:,-1:1]), 0.5) # Equator region
        assert np.allclose(grid.mean(field, region=grid.region[10:260,:-40]), 0.) # Southern region


class TestGridFiltering:

    @pytest.mark.parametrize("window", ["boxcar", "hann", "triang"])
    def test_get_filter_window_width(self, window):
        grid = Grid(resolution=1.)
        assert grid.get_filter_window(window, 7.).shape == (7,)
        assert grid.get_filter_window(window, 9.).shape == (9,)
        # If width isn't odd on the grid, next-larger is chosen
        assert grid.get_filter_window(window, 8.).shape == (9,)
        assert grid.get_filter_window(window, 14.4).shape == (15,)

    def test_filter_meridional_call(self):
        grid = Grid(resolution=2.0)
        assert grid.filter_meridional(grid.lat, "boxcar", 10.).shape == (grid.nlat,)
        assert grid.filter_meridional(grid.lat2, "hann", 14.).shape == grid.shape
        assert grid.filter_meridional(grid.lat, np.array([0.1, 0.2, 0.3, 0.1, 0.3])).shape == (grid.nlat,)
        with pytest.raises(AssertionError):
            grid.filter_meridional(grid.lon, "hann", 13.)

    def test_filter_zonal_call(self):
        grid = Grid(resolution=2.0)
        assert grid.filter_zonal(grid.lon, "boxcar", 10.).shape == (grid.nlon,)
        assert grid.filter_zonal(grid.lon2, "hann", 14.).shape == grid.shape
        assert grid.filter_zonal(grid.lon, np.array([0.1, 0.2, 0.3, 0.1, 0.3])).shape == (grid.nlon,)
        with pytest.raises(AssertionError):
            grid.filter_zonal(grid.lat, "hann", 13.)



class TestGridRegion:

    def test_global(self):
        grid = Grid(resolution=10.)
        region = grid.region[:,:]
        assert region.shape == grid.shape
        assert np.allclose(region.lat, grid.lat)
        assert np.allclose(region.lon, grid.lon)
        assert np.allclose(region.extract(grid.lon), grid.lon)
        assert np.allclose(region.extract(grid.lat), grid.lat)
        assert np.allclose(region.extract(grid.lon2), grid.lon2)

    @pytest.mark.parametrize("lo,hi", [(30, 60), (60, 30)])
    def test_latitude(self, lo, hi):
        grid = Grid(resolution=10.)
        region = grid.region[:,lo:hi]
        assert region.shape == (4, 36)
        assert np.allclose(region.lat, [60., 50., 40., 30.])
        assert np.allclose(region.lon, grid.lon)
        assert np.all(region.extract(grid.lat) <= 60.)
        assert np.all(30. <= region.extract(grid.lat))
        assert np.all(region.extract(grid.lat2) <= 60.)
        assert np.all(30. <= region.extract(grid.lat2))
        assert np.allclose(region.extract(grid.lon), grid.lon)

    def test_longitude1(self):
        grid = Grid(resolution=10.)
        region = grid.region[265:,:]
        assert region.shape == (19, 9)
        assert np.allclose(region.lat, grid.lat)
        assert np.allclose(region.lon, [270, 280, 290, 300, 310, 320, 330, 340, 350])
        assert np.all(region.extract(grid.lon) < 360.)
        assert np.all(265 <= region.extract(grid.lon))
        assert np.all(region.extract(grid.lon2) < 360.)
        assert np.all(265. <= region.extract(grid.lon2))
        assert np.allclose(region.extract(grid.lat), grid.lat)

    def test_longitude2(self):
        grid = Grid(resolution=10.)
        region = grid.region[310:40,:]
        assert region.shape == (19, 10)
        assert np.allclose(region.lat, grid.lat)
        assert np.allclose(region.lon, [310, 320, 330, 340, 350, 0, 10, 20, 30, 40])
        assert region.extract(grid.lon).shape == (10,)
        assert region.extract(grid.lon2).shape == (19, 10)
        assert np.allclose(region.extract(grid.lat), grid.lat)

    def test_combined(self):
        grid = Grid(resolution=10.)
        region = grid.region[20:50,:-50]
        assert region.shape == (5, 4)
        assert np.allclose(region.lat, [-50, -60, -70, -80, -90])
        assert np.allclose(region.lon, [20, 30, 40, 50])
        assert np.allclose(region.extract(grid.lat), [-50, -60, -70, -80, -90])
        assert np.allclose(region.extract(grid.lon), [20, 30, 40, 50])
        assert region.extract(grid.phi2).shape == (5, 4)

    def test_single(self):
        grid = Grid(resolution=10.)
        region = grid.region[80:80,40:40]
        assert region.shape == (1, 1)
        assert np.allclose(region.lat, [40])
        assert np.allclose(region.lon, [80])
        assert region.extract(grid.fcor2).shape == (1, 1)

    def test_empty(self):
        grid = Grid(resolution=10.)
        region = grid.region[500:-200,-200:-300]
        assert region.shape == (0, 0)
        assert region.lat.size == 0
        assert region.lon.size == 0
        assert region.extract(grid.lat).size == 0

