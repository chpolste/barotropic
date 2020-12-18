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

