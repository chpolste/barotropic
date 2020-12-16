import unittest

import numpy as np
from barotropic import Grid


class TestGrid(unittest.TestCase):

    def test_gridpoint_area(self):
        """Area of the sphere must be reproduced exactly"""
        # Test with different radii (r) and grid resolutions (x)
        for r in range(1000, 10000, 100000):
            for x in [2.5, 5, 10, 30, 90]:
                with self.subTest(r=r, x=x):
                    grid = Grid(rsphere=float(r), resolution=x)
                    area_sum = np.sum(grid.gridpoint_area)
                    area_ref = 4 * np.pi * r * r
                    self.assertTrue(np.isclose(area_sum, area_ref))

    def test_quad_sptrapz(self):
        """Integration over 1 must reproduce area of sphere exactly"""
        # Test with different radii (r) and grid resolutions (x)
        for r in range(1000, 10000, 100000):
            for x in [2.5, 5, 10, 30, 90]:
                with self.subTest(r=r, x=x):
                    grid = Grid(rsphere=float(r), resolution=x)
                    y = np.ones_like(grid.lat)
                    area_sum = grid.quad_sptrapz(y)
                    area_ref = 4 * np.pi * r * r
                    self.assertTrue(np.isclose(area_sum, area_ref))

    def test_ddphi_1D_with_phi(self):
        """"""
        grid = Grid()
        ddphi2 = grid.ddphi(grid.phis, order=2)
        ddphi4 = grid.ddphi(grid.phis, order=4)
        # For 1D input, polar points should be set to 0
        self.assertTrue(np.isclose(ddphi2[ 0], 0.))
        self.assertTrue(np.isclose(ddphi2[-1], 0.))
        self.assertTrue(np.isclose(ddphi4[ 0], 0.))
        self.assertTrue(np.isclose(ddphi4[-1], 0.))
        # Other points are normal
        self.assertTrue(all(np.isclose(ddphi2[1:-1], 1.)))
        self.assertTrue(all(np.isclose(ddphi4[1:-1], 1.)))

    def test_ddphi_1D_with_lats(self):
        grid = Grid()
        ddphi2 = grid.ddphi(grid.lats, order=2)
        ddphi4 = grid.ddphi(grid.lats, order=4)
        self.assertTrue(np.isclose(ddphi2[ 0], 0.))
        self.assertTrue(np.isclose(ddphi2[-1], 0.))
        self.assertTrue(np.isclose(ddphi4[ 0], 0.))
        self.assertTrue(np.isclose(ddphi4[-1], 0.))
        self.assertTrue(all(np.isclose(ddphi2[1:-1], 180/np.pi)))
        self.assertTrue(all(np.isclose(ddphi4[1:-1], 180/np.pi)))


if __name__ == '__main__':
    unittest.main()

