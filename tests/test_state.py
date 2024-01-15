import pytest

import numpy as np
from barotropic import Grid, State


# np.isclose and np.allclose must be given an appropriate value of atol for
# different fields. Also, pyspharm uses single-precision floats, so the values
# must account for the limited accuracy possible even when comparing doubles.
ATOL_WIND = 1e-4 # ...
ATOL_PV   = 1e-10 # ...
ATOL_ZETA = 1e-10 # ...
ATOL_PSI  = 1e+1 # ...



class TestConversions:

    def test_motionless(self):
        grid = Grid()
        zeros = np.zeros(grid.shape, dtype=float)
        from_wind = State.from_wind(grid, 0., zeros, zeros)
        from_vort = State.from_vorticity(grid, 0., zeros)
        from_pv   = State(grid, 0., pv=grid.fcor2)
        for state in [from_wind, from_vort, from_pv]:
            assert np.allclose(state.pv, grid.fcor2, atol=ATOL_PV)
            assert np.allclose(state.u, 0., atol=ATOL_WIND)
            assert np.allclose(state.v, 0., atol=ATOL_WIND)
            assert np.allclose(state.vorticity, 0., atol=ATOL_ZETA)
            assert np.allclose(state.streamfunction, 0., atol=ATOL_PSI)

    def test_pv_to_other_to_pv(self):
        grid = Grid()
        pv = grid.fcor2 + 1.5e-5 * np.sin(2*grid.phi2)**3 * np.sin(3*grid.lam2)
        state0 = State(grid, 0., pv=pv)
        # Via wind
        state1 = State.from_wind(grid, 0., state0.u, state0.v)
        assert np.allclose(state0.pv, state1.pv, atol=ATOL_PV)
        assert np.allclose(state0.u,  state1.u,  atol=ATOL_WIND)
        assert np.allclose(state0.v,  state1.v,  atol=ATOL_WIND)
        # Via vorticity
        state2 = State.from_vorticity(grid, 0., state0.vorticity)
        assert np.allclose(state0.pv, state2.pv, atol=ATOL_PV)
        assert np.allclose(state0.u,  state2.u,  atol=ATOL_WIND)
        assert np.allclose(state0.v,  state2.v,  atol=ATOL_WIND)

    def test_vorticity_to_other_to_vorticity(self):
        grid = Grid()
        vort = 2.0e-5 * np.sin(2*grid.phi2)**3 * np.cos(2*grid.lam2)
        state0 = State.from_vorticity(grid, 0., vort)
        # Via PV
        state1 = State(grid, 0., pv=state0.pv)
        assert np.allclose(state0.pv, state1.pv, atol=ATOL_PV)
        assert np.allclose(state0.u,  state1.u,  atol=ATOL_WIND)
        assert np.allclose(state0.v,  state1.v,  atol=ATOL_WIND)
        # Via wind
        state2 = State.from_wind(grid, 0., state0.u, state0.v)
        assert np.allclose(state0.pv, state2.pv, atol=ATOL_PV)
        assert np.allclose(state0.u,  state2.u,  atol=ATOL_WIND)
        assert np.allclose(state0.v,  state2.v,  atol=ATOL_WIND)

    def test_wind_to_other_to_wind(self):
        grid = Grid()
        u = 15. * np.sin(2*grid.phi2)**3
        v = 0.5 * u * np.sin(2*grid.lam2)
        state0 = State.from_wind(grid, 0., u, v)
        # Via PV
        state1 = State(grid, 0., state0.pv)
        assert np.allclose(state0.pv, state1.pv, atol=ATOL_PV)
        assert np.allclose(state0.u,  state1.u,  atol=ATOL_WIND)
        assert np.allclose(state0.v,  state1.v,  atol=ATOL_WIND)
        # Via vorticity
        state2 = State.from_vorticity(grid, 0., state0.vorticity)
        assert np.allclose(state0.pv, state2.pv, atol=ATOL_PV)
        assert np.allclose(state0.u,  state2.u,  atol=ATOL_WIND)
        assert np.allclose(state0.v,  state2.v,  atol=ATOL_WIND)

