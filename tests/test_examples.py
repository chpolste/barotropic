import pytest

import barotropic as bt


def test_readme_example():
    # Run the example from the README file
    grid    = bt.Grid(resolution=2.5)
    initial = bt.init.solid_body_rotation(grid, amplitude=15)
    forcing = bt.rhs.GaussianMountain(center=(30, 45), stdev=(10, 10), height=2000)
    model   = bt.BarotropicModel(forcing, diffusion_order=2)
    last, all_states = model.run(initial, 15*bt.MIN, 10*bt.DAY, save_every=6*bt.HOUR)
    # Just make sure that it ran without issues and check a few very basic properties
    assert 10*bt.DAY >= last.time
    assert last.time <= (10*bt.DAY + 15*bt.MIN)
    assert len(all_states) == 1 + 10*4 # initial state + 4 states per day of integration

