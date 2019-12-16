from numbers import Number
import numpy as np
from .state import State


class BarotropicModel:
    """Integrate the barotropic PV equation on the sphere forward in time"""

    def __init__(self, rhs, diffusion_order=2, diffusion_kappa=1.e15):
        """Set up a numerical model of the barotropic PV equation on the sphere

        The integrated model equation is

            Dq/Dt = ∂q/∂t + ∇(u·q) = RHS,

        using ∇u = 0. The right-hand side RHS is specified by the argument
        `rhs` which, when called with the current model state, returns the RHS
        values in spectral space. Predefined RHS terms are found in the `.rhs`
        submodule.

        Numerical stability is enhanced by smoothing the PV field with
        a diffusion term at every step. The order and strength of this
        diffusion can be tuned with parmaters `diffusion_order` and
        `diffusion_kappa`.
        """
        self.rhs = rhs
        self.diffusion_order = diffusion_order
        self.diffusion_kappa = diffusion_kappa

    def run(self, state_init, dt, tend, save_every=0):
        if tend < state_init.time:
            return state_init, [state_init]
        # First integration with Euler forward step (multistep leapfrog scheme
        # requires initialization)
        state = self.euler(state_init, dt)
        state, state_new = state_init, state
        # Save model states during the integration
        states = [state]
        t_save = save_every
        # Integrate one step further than requested so Robert-Asselin filter is
        # applied to model state at the end
        while state.time < tend:
            state, state_new = self.leapfrog(state, state_new)
            if save_every > 0 and state.time >= t_save:
                states.append(state)
                t_save += save_every
        return state, states
    
    def euler(self, state_now, dt):
        """Step forward in time by `dt` starting from `state_now`"""
        dts = _to_seconds(dt)
        pv_new_spectral = state_now.pv_spectral + dts * self._pv_tendency_spectral(state_now)
        pv_new_spectral = self._apply_diffusion(state_now.grid, dts, pv_new_spectral)
        state_new = State(
                grid=state_now.grid,
                time=state_now.time + dt,
                pv_spectral=pv_new_spectral
                )
        return state_new

    def leapfrog(self, state_old, state_now, filter_parameter=0.2):
        """"Step forward in time using a filtered leapfrog scheme

        The timestep size is determined from the difference of the given input
        fields `state_old` and `state_now`.

        Both the current and the new model state are returned as the applied
        Robert-Asselin filter modifies the current state after the integration.
        The filter strength can be configured with the `filter_parameter`.
        """
        # Determine timestep from temporal difference of old and current state
        dt = state_now.time - state_old.time
        dts = _to_seconds(dt)
        # Evaluate the PV tendency at the current state and use it to advance
        # from the old state with twice the time step
        pv_new_spectral = state_old.pv_spectral + 2 * dts * self._pv_tendency_spectral(state_now)
        # Apply numerical diffusion
        pv_new_spectral = self._apply_diffusion(state_now.grid, dts, pv_new_spectral)
        state_new = State(
                grid=state_now.grid,
                time=state_now.time + dt,
                pv_spectral=pv_new_spectral
                )
        # Apply Robert-Asselin filter to current state
        pv_now_spectral = (
                (1. - 2. * filter_parameter) * state_now.pv_spectral
                + filter_parameter * (state_old.pv_spectral + pv_new_spectral)
                )
        state_now = State(
                grid=state_now.grid,
                time=state_now.time,
                pv_spectral=pv_now_spectral
                )
        return state_now, state_new

    def _pv_tendency_spectral(self, state):
        return - state.pv_flux_spectral + self.rhs(state)

    def _apply_diffusion(self, grid, dt, pv_spectral):
        eigenvalues_exp = grid.laplacian_eigenvalues ** self.diffusion_order
        return pv_spectral / ( 1. + 2. * dt * self.diffusion_kappa * eigenvalues_exp )


def _to_seconds(dt):
    """Return time interval dt as number of seconds
    
    Provides compatibility with both timedelta objects and direct input of dt
    as a number of seconds.
    """
    if isinstance(dt, Number):
        return dt
    return dt.total_seconds()

