from numbers import Number
import numpy as np
from . import formatting
from .state import State


class BarotropicModel:
    """Integrate the barotropic PV equation on the sphere forward in time.
    
        Dq/Dt = ∂q/∂t + ∇(u·q) = RHS,

    where `q` is the barotropic potential vorticity (i.e. absolute vorticity)
    and `u` is the divergence-free horizontal wind (which can be obtained from
    the PV by inversion).

    A `BarotropicModel` instance does not contain state aside from the forcing
    and diffusion setup and can be reused for multiple integrations with
    different `barotropic.State`s.
    """

    def __init__(self, rhs=None, diffusion_order=2, diffusion_coeff=1.e15):
        """BarotropicModel constructor.

        - `rhs` specifies the model forcing, which can be built from the components
          provided in `barotropic.rhs` or a custom implementation with the same
          interface.
        - Numerical stability is enhanced by smoothing the PV field with
          a laplacian diffusion term after every time step. The order and strength
          of this diffusion can be tuned with parameters `diffusion_order` and
          `diffusion_coeff`.
        """
        self.rhs = rhs
        self.diffusion_order = diffusion_order
        self.diffusion_coeff = diffusion_coeff

    def __repr__(self):
        return formatting.barotropic_model_repr(self)

    def run(self, state_init, dt, tend, save_every=0):
        """Run the model until a specified time is reached.

        Initializes with a first-order Euler forward step, then continues with
        second-order Leapfrog steps.

        - `state_init` is the initial state from which the integration starts.
        - `dt` is the step size used in the integration.
        - `tend` is the time after which the integration stops. The model will
          not try to reach this time exactly, but stop as soon as `tend` is
          exeeded.
        - If `save_every` is larger than zero, an intermediate state is saved
          and then returned every time this time interval is exceeded.

        Time and time intervals should be given as a number type representing
        seconds if `state_init` uses a number type to denote time, or
        `datetime.datetime` and `datetime.timedelta` if `state_init` uses
        `datetime.datetime` to denote time.

        Returns the final state and a list of all saved intermediate states.
        """
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
        """Step forward in time with a first-order Euler-forward scheme

        Advances by `dt` starting from `state_now` and returns the new state.
        """
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
        """Step forward in time using a filtered leapfrog scheme.

        - The timestep size is determined automatically from the difference of
          the given input fields `state_old` and `state_now`.
        - The filter strength can be configured with the `filter_parameter`.

        Both the current and the new model state are returned in a tuple since
        the Robert-Asselin filter modifies the current state too.
        """
        # Determine timestep from temporal difference of old and current state
        dt = state_now.time - state_old.time
        dts = _to_seconds(dt)
        # Evaluate the PV tendency at the current state and use it to advance
        # from the old state with twice the time step
        pv_new_spectral = state_old.pv_spectral + 2 * dts * self._pv_tendency_spectral(state_now)
        # Apply numerical diffusion
        pv_new_spectral = self._apply_diffusion(state_now.grid, 2 * dts, pv_new_spectral)
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
        """Evaluate ∂q/∂t = -∇(u·q) + RHS in spectral space"""
        tendency = -state.pv_flux_spectral 
        if self.rhs is not None:
            tendency += state.grid.to_spectral(self.rhs(state))
        return tendency 

    def _apply_diffusion(self, grid, dt, pv_spectral):
        eigenvalues_exp = grid.laplacian_eigenvalues ** self.diffusion_order
        return pv_spectral / ( 1. + dt * self.diffusion_coeff * eigenvalues_exp )


def _to_seconds(dt):
    """Return time interval dt as number of seconds
    
    Provides compatibility with both timedelta objects and direct input of dt
    as a number of seconds.
    """
    if isinstance(dt, Number):
        return dt
    return dt.total_seconds()

