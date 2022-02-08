from numbers import Number
import numpy as np
from . import formatting
from .state import State


class BarotropicModel:
    """Integrate the barotropic PV equation on the sphere forward in time.

    Parameters:
        rhs (:py:class:`rhs.RHS`): Model forcing. Build from the components
            provided in :py:mod:`rhs`.
        diffusion_order (int): Order of diffusion term added for numerical
            stability to the PV equation. Hyperdiffusion by default.
        diffusion_coeff (number): Strength of diffusion.

    Barotropic PV equation with forcing::

        Dq/Dt = ∂q/∂t + ∇(u·q) = RHS,

    where ``q`` is the barotropic potential vorticity (i.e. absolute vorticity)
    and ``u`` is the divergence-free horizontal wind (which can be obtained
    from the PV by inversion).

    Does not carry state aside from the forcing and diffusion setup and can be
    reused for multiple integrations with different :py:class`barotropic.State`
    instances.
    """

    def __init__(self, rhs=None, diffusion_order=2, diffusion_coeff=1.e15):
        self.rhs = rhs
        self.diffusion_order = diffusion_order
        self.diffusion_coeff = diffusion_coeff

    def __repr__(self):
        return formatting.barotropic_model_repr(self)

    def run(self, state_init, dt, tend, save_every=0):
        """Run the model until a specified time is reached.

        Parameters:
            state_init (:py:class:`State`): State from which the integration
                starts.
            dt (number | timedelta): Time step size. Either a number in seconds
                or a timedelta.
            tend (number | datetime): Time after which the integration stops.
                The model will not try to reach this time exactly, but stop as
                soon as it is exeeded.
            save_every (number): If larger than zero, an intermediate state is
                saved and returned in the list of states every time this time
                interval is exceeded. Does not work for datetime/timedelta
                **dt** currently.

        Returns:
            Tuple containing the final state and a list of all saved
            intermediate states.

        Initializes with a first-order Euler forward step, then continues with
        second-order Leapfrog steps.

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

        Parameters:
            state_now (:py:class:`State`): Initial state.
            dt (number | timedelta): Time step size.

        Returns:
            New :py:class:`State` instance.
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

        Parameters:
            state_old (:py:class:`State`): Last State.
            state_now (:py:class:`State`): Current state.
            filter_parameter (number): Strength paramter of the Robert-Asselin
                filter.

        Returns:
            Tuple containing filtered current state and next step as new
            :py:class:`State` instances.

        The timestep size is determined automatically from the difference of
        the given input fields **state_old** and **state_now**.
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
        return grid.solve_diffusion_spectral(
            pv_spectral,
            dt=dt,
            coeff=self.diffusion_coeff,
            order=self.diffusion_order
        )


def _to_seconds(dt):
    """Return time interval dt as number of seconds

    Provides compatibility with both timedelta objects and direct input of dt
    as a number of seconds.
    """
    if isinstance(dt, Number):
        return dt
    return dt.total_seconds()

