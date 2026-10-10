import math
import torch


def state_derivative(acceleration, state, u):
    """Return (dy/dt, dv/dt) = (v, acceleration(y, v, u)).

    The last dimension of state contains [y, v]. Leading dimensions can
    represent a batch of independent trajectories.
    """
    y = state[..., 0]
    v = state[..., 1]
    a = acceleration(y, v, u)
    return torch.stack((v, a), dim=-1)


def rk4_step(acceleration, state, u_start, u_end, dt):
    """Advance the state by dt with classical fourth-order Runge-Kutta.

    The measured input is linearly interpolated between the two endpoints.
    The k values are derivatives; multiplication by dt happens below.
    """
    u_mid = (u_start + u_end) / 2

    k1 = state_derivative(acceleration, state, u_start)
    k2 = state_derivative(acceleration, state + dt * k1 / 2, u_mid)
    k3 = state_derivative(acceleration, state + dt * k2 / 2, u_mid)
    k4 = state_derivative(acceleration, state + dt * k3, u_end)

    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rollout(acceleration, initial_state, u, dt):
    """Simulate a complete input sequence without resetting the state.

    For a single trajectory:
        initial_state has shape (2,), containing [y0, v0].
        u has shape (N,).
        The returned states have shape (N, 2).

    For a batch:
        initial_state has shape (B, 2), u has shape (N, B), and the
        returned states have shape (N, B, 2).

    N input samples require N - 1 integration steps. states[0] is the
    supplied initial state; states[i] corresponds to input sample u[i].
    initial_state and u must be floating-point tensors of the same dtype
    and device. No measured outputs are used after initialization.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive, finite sampling interval.")
    if initial_state.ndim < 1 or initial_state.shape[-1] != 2:
        raise ValueError("The last dimension of initial_state must contain [y, v].")
    if u.ndim < 1 or u.shape[0] == 0:
        raise ValueError("u must contain at least one input sample.")
    if u.shape[1:] != initial_state.shape[:-1]:
        raise ValueError("The batch dimensions of u and initial_state must match.")
    if not initial_state.is_floating_point() or not u.is_floating_point():
        raise ValueError("initial_state and u must be floating-point tensors.")
    if initial_state.dtype != u.dtype or initial_state.device != u.device:
        raise ValueError("initial_state and u must have the same dtype and device.")

    state = initial_state
    states = [state]

    for i in range(u.shape[0] - 1):
        state = rk4_step(acceleration, state, u[i], u[i + 1], dt)
        states.append(state)

    # Stacking existing tensors preserves the computation graph for training.
    return torch.stack(states, dim=0)