"""
Sink-particle formation, following Federrath et al. (2010), Sect. 2.2.

After every hydro step, a slot of the sink-particle buffer opens at a cell
that

- lies above the density threshold
  ``rho_thr = pi c_s^2 / (4 G r_acc^2)`` (their Eq. 32), with ``r_acc`` the
  accretion radius ``config.sink_accretion_radius`` (in cells),
- is the potential minimum of its control volume, the sphere of radius
  ``r_acc`` around it,
- passes every enabled check (converging flow, Jeans instability,
  boundedness),
- has no occupied slot within the accretion radius, and
- has no other candidate of the same step within the accretion radius with a
  lower potential.

The density threshold, converging-flow and potential-minimum checks are
evaluated on the whole grid; the Jeans and bound-state checks, which need the
self-gravity of the control volume, only on the cells that pass them.

A slot that opens takes ``max(rho_c - rho_thr, 0) * V`` from every cell ``c``
within its accretion radius (a cell within the radius of several opening
slots feeds the closest one). Occupied slots take no gas: there is no
accretion yet.

The grid is periodic: neighbours are reached by rolling the arrays and slot
distances use the minimum image.
"""

# general
from functools import partial
import math

# numerics
import numpy as np

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    ISOTHERMAL,
    PERIODIC_ROLL,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_state_struct import SinkParticles
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._gravity._gravity import _compute_total_potential
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._stencil_operations._stencil_operations import _stencil_add

def _sphere_offsets(radius):
    """Integer cell offsets whose centres lie within ``radius`` cells."""
    n = int(math.floor(radius))
    return [
        (i, j, k)
        for i in range(-n, n + 1)
        for j in range(-n, n + 1)
        for k in range(-n, n + 1)
        if i * i + j * j + k * k <= radius**2
    ]


def _shifted(field, offset):
    """``field`` at cell ``c + offset``, for fields whose last three axes are
    spatial."""
    return jnp.roll(field, tuple(-o for o in offset), axis=(-3, -2, -1))


def _periodic_distance_squared(a, b, box):
    """Squared minimum-image distance between positions ``a`` and ``b`` (last
    axis 3)."""
    d = a - b
    d = d - box * jnp.round(d / box)
    return jnp.sum(d**2, axis=-1)


def _converging_flow_check(u, config):
    """
    Converging-flow check, 1.0 where it passes (or is disabled) and 0.0 where
    not.

    Args:
        u: Gas velocities, shape (3, nx, ny, nz).
        config: The simulation configuration.

    Returns:
        The check, shape (nx, ny, nz).
    """
    checks = jnp.ones_like(u[0])

    # Gas must flow inward along each axis separately; a region squeezed along
    # one axis and escaping along another (a shock) fails.
    if config.sink_converging_flow_check:
        for axis in range(3):
            du = _stencil_add(u[axis], indices=(1, -1), factors=(1.0, -1.0), axis=axis)
            checks = checks * (du < 0).astype(u.dtype)

    return checks


def _control_volume_checks(centre_cell, m, u, c_s2, G, config):
    """
    Jeans-instability and bound-state checks on the control volumes of the
    given cells, as boolean arrays that are True where the enabled checks
    pass.

    The energies follow Federrath et al. (2010), Eqs. 5, 6, 11 and 12. The
    potential in Eq. 6 is the one of the gas inside the control volume, each
    cell a point mass, so ``E_grav = -G sum_{a != b} M_a M_b / r_ab``.

    Args:
        centre_cell: Integer indices of the central cells, shape (K, 3).
        m: Cell masses, shape (nx, ny, nz).
        u: Gas velocities, shape (3, nx, ny, nz).
        c_s2: Squared sound speeds, shape (nx, ny, nz).
        G: The gravitational constant.
        config: The simulation configuration.

    Returns:
        The combined check, shape (K,).
    """
    passed = jnp.ones(centre_cell.shape[0], dtype=bool)
    if not (config.sink_jeans_check or config.sink_bound_check):
        return passed

    sphere = np.array(_sphere_offsets(config.sink_accretion_radius))

    # inverse cell-cell distances in the control volume, without self-terms
    separation = np.linalg.norm(sphere[:, None] - sphere[None], axis=-1)
    inverse_separation = np.where(
        separation > 0, 1.0 / np.where(separation > 0, separation, 1.0), 0.0
    ) / config.grid_spacing

    # (K, num_offsets)
    cell = centre_cell[:, None, :] + sphere[None]
    flat_cell = jnp.ravel_multi_index(
        tuple(cell[..., axis] for axis in range(3)), m.shape, mode="wrap"
    )
    mass = m.ravel()[flat_cell]
    velocity = jnp.moveaxis(u.reshape(3, m.size)[:, flat_cell], 0, -1)

    e_grav = -G * jnp.einsum("ka,ab,kb->k", mass, inverse_separation, mass)
    e_th = 0.5 * jnp.sum(mass * c_s2.ravel()[flat_cell], axis=1)

    if config.sink_jeans_check:
        passed = passed & (-e_grav > 2.0 * e_th)

    if config.sink_bound_check:
        # kinetic energy in the centre-of-mass frame of the control volume
        momentum = jnp.sum(mass[..., None] * velocity, axis=1)
        e_kin = (
            0.5 * jnp.sum(mass * jnp.sum(velocity**2, axis=-1), axis=1)
            - 0.5 * jnp.sum(momentum**2, axis=-1) / jnp.sum(mass, axis=1)
        )
        passed = passed & (e_grav + e_th + e_kin < 0)

    return passed


def _form_sinks(
    state: STATE_TYPE,
    phi,
    sinks: SinkParticles,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
):
    """
    Move gas into the sink slots, on the unpadded periodic grid.

    Args:
        state: The unpadded primitive state.
        phi: The gravitational potential on the same grid.
        sinks: The sink-particle buffer.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        ``(state, sinks)`` after the transfer.
    """
    density_index = registered_variables.density_index
    pressure_index = registered_variables.pressure_index
    velocity_index = registered_variables.velocity_index

    rho = state[density_index]
    u = jnp.stack([
        state[velocity_index.x], state[velocity_index.y], state[velocity_index.z]
    ])
    p = state[pressure_index]

    shape = rho.shape
    num_cells = rho.size
    num_slots = config.num_sink_slots
    dx = config.grid_spacing
    volume = dx**3
    box = jnp.array(shape) * dx
    accretion_radius = config.sink_accretion_radius * dx
    G = params.gravitational_constant

    m = rho * volume
    if config.equation_of_state == ISOTHERMAL:
        c_s2 = params.isothermal_sound_speed**2 * jnp.ones_like(rho)
    else:
        c_s2 = params.gamma * p / rho

    # Density threshold of Federrath et al. (2010), Eq. 32: the Jeans length
    # is resolved by the accretion diameter. Per cell through the local sound
    # speed.
    rho_thr = jnp.pi * c_s2 / (4.0 * G * accretion_radius**2)

    converging = _converging_flow_check(u, config)

    # -------------------------------------------------------------
    # ================= ↓ opening new slots ↓ =====================
    # -------------------------------------------------------------

    # Lowest potential in each cell's control volume; a cell equal to it is
    # the potential minimum
    phi_min = phi
    for offset in _sphere_offsets(config.sink_accretion_radius):
        phi_min = jnp.minimum(phi_min, _shifted(phi, offset))

    candidate = (phi == phi_min) & (converging == 1.0) & (rho > rho_thr)

    # At most num_slots slots can open in one step, which bounds the list. The
    # Jeans and bound checks come after the list is filled, so if more cells
    # pass the checks above than there are slots, the surplus is dropped for
    # this step.
    candidate_cell = jnp.nonzero(candidate.ravel(), size=num_slots, fill_value=-1)[0]
    valid = candidate_cell >= 0
    candidate_cell = jnp.maximum(candidate_cell, 0)
    candidate_index = jnp.stack(jnp.unravel_index(candidate_cell, shape), axis=-1)
    candidate_position = (candidate_index + 0.5) * dx

    valid = valid & _control_volume_checks(candidate_index, m, u, c_s2, G, config)
    candidate_phi = phi.ravel()[candidate_cell]

    # no occupied slot within the accretion radius
    occupied = sinks.mass > 0
    distance_to_slots = _periodic_distance_squared(
        candidate_position[:, None], sinks.position[None, :], box
    )
    valid = valid & ~jnp.any(
        occupied[None, :] & (distance_to_slots <= accretion_radius**2), axis=1
    )

    # of two candidates closer than the accretion radius, the one with lower
    # potential wins (ties go to the lower list index)
    distance_between = _periodic_distance_squared(
        candidate_position[:, None], candidate_position[None, :], box
    )
    index = jnp.arange(num_slots)
    beaten = (
        valid[None, :]
        & (distance_between <= accretion_radius**2)
        & (
            (candidate_phi[None, :] < candidate_phi[:, None])
            | ((candidate_phi[None, :] == candidate_phi[:, None])
               & (index[None, :] < index[:, None]))
        )
    )
    accepted = valid & ~jnp.any(beaten, axis=1)

    # candidate r takes the r-th free slot; -1 means none is left
    free = jnp.nonzero(sinks.mass == 0, size=num_slots, fill_value=-1)[0]
    rank = jnp.cumsum(accepted) - 1
    slot = jnp.where(accepted, free[jnp.maximum(rank, 0)], -1)
    # out-of-range targets are dropped by the scatters below
    target = jnp.where(slot >= 0, slot, num_slots)

    opening = jnp.zeros(num_slots, dtype=bool).at[target].set(True, mode="drop")
    centre = sinks.position.at[target].set(candidate_position, mode="drop")
    centre = jnp.where(opening[:, None], centre, 0.0)

    # -------------------------------------------------------------
    # ================= ↑ opening new slots ↑ =====================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============== ↓ assigning cells to slots ↓ =================
    # -------------------------------------------------------------

    # Every cell whose centre can be within the accretion radius of a slot lies
    # in this sphere around the slot's cell: the slot is at most sqrt(3)/2
    # cells from its cell centre.
    candidate_offsets = jnp.array(
        _sphere_offsets(config.sink_accretion_radius + math.sqrt(3) / 2)
    )

    # (num_slots, num_offsets, 3); positions are not wrapped, so they stay
    # continuous around the slot
    centre_cell = jnp.floor(centre / dx).astype(jnp.int32)
    cell = centre_cell[:, None, :] + candidate_offsets[None]
    cell_position = (cell + 0.5) * dx
    flat_cell = jnp.ravel_multi_index(
        tuple(cell[..., axis] for axis in range(3)), shape, mode="wrap"
    )
    r_squared = jnp.sum((cell_position - centre[:, None]) ** 2, axis=-1)
    # only opening slots take gas: there is no accretion yet
    inside = opening[:, None] & (r_squared <= accretion_radius**2)

    cell_velocity = jnp.moveaxis(u.reshape(3, num_cells)[:, flat_cell], 0, -1)

    # A cell within the radius of several slots feeds the closest one.
    # TODO: once sinks move under gravity, feed the slot the cell is most
    # strongly bound to instead.
    distance = jnp.where(inside, r_squared, jnp.inf)
    closest = jnp.full(num_cells, jnp.inf).at[flat_cell].min(distance)
    winner = inside & (distance == closest[flat_cell])
    # ties go to the lower slot index
    slot_index = jnp.arange(num_slots)[:, None]
    first_winner = jnp.full(num_cells, num_slots).at[flat_cell].min(
        jnp.where(winner, slot_index, num_slots)
    )
    winner = winner & (first_winner[flat_cell] == slot_index)

    # -------------------------------------------------------------
    # ============== ↑ assigning cells to slots ↑ =================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ===================== ↓ transfer ↓ ==========================
    # -------------------------------------------------------------

    excess_mass = jnp.maximum(rho - rho_thr, 0.0).ravel() * volume
    dm = jnp.where(winner, excess_mass[flat_cell], 0.0)

    # The gas keeps its velocity and specific internal energy, so density and
    # pressure drop by the removed fraction.
    removed = jnp.zeros(num_cells, dtype=m.dtype).at[flat_cell].add(dm).reshape(shape)
    remaining_fraction = 1.0 - removed / m
    state = state.at[density_index].multiply(remaining_fraction)
    state = state.at[pressure_index].multiply(remaining_fraction)

    # The slot sits at the centre of mass, and moves with the momentum, of
    # everything it has taken. At opening the old mass is zero, so the stale
    # position and velocity drop out; occupied slots are left untouched.
    m_old = sinks.mass
    m_new = m_old + jnp.sum(dm, axis=1)
    m_safe = jnp.where(m_new > 0, m_new, 1.0)[:, None]
    grew = (opening & (m_new > 0))[:, None]
    position = jnp.where(
        grew,
        jnp.mod(
            (m_old[:, None] * sinks.position
             + jnp.sum(dm[..., None] * cell_position, axis=1)) / m_safe,
            box,
        ),
        sinks.position,
    )
    velocity = jnp.where(
        grew,
        (m_old[:, None] * sinks.velocity
         + jnp.sum(dm[..., None] * cell_velocity, axis=1)) / m_safe,
        sinks.velocity,
    )

    # -------------------------------------------------------------
    # ===================== ↑ transfer ↑ ==========================
    # -------------------------------------------------------------

    return state, SinkParticles(mass=m_new, position=position, velocity=velocity)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _sink_formation(
    primitive_state: STATE_TYPE,
    sinks: SinkParticles,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
):
    """
    Form and feed sink particles on the (possibly padded) primitive state.

    Computes the gravitational potential, then runs :func:`_form_sinks` on the
    interior cells and, with ghost cells, refills them from the new interior.

    Args:
        primitive_state: The primitive state, padded unless the boundaries are
            enforced by rolling.
        sinks: The sink-particle buffer.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        ``(primitive_state, sinks)`` after the transfer.
    """
    phi = _compute_total_potential(
        primitive_state[registered_variables.density_index],
        config.grid_spacing,
        config,
        params,
        registered_variables,
        params.gravitational_constant,
    )

    if config.boundary_handling == PERIODIC_ROLL:
        return _form_sinks(
            primitive_state, phi, sinks, config, params, registered_variables
        )

    g = config.num_ghost_cells
    interior = (slice(g, -g),) * 3
    state, sinks = _form_sinks(
        primitive_state[(slice(None),) + interior],
        phi[interior],
        sinks,
        config,
        params,
        registered_variables,
    )

    primitive_state = primitive_state.at[(slice(None),) + interior].set(state)

    # update ghost cells after sink formation
    primitive_state = _boundary_handler(
        primitive_state, config, registered_variables, params
    )

    return primitive_state, sinks
