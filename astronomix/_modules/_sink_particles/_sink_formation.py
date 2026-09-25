"""
Sink-particle formation.

After every hydro step, gas is moved from the grid into the slots of the
sink-particle buffer. A cell ``c`` gives

    dm_c = min(0.25 m_c, F_c * max(rho_c - rho_thr, 0) * V)

to the slot whose accretion sphere contains it and to which it is most strongly
bound. ``F_c`` is the product of the enabled transfer checks (converging flow,
Jeans instability, boundedness), each 0 or 1, evaluated on the control volume
of the cell holding that slot: the sphere of radius
``config.sink_accretion_radius`` (in cells) around it. ``rho_thr`` is the
Truelove threshold.

A slot opens at a cell that is the potential minimum of its control volume,
passes every enabled check, lies above the threshold, has no occupied slot
within the accretion radius, and has no other candidate of the same step within
the accretion radius with a lower potential.

The grid is periodic: neighbours are reached by rolling the arrays and slot
distances use the minimum image.
"""

# general
from functools import partial
import math

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

#: Jeans number of the Truelove threshold: four cells per Jeans length.
TRUELOVE_JEANS_NUMBER = 0.25

#: Largest fraction of its mass a cell may give in one step.
MAX_REMOVED_FRACTION = 0.25


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


def _shell_offsets(radius):
    """The one-cell layer just outside :func:`_sphere_offsets`."""
    return [
        offset for offset in _sphere_offsets(radius + 1)
        if sum(o * o for o in offset) > radius**2
    ]


def _shifted(field, offset):
    """``field`` at cell ``c + offset``, for fields whose last three axes are
    spatial."""
    return jnp.roll(field, tuple(-o for o in offset), axis=(-3, -2, -1))


def _unit(axis, sign):
    """The offset of one cell along ``axis`` in direction ``sign``."""
    offset = [0, 0, 0]
    offset[axis] = sign
    return tuple(offset)


def _periodic_distance_squared(a, b, box):
    """Squared minimum-image distance between positions ``a`` and ``b`` (last
    axis 3)."""
    d = a - b
    d = d - box * jnp.round(d / box)
    return jnp.sum(d**2, axis=-1)


def _transfer_checks(m, u, e_th, phi, config):
    """
    Product of the enabled transfer checks, each 1.0 if it passes and 0.0 if
    not.

    Args:
        m: Cell masses, shape (nx, ny, nz).
        u: Gas velocities, shape (3, nx, ny, nz).
        e_th: Cell thermal energies, shape (nx, ny, nz).
        phi: Gravitational potential, shape (nx, ny, nz).
        config: The simulation configuration.

    Returns:
        ``F_c``, shape (nx, ny, nz).
    """
    checks = jnp.ones_like(m)

    # Gas must flow inward along each axis separately; a region squeezed along
    # one axis and escaping along another (a shock) fails.
    if config.sink_converging_flow_check:
        for axis in range(3):
            du = _shifted(u[axis], _unit(axis, 1)) - _shifted(u[axis], _unit(axis, -1))
            checks = checks * (du < 0).astype(m.dtype)

    if config.sink_jeans_check or config.sink_bound_check:
        sphere = _sphere_offsets(config.sink_accretion_radius)

        fields = jnp.stack([
            m,
            m * u[0],
            m * u[1],
            m * u[2],
            0.5 * m * jnp.sum(u**2, axis=0),
            e_th,
            m * phi,
        ])
        sums = sum(_shifted(fields, offset) for offset in sphere)
        mass, px, py, pz, kinetic, thermal, mass_phi = sums

        # The gas escapes the well over the lowest point of its rim, so the
        # potential energy is measured from the lowest potential in the layer
        # of cells just outside the control volume.
        phi_edge = jnp.min(
            jnp.stack([
                _shifted(phi, offset)
                for offset in _shell_offsets(config.sink_accretion_radius)
            ]),
            axis=0,
        )
        e_grav = mass_phi - mass * phi_edge

        # kinetic energy in the centre-of-mass frame of the control volume
        e_kin = kinetic - 0.5 * (px**2 + py**2 + pz**2) / mass

        if config.sink_jeans_check:
            checks = checks * (-e_grav > 2.0 * thermal).astype(m.dtype)
        if config.sink_bound_check:
            checks = checks * (e_grav + thermal + e_kin < 0).astype(m.dtype)

    return checks


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
        e_th = 1.5 * m * c_s2
    else:
        c_s2 = params.gamma * p / rho
        e_th = p * volume / (params.gamma - 1)

    # Truelove threshold, per cell through the local sound speed
    rho_thr = TRUELOVE_JEANS_NUMBER**2 * jnp.pi * c_s2 / (G * dx**2)

    checks = _transfer_checks(m, u, e_th, phi, config)

    # -------------------------------------------------------------
    # ================= ↓ opening new slots ↓ =====================
    # -------------------------------------------------------------

    phi_min = jnp.min(
        jnp.stack([
            _shifted(phi, offset)
            for offset in _sphere_offsets(config.sink_accretion_radius)
        ]),
        axis=0,
    )
    candidate = (phi <= phi_min) & (checks == 1.0) & (rho > rho_thr)

    # At most num_slots slots can open in one step, which bounds the list.
    candidate_cell = jnp.nonzero(candidate.ravel(), size=num_slots, fill_value=-1)[0]
    valid = candidate_cell >= 0
    candidate_cell = jnp.maximum(candidate_cell, 0)
    candidate_position = (
        jnp.stack(jnp.unravel_index(candidate_cell, shape), axis=-1) + 0.5
    ) * dx
    candidate_phi = phi.ravel()[candidate_cell]
    candidate_velocity = u.reshape(3, num_cells)[:, candidate_cell].T

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
    active = occupied | opening
    centre = sinks.position.at[target].set(candidate_position, mode="drop")
    centre = jnp.where(active[:, None], centre, 0.0)
    reference_velocity = sinks.velocity.at[target].set(candidate_velocity, mode="drop")

    # -------------------------------------------------------------
    # ================= ↑ opening new slots ↑ =====================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============== ↓ assigning cells to slots ↓ =================
    # -------------------------------------------------------------

    # Every cell whose centre can be within the accretion radius of a slot lies
    # in this cube around the slot's cell.
    half_width = int(math.floor(config.sink_accretion_radius + 0.5))
    cube = jnp.array([
        (i, j, k)
        for i in range(-half_width, half_width + 1)
        for j in range(-half_width, half_width + 1)
        for k in range(-half_width, half_width + 1)
    ])

    # (num_slots, num_offsets, 3); positions are not wrapped, so they stay
    # continuous around the slot
    centre_cell = jnp.floor(centre / dx).astype(jnp.int32)
    cell = centre_cell[:, None, :] + cube[None]
    cell_position = (cell + 0.5) * dx
    flat_cell = jnp.ravel_multi_index(
        tuple(cell[..., axis] for axis in range(3)), shape, mode="wrap"
    )
    r_squared = jnp.sum((cell_position - centre[:, None]) ** 2, axis=-1)
    inside = active[:, None] & (r_squared <= accretion_radius**2)

    cell_velocity = jnp.moveaxis(u.reshape(3, num_cells)[:, flat_cell], 0, -1)

    # A cell within the radius of several slots feeds the one it is most
    # strongly bound to. Sink gravity is a point mass, softened to half a cell
    # so the cell holding the sink stays finite.
    binding = (
        -G * sinks.mass[:, None] / jnp.sqrt(jnp.maximum(r_squared, (0.5 * dx) ** 2))
        + 0.5 * jnp.sum((cell_velocity - reference_velocity[:, None]) ** 2, axis=-1)
    )
    binding = jnp.where(inside, binding, jnp.inf)
    best_binding = jnp.full(num_cells, jnp.inf).at[flat_cell].min(binding)
    winner = inside & (binding == best_binding[flat_cell])
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

    slot_checks = checks.ravel()[
        jnp.ravel_multi_index(
            tuple(centre_cell[:, axis] for axis in range(3)), shape, mode="wrap"
        )
    ]
    dm = jnp.minimum(
        MAX_REMOVED_FRACTION * m.ravel()[flat_cell],
        slot_checks[:, None]
        * jnp.maximum(rho - rho_thr, 0.0).ravel()[flat_cell] * volume,
    )
    dm = jnp.where(winner, dm, 0.0)

    # The gas keeps its velocity and specific internal energy, so density and
    # pressure drop by the removed fraction.
    removed = jnp.zeros(num_cells, dtype=m.dtype).at[flat_cell].add(dm).reshape(shape)
    remaining_fraction = 1.0 - removed / m
    state = state.at[density_index].multiply(remaining_fraction)
    state = state.at[pressure_index].multiply(remaining_fraction)

    # The slot sits at the centre of mass, and moves with the momentum, of
    # everything it has taken. At opening the old mass is zero, so the stale
    # position and velocity drop out.
    m_old = sinks.mass
    m_new = m_old + jnp.sum(dm, axis=1)
    m_safe = jnp.where(m_new > 0, m_new, 1.0)[:, None]
    grew = (m_new > 0)[:, None]
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
    interior cells.

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

    return primitive_state.at[(slice(None),) + interior].set(state), sinks
