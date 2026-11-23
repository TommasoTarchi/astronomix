"""
The update of the sink particles at the end of a time step, following
Federrath et al. (2010), ApJ 713, 269, with smooth formation and accretion:

1. create candidates where the gas passes the relaxed creation checks;
2. raise the activation of every occupied slot to its creation weight;
3. and 4. accrete gas onto every occupied slot, and update the gas and the
   slots;
5. relax the velocity of every slot toward the centre-of-mass velocity of the
   slot and the gas around it by the fraction 1 − a, so that a candidate
   moves with its gas;
6. free the slots of candidates that have neither activation nor mass and
   whose gas fails the relaxed creation checks, or that lie within r_acc of a
   slot that ranks above them;
7. move every slot that holds less than m_min and is not fully active toward
   the soft-minimum position of the potential around it, so that a candidate
   activates where the gas puts the minimum at that time, whatever the step
   or the cell in which it was created.

Steps 1 and 6 are hard; the others change the gas and the slots smoothly.
The move of step 7 is made before step 6 frees the slots, so that the
proximity check of step 6 sees the moved positions.
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import IDEAL_GAS, STATE_TYPE

# astronomix containers
from astronomix.data_classes.simulation_state_struct import SinkParticles
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._sink_particles._sink_particle_accretion import _accrete
from astronomix._modules._sink_particles._sink_particle_dynamics import _tsc_weights
from astronomix._modules._sink_particles._sink_particle_formation import (
    CellQuantities,
    SlotChecks,
    SlotNeighbourhood,
    _activate,
    _cell_quantities,
    _create_candidates,
    _near_slots_mask,
    _passes_relaxed_checks,
    _read_at_positions,
    _slot_checks,
    _slot_neighbourhood,
    _slot_separation_squared,
)
from astronomix._modules._sink_particles._sink_particle_helpers import (
    _accreted_cell_density,
    _interior_mask,
    _periodic_box,
)
from astronomix._modules._sink_particles._sink_particle_smoothing import (
    _safe_divide,
    _smooth_step,
    _soft_minimum_weights,
)


def _velocity_rows(registered_variables: RegisteredVariables) -> jax.Array:
    """The rows of the three velocity components in the primitive state."""
    return jnp.array(
        [
            registered_variables.velocity_index.x,
            registered_variables.velocity_index.y,
            registered_variables.velocity_index.z,
        ]
    )


def _move_candidates_with_gas(
    primitive_state: STATE_TYPE,
    sink_particles: SinkParticles,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> tuple[STATE_TYPE, SinkParticles]:
    """
    Step 5: keep only the fraction a of each occupied slot's velocity
    relative to the centre-of-mass velocity of the slot and the gas around
    it,

        M_gas = Σ W ρ ΔV,   v_gas = Σ W ρ v / Σ W ρ,
        v_cm = (m v + M_gas v_gas) / (m + M_gas),   v ← v + (a − 1)(v − v_cm),

    with the TSC weights W of the dynamics. The momentum change m Δv is taken
    from the gas of the same cells, in proportion to W ρ, at fixed pressure,
    which moves the gas velocity by at most (1 − a) |v − v_gas|, also for a
    slot much heavier than its gas. A candidate with a = 0 and m ≪ M_gas
    moves with the gas; a sink with a = 1 is unaffected.

    Args:
        primitive_state: The (padded) primitive state.
        sink_particles: The sink particles.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        ``(primitive_state, sink_particles)`` with the new velocities.
    """
    cell_volume = config.grid_spacing**3
    velocity_rows = _velocity_rows(registered_variables)
    density = primitive_state[registered_variables.density_index]
    grid_shape = density.shape

    cell_indices, weights = _tsc_weights(sink_particles.position, grid_shape, config)
    weighted_density = weights * density[cell_indices]
    total_weighted_density = jnp.sum(weighted_density, axis=-1)
    gas_velocity_in_cells = jnp.stack(
        [primitive_state[row][cell_indices] for row in velocity_rows],
        axis=-1,
    )
    gas_velocity = _safe_divide(
        jnp.sum(weighted_density[..., None] * gas_velocity_in_cells, axis=1),
        total_weighted_density[:, None],
    )
    gas_mass = total_weighted_density * cell_volume
    center_of_mass_velocity = _safe_divide(
        sink_particles.mass[:, None] * sink_particles.velocity
        + gas_mass[:, None] * gas_velocity,
        (sink_particles.mass + gas_mass)[:, None],
    )

    is_moved = sink_particles.occupied & (total_weighted_density > 0.0)
    velocity_change = jnp.where(
        is_moved[:, None],
        (sink_particles.activation[:, None] - 1.0)
        * (sink_particles.velocity - center_of_mass_velocity),
        0.0,
    )

    # The gas velocity changes by −m Δv W / (Σ W ρ ΔV) in each cell, which
    # removes m Δv W ρ / Σ W ρ of momentum from it.
    gas_velocity_change = jnp.zeros((3,) + grid_shape, dtype=density.dtype)
    for axis in range(3):
        gas_velocity_change = gas_velocity_change.at[(axis,) + cell_indices].add(
            -_safe_divide(
                sink_particles.mass[:, None] * velocity_change[:, axis, None] * weights,
                total_weighted_density[:, None] * cell_volume,
            )
        )
    primitive_state = primitive_state.at[velocity_rows].add(gas_velocity_change)

    return primitive_state, sink_particles._replace(
        velocity=sink_particles.velocity + velocity_change
    )


def _soft_minimum_position(
    position: jax.Array,
    neighbourhood: SlotNeighbourhood,
    cells: CellQuantities,
    config: SimulationConfig,
) -> jax.Array:
    """
    The soft-minimum position of the potential around each position: the
    cell centres within r_acc averaged with the weights of the soft minimum
    of the potential-minimum check,

        p = x + Σ K r e^(−φ/T) / Σ K e^(−φ/T),   T = ω c_s²,

    with r the separation from x to the cell centre and c_s² read at x. At
    ω = 0 it is the centre of the cell of lowest φ.

    Args:
        position: The positions x, shape (num_positions, 3).
        neighbourhood: The cells around each position.
        cells: The cell quantities.
        config: The simulation configuration.

    Returns:
        p, shape (num_positions, 3), not wrapped back along periodic axes; x
        where no cell is within r_acc.
    """
    weights = _soft_minimum_weights(
        cells.potential[neighbourhood.cell_indices],
        neighbourhood.radial_weight,
        config.sink_particle_config.check_width,
        _read_at_positions(cells.sound_speed_squared, neighbourhood),
    )
    return position + jnp.sum(weights[..., None] * neighbourhood.separation, axis=1)


def _relocate_candidates(
    sink_particles: SinkParticles,
    minimum_position: jax.Array,
    checks: SlotChecks,
    config: SimulationConfig,
) -> SinkParticles:
    """
    Step 7: move every occupied slot toward the soft-minimum position p of
    the potential around it (``_soft_minimum_position``),

        x ← x + λ (p − x),   λ = (1 − a) [1 − step_{m_min}(m / (ρ_thr ΔV))],

    with m_min = ``recycling_mass`` and ρ_thr read at the slot. λ is 1 for a
    massless candidate and 0 for a sink and for a slot that holds m_min ρ_thr
    ΔV or more; a slot with no cell to read ρ_thr from does not move either.
    The velocity is unchanged, and the position is wrapped back along
    periodic axes.

    A massless candidate carries neither mass nor momentum, so moving it
    changes nothing in the gas. Repeated at every step, the move settles it
    at the bottom of the smoothed potential well it is in, wherever in the
    well it was created. The derivative of the move is kept, so that the
    position at activation follows the gas.

    Args:
        sink_particles: The sink particles.
        minimum_position: p of every slot, shape (num_slots, 3).
        checks: The quantities of the creation checks at every slot.
        config: The simulation configuration.

    Returns:
        The sink particles with the new positions.
    """
    is_periodic, box_length = _periodic_box(config)
    scaled_mass = _safe_divide(
        sink_particles.mass,
        checks.density_threshold * config.grid_spacing**3,
    )
    fraction = jnp.where(
        sink_particles.occupied & checks.has_cells,
        (1.0 - sink_particles.activation)
        * (1.0 - _smooth_step(scaled_mass, config.sink_particle_config.recycling_mass)),
        0.0,
    )
    displacement = minimum_position - sink_particles.position
    displacement = jnp.where(
        is_periodic,
        displacement - box_length * jnp.round(displacement / box_length),
        displacement,
    )
    position = sink_particles.position + fraction[:, None] * displacement
    return sink_particles._replace(
        position=jnp.where(is_periodic, jnp.mod(position, box_length), position)
    )


def _recycle_slots(
    primitive_state: STATE_TYPE,
    sink_particles: SinkParticles,
    checks: SlotChecks,
    slot_separation_squared: jax.Array,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> tuple[STATE_TYPE, SinkParticles, jax.Array]:
    """
    Step 6: free the slot of every candidate with an activation below
    ``recycling_activation`` and a mass below ``recycling_mass`` ρ_thr ΔV
    that fails a relaxed creation check: (b) an occupied slot that ranks
    above it lies within r_acc, or (c) to (g) fail at the slot, with the
    quantities of step 2. Slot j ranks above slot i when (a_j, m_j, −j) >
    (a_i, m_i, −i) in lexicographic order, so of two candidates within r_acc
    of each other only the lower-ranked one is freed. Its mass and momentum
    go back to the gas with the TSC weights of the dynamics; for an ideal gas
    the pressure and dual-energy g grow with the density, which keeps c_s²
    unchanged. A freed slot is zero in every field, like a slot never filled.

    Args:
        primitive_state: The (padded) primitive state.
        sink_particles: The sink particles.
        checks: The quantities of the creation checks at every slot, from
            step 2.
        slot_separation_squared: The squared distances between the slots at
            their current positions.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        ``(primitive_state, sink_particles, is_freed)``, the last of shape
        (num_slots,).
    """
    sink_particle_config = config.sink_particle_config
    cell_volume = config.grid_spacing**3
    accretion_radius = sink_particle_config.accretion_radius_in_cells * config.grid_spacing
    num_slots = sink_particles.mass.shape[0]

    # ranks_above[i, j]: slot j ranks above slot i.
    activation = sink_particles.activation
    mass = sink_particles.mass
    slot = jnp.arange(num_slots)
    ranks_above = (activation[None, :] > activation[:, None]) | (
        (activation[None, :] == activation[:, None])
        & (
            (mass[None, :] > mass[:, None])
            | ((mass[None, :] == mass[:, None]) & (slot[None, :] < slot[:, None]))
        )
    )
    is_higher_slot_near = jnp.any(
        sink_particles.occupied[None, :]
        & ranks_above
        & (slot_separation_squared <= accretion_radius**2),
        axis=-1,
    )
    is_freed = (
        sink_particles.occupied
        & (activation < sink_particle_config.recycling_activation)
        & (mass < sink_particle_config.recycling_mass * checks.density_threshold * cell_volume)
        & (is_higher_slot_near | ~_passes_relaxed_checks(checks, config))
    )

    # -------------------------------------------------------------
    # ============ ↓ Mass and momentum back to the gas ↓ ==========
    # -------------------------------------------------------------

    velocity_rows = _velocity_rows(registered_variables)
    density = primitive_state[registered_variables.density_index]
    grid_shape = density.shape
    cell_indices, weights = _tsc_weights(sink_particles.position, grid_shape, config)
    returned_mass = jnp.where(is_freed, sink_particles.mass, 0.0)[:, None] * weights

    added_density = (
        jnp.zeros(grid_shape, dtype=density.dtype).at[cell_indices].add(returned_mass)
        / cell_volume
    )
    new_density = density + added_density
    velocity = primitive_state[velocity_rows]
    new_velocity = velocity
    for axis in range(3):
        added_momentum = (
            jnp.zeros(grid_shape, dtype=density.dtype)
            .at[cell_indices]
            .add(returned_mass * sink_particles.velocity[:, axis, None])
            / cell_volume
        )
        new_velocity = new_velocity.at[axis].add(
            (added_momentum - added_density * velocity[axis]) / new_density
        )

    primitive_state = primitive_state.at[registered_variables.density_index].set(
        new_density
    )
    primitive_state = primitive_state.at[velocity_rows].set(new_velocity)
    if config.equation_of_state == IDEAL_GAS:
        # Only the cells that got gas back are scaled, so that the others keep
        # their pressure to the last bit.
        has_gained_gas = new_density != density
        energy_rows = [registered_variables.pressure_index]
        if registered_variables.internal_energy_active:
            energy_rows.append(registered_variables.internal_energy_index)
        for row in energy_rows:
            primitive_state = primitive_state.at[row].set(
                jnp.where(
                    has_gained_gas,
                    primitive_state[row] * (new_density / density),
                    primitive_state[row],
                )
            )

    # -------------------------------------------------------------
    # ============ ↑ Mass and momentum back to the gas ↑ ==========
    # -------------------------------------------------------------

    sink_particles = jax.tree.map(
        lambda field: jnp.where(
            is_freed.reshape((num_slots,) + (1,) * (field.ndim - 1)),
            jnp.zeros_like(field),
            field,
        ),
        sink_particles,
    )
    return primitive_state, sink_particles, is_freed


def _print_diagnostics(
    is_created: jax.Array,
    is_freed: jax.Array,
    freed_sink_particles: SinkParticles,
    sink_particles: SinkParticles,
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> None:
    """
    Print the candidates created in this step with their largest activation
    and mass after it, the candidates freed with their largest activation and
    mass before they were freed, the number of occupied slots, and the gas
    mass that accretion would take farther than r_acc from every slot.

    Args:
        is_created: Whether each slot got a new candidate in this step.
        is_freed: Whether each slot was freed in this step.
        freed_sink_particles: The sink particles just before recycling.
        sink_particles: The sink particles at the end of the step.
        primitive_state: The (padded) primitive state at the end of the step.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
    """
    cells = _cell_quantities(
        primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )
    grid_shape = cells.density.shape
    is_near_slot = _near_slots_mask(
        _slot_neighbourhood(sink_particles.position, grid_shape, config),
        sink_particles.occupied,
        grid_shape,
        config,
    )
    uncovered_mass = jnp.sum(
        jnp.where(
            _interior_mask(cells.density, config) & ~is_near_slot,
            jnp.maximum(
                0.0,
                cells.density
                - _accreted_cell_density(cells.density_threshold, config, params),
            ),
            0.0,
        )
    ) * config.grid_spacing**3

    jax.lax.cond(
        jnp.any(is_created),
        lambda: jax.debug.print(
            "sink candidates created: {}, largest activation {} and mass {} "
            "after their first update",
            jnp.sum(is_created),
            jnp.max(jnp.where(is_created, sink_particles.activation, 0.0)),
            jnp.max(jnp.where(is_created, sink_particles.mass, 0.0)),
        ),
        lambda: None,
    )
    jax.lax.cond(
        jnp.any(is_freed),
        lambda: jax.debug.print(
            "sink candidates recycled: {}, largest activation {} and mass {}",
            jnp.sum(is_freed),
            jnp.max(jnp.where(is_freed, freed_sink_particles.activation, 0.0)),
            jnp.max(jnp.where(is_freed, freed_sink_particles.mass, 0.0)),
        ),
        lambda: None,
    )
    jax.debug.print(
        "occupied sink slots: {}, gas mass that accretion would take farther "
        "than r_acc from every slot: {}",
        jnp.sum(sink_particles.occupied),
        uncovered_mass,
    )


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _update_sink_particles(
    primitive_state: STATE_TYPE,
    sink_particles: SinkParticles,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> tuple[STATE_TYPE, SinkParticles]:
    """
    Create, activate and recycle candidates, let every occupied slot accrete
    gas, and move the light candidates toward the potential minimum, after
    the hydro update and the leapfrog of the sinks.

    The checks and the move read one potential, of the gas and of the slots'
    masses, solved before any change, and the quantities of step 2 are
    evaluated once, at the slots' positions before accretion.

    Args:
        primitive_state: The (padded) primitive state after the hydro update.
        sink_particles: The sink particles after the leapfrog.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        ``(primitive_state, sink_particles)`` after the update.
    """
    grid_shape = primitive_state.shape[1:]
    was_occupied = sink_particles.occupied

    cells = _cell_quantities(
        primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )

    sink_particles = _create_candidates(sink_particles, cells, config)

    neighbourhood = _slot_neighbourhood(sink_particles.position, grid_shape, config)
    checks = _slot_checks(neighbourhood, sink_particles.mass, cells, config)
    minimum_position = _soft_minimum_position(
        sink_particles.position,
        neighbourhood,
        cells,
        config,
    )
    slot_separation_squared = _slot_separation_squared(sink_particles.position, config)
    sink_particles = _activate(sink_particles, checks, slot_separation_squared, config)

    primitive_state, sink_particles = _accrete(
        primitive_state,
        sink_particles,
        neighbourhood,
        cells,
        checks.density_threshold,
        config,
        params,
        registered_variables,
    )
    primitive_state, sink_particles = _move_candidates_with_gas(
        primitive_state,
        sink_particles,
        config,
        registered_variables,
    )

    # Step 7 before step 6, so that the proximity check of recycling sees the
    # moved positions.
    sink_particles = _relocate_candidates(sink_particles, minimum_position, checks, config)

    unrecycled_sink_particles = sink_particles
    primitive_state, sink_particles, is_freed = _recycle_slots(
        primitive_state,
        sink_particles,
        checks,
        _slot_separation_squared(sink_particles.position, config),
        config,
        registered_variables,
    )

    if config.sink_particle_config.print_diagnostics:
        _print_diagnostics(
            unrecycled_sink_particles.occupied & ~was_occupied,
            is_freed,
            unrecycled_sink_particles,
            sink_particles,
            primitive_state,
            config,
            params,
            registered_variables,
        )

    return primitive_state, sink_particles
