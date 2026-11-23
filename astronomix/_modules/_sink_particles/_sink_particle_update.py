"""
The update of the sink particles at the end of a time step, following
Federrath et al. (2010), ApJ 713, 269, with smooth formation and accretion:

1. create candidates where the gas passes the relaxed creation checks;
2. raise the activation of every occupied slot to its creation weight;
3. and 4. accrete gas onto every occupied slot, and update the gas and the
   slots;
5. relax the velocity of every slot toward that of the gas around it by the
   fraction 1 − a, so that a candidate moves with its gas;
6. free the slots of candidates that have neither activation nor mass and
   whose gas fails the relaxed creation checks.

Steps 1 and 6 are hard; the others change the gas and the slots smoothly.
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
    SlotChecks,
    _activate,
    _cell_quantities,
    _create_candidates,
    _near_slots_mask,
    _passes_relaxed_checks,
    _slot_checks,
    _slot_neighbourhood,
    _slot_separation_squared,
)
from astronomix._modules._sink_particles._sink_particle_helpers import _interior_mask
from astronomix._modules._sink_particles._sink_particle_smoothing import _safe_divide


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
    relative to the gas around it,

        v_gas = Σ W ρ v / Σ W ρ,   v ← v + (a − 1)(v − v_gas),

    with the TSC weights W of the dynamics. The momentum change m Δv is taken
    from the gas of the same cells, in proportion to W ρ, at fixed pressure. A
    candidate with a = 0 moves with the gas; a sink with a = 1 is unaffected.

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

    is_moved = sink_particles.occupied & (total_weighted_density > 0.0)
    velocity_change = jnp.where(
        is_moved[:, None],
        (sink_particles.activation[:, None] - 1.0)
        * (sink_particles.velocity - gas_velocity),
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
    that fails a relaxed creation check: (b) another occupied slot lies
    within r_acc, or (c) to (g) fail at the slot, with the quantities of
    step 2. Its mass and momentum go back to the gas with the TSC weights of
    the dynamics; for an ideal gas the pressure and dual-energy g grow with
    the density, which keeps c_s² unchanged. A freed slot is zero in every
    field, like a slot never filled.

    Args:
        primitive_state: The (padded) primitive state.
        sink_particles: The sink particles.
        checks: The quantities of the creation checks at every slot, from
            step 2.
        slot_separation_squared: The squared distances between slots, from
            step 2.
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

    is_other_slot_near = jnp.any(
        sink_particles.occupied[None, :]
        & ~jnp.eye(num_slots, dtype=bool)
        & (slot_separation_squared <= accretion_radius**2),
        axis=-1,
    )
    is_freed = (
        sink_particles.occupied
        & (sink_particles.activation < sink_particle_config.recycling_activation)
        & (
            sink_particles.mass
            < sink_particle_config.recycling_mass * checks.density_threshold * cell_volume
        )
        & (is_other_slot_near | ~_passes_relaxed_checks(checks, config))
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
    mass above the density threshold farther than r_acc from every slot.

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
            jnp.maximum(0.0, cells.density - cells.density_threshold),
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
        "occupied sink slots: {}, gas mass above the density threshold farther "
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
    Create, activate and recycle candidates, and let every occupied slot
    accrete gas, after the hydro update and the leapfrog of the sinks.

    The checks read one potential, of the gas and of the slots' masses,
    solved before any change, and the quantities of step 2 are evaluated
    once, at the slots' positions before accretion.

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

    unrecycled_sink_particles = sink_particles
    primitive_state, sink_particles, is_freed = _recycle_slots(
        primitive_state,
        sink_particles,
        checks,
        slot_separation_squared,
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
