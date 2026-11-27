"""
Gas accretion onto sink particles following Federrath et al. (2010), ApJ 713,
269, with smooth checks (``_sink_particle_smoothing.py``).

Once per time step, after sink formation, every active slot takes gas above
the density threshold from the cells within its accretion radius (Section
2.3). A cell can give at most its excess e = (ρ − ρ_res) ΔV (Eq. 13). For a
polytropic gas ρ_res drops with the density, and the excess is
e = (ρ − ρ*) ΔV instead, with ρ* the density at which the gas is at its own
threshold. The gas must be bound to the sink and move toward it, except in
the cell that contains the sink. Each check is a smooth step, and the weight
of a sink-cell pair is

    g = a K [1 − (1 − b n)(1 − h)],

with a the activation of the sink, K the radial weight, b the bound check,
n the radial-velocity check and h the cell weight: the bracket is a smooth
OR of the motion checks and of the sink lying in the cell. Everything a sink
takes carries the factor a. The fraction F = 1 − Π (1 − g) of the excess is
taken from the cell, and shared between the sinks with a softmax of
sharpness β that favours the sink the gas is most bound to. Mass and linear
momentum are conserved (Appendix B).

After accretion, the velocity of every forming seed is tied to that of the
gas of its cells (``_couple_to_gas``), which also conserves momentum.

With every width at its hard limit (ω = ℓ = ε = 0, β = ∞), a cell passes or
fails each check as in the hard scheme, a sink of activation 1 takes the
whole excess of a cell that passes, and a cell within reach of several sinks
goes to the one it is most bound to, ties going to the lowest slot.

Only gas in interior cells is accreted. The cells around a sink wrap around
periodic axes and end at the edge of the box on non-periodic axes. Reflective
walls are treated like the other edges: mirrored gas is not counted.

The work is done per sink rather than per cell: each sink gathers the few cells
around it, so the cost scales with the number of sinks, not with the grid.
"""

# general
import math
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    IDEAL_GAS,
    POLYTROPIC,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_state_struct import SinkParticles
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._sink_particles._sink_particle_dynamics import _couple_to_gas
from astronomix._modules._sink_particles._sink_particle_formation import (
    CellQuantities,
    SlotNeighbourhood,
    _gas_quantities,
    _slot_neighbourhood,
)
from astronomix._modules._sink_particles._sink_particle_helpers import (
    _periodic_box,
    _polytropic_resolved_density,
)
from astronomix._modules._sink_particles._sink_particle_smoothing import (
    _point_mass_potential,
    _safe_divide,
    _smooth_step,
)

#: The largest radial velocity, in units of c_s, of gas that counts as moving
#: toward a sink. FLASH accepts gas up to this tolerance, since gas moving with
#: the sink has v_r = 0 only up to rounding.
RADIAL_VELOCITY_TOLERANCE = 1e-5


def _accreted_cell_density(
    density_threshold: jax.Array,
    config: SimulationConfig,
    params: SimulationParams,
) -> jax.Array:
    """
    The density a cell is left at when its whole excess is taken.

    For an ideal or isothermal gas c_s² does not change with the accretion,
    so this is the threshold. For a polytropic gas c_s² drops with the
    density, and the threshold with it, so a cell left at the threshold of
    its density before accretion would still be above its threshold; it is
    left at ρ* instead, the density at which it is at its own threshold.

    Args:
        density_threshold: The density threshold of the cells.
        config: The simulation configuration.
        params: The simulation parameters.

    Returns:
        The density, with the shape of ``density_threshold``.
    """
    if config.equation_of_state == POLYTROPIC:
        return jnp.full_like(
            density_threshold,
            _polytropic_resolved_density(config, params),
        )
    return density_threshold


def _pair_weights(
    sink_particles: SinkParticles,
    neighbourhood: SlotNeighbourhood,
    cells: CellQuantities,
    config: SimulationConfig,
    params: SimulationParams,
) -> tuple[jax.Array, jax.Array]:
    """
    The weight g of every slot-cell pair, and the specific energy η of the
    gas of the cell relative to the slot, in units of c_s².

    η has the bound-check potential of FLASH: the gas within r_acc as a
    uniform sphere of density ρ_res, plus the slot as a point mass, both
    measured from the edge of the accretion sphere,

        η = [−G (2π/3) ρ_res (r_acc² − r²) + ψ(r) + ½ |v − v_n|²] / c_s².

    The radial velocity is v_r = r · (v − v_n) / √(r² + λ²), with λ = ℓ Δx; at
    λ = 0 and r = 0 the denominator is r_acc, so that v_r = 0 there.

    Args:
        sink_particles: The sink particles.
        neighbourhood: The cells around every slot.
        cells: The cell quantities.
        config: The simulation configuration.
        params: The simulation parameters.

    Returns:
        ``(weight, energy)``, each of shape (num_slots, num_offsets).
    """
    sink_particle_config = config.sink_particle_config
    check_width = sink_particle_config.check_width
    softening_length = sink_particle_config.spatial_width_in_cells * config.grid_spacing
    accretion_radius = sink_particle_config.accretion_radius_in_cells * config.grid_spacing
    gravitational_constant = params.gravitational_constant

    cell_indices = neighbourhood.cell_indices
    distance_squared = neighbourhood.distance_squared
    sound_speed_squared = cells.sound_speed_squared[cell_indices]
    relative_velocity = cells.velocity[cell_indices] - sink_particles.velocity[:, None, :]

    energy = (
        -gravitational_constant
        * (2.0 * jnp.pi / 3.0)
        * cells.density_threshold[cell_indices]
        * (accretion_radius**2 - distance_squared)
        + _point_mass_potential(
            distance_squared,
            sink_particles.mass[:, None],
            accretion_radius,
            softening_length,
            gravitational_constant,
        )
        + 0.5 * jnp.sum(relative_velocity**2, axis=-1)
    ) / sound_speed_squared

    if softening_length == 0.0:
        is_positive = distance_squared > 0.0
        radial_distance = jnp.where(
            is_positive,
            jnp.sqrt(jnp.where(is_positive, distance_squared, 1.0)),
            accretion_radius,
        )
    else:
        radial_distance = jnp.sqrt(distance_squared + softening_length**2)
    radial_velocity = (
        jnp.sum(neighbourhood.separation * relative_velocity, axis=-1) / radial_distance
    )

    bound = _smooth_step(-energy, check_width)
    moving_inward = 1.0 - _smooth_step(
        radial_velocity / jnp.sqrt(sound_speed_squared) - RADIAL_VELOCITY_TOLERANCE,
        check_width,
    )
    weight = (
        sink_particles.activation[:, None]
        * neighbourhood.radial_weight
        * (1.0 - (1.0 - bound * moving_inward) * (1.0 - neighbourhood.cell_weight))
    )
    return weight, energy


def _taken_fraction(
    weight: jax.Array,
    cell_indices: tuple,
    grid_shape: tuple,
) -> jax.Array:
    """
    The fraction F = 1 − Π_i (1 − g_i) of a cell's excess that is taken, at the
    cell of every slot-cell pair.

    The product over the slots is a sum of logarithms scattered onto the
    grid; pairs with g = 1 are counted apart, since they make the product 0.
    F is −expm1 of that sum, which keeps its relative precision when every g
    is small.

    Args:
        weight: The weight g of every pair, shape (num_slots, num_offsets).
        cell_indices: The cell of every pair.
        grid_shape: The shape of a (padded) field.

    Returns:
        F, shape (num_slots, num_offsets).
    """
    is_saturated = weight >= 1.0
    log_remaining = jnp.where(
        is_saturated,
        0.0,
        jnp.log1p(-jnp.where(is_saturated, 0.0, weight)),
    )
    num_saturated = jnp.zeros(grid_shape, dtype=jnp.int32).at[cell_indices].add(
        is_saturated.astype(jnp.int32)
    )
    total_log_remaining = jnp.zeros(grid_shape, dtype=weight.dtype).at[
        cell_indices
    ].add(log_remaining)
    return jnp.where(
        num_saturated[cell_indices] > 0,
        1.0,
        -jnp.expm1(total_log_remaining[cell_indices]),
    )


def _shares(
    weight: jax.Array,
    energy: jax.Array,
    cell_indices: tuple,
    grid_shape: tuple,
    config: SimulationConfig,
) -> jax.Array:
    """
    The share of every slot in the gas taken from a cell: the softmax
    g e^(−β η) / Σ g e^(−β η) over the slots that reach the cell.

    The softmax runs over the pairs with g > 0 only, with the largest of their
    exponents in each cell subtracted (its derivative stopped), so that a
    deeply bound slot with g = 0 cannot make the terms of the others
    underflow. At β = ∞ the slot with g > 0 and the lowest η takes the whole
    share, ties going to the lowest slot.

    Args:
        weight: The weight g of every pair, shape (num_slots, num_offsets).
        energy: The specific energy η of every pair.
        cell_indices: The cell of every pair.
        grid_shape: The shape of a (padded) field.
        config: The simulation configuration.

    Returns:
        The shares, shape (num_slots, num_offsets); 0 in cells that no slot
        reaches with g > 0.
    """
    sharing_sharpness = config.sink_particle_config.sharing_sharpness
    can_take = weight > 0.0

    if math.isinf(sharing_sharpness):
        candidate_energy = jnp.where(can_take, energy, jnp.inf)
        lowest_energy = jnp.full(grid_shape, jnp.inf, dtype=energy.dtype).at[
            cell_indices
        ].min(candidate_energy)
        is_lowest = can_take & (candidate_energy == lowest_energy[cell_indices])
        num_slots = weight.shape[0]
        slot = jnp.broadcast_to(jnp.arange(num_slots)[:, None], weight.shape)
        lowest_slot = jnp.full(grid_shape, num_slots).at[cell_indices].min(
            jnp.where(is_lowest, slot, num_slots)
        )
        return (is_lowest & (slot == lowest_slot[cell_indices])).astype(weight.dtype)

    exponent = jnp.where(can_take, -sharing_sharpness * energy, -jnp.inf)
    largest_exponent = jax.lax.stop_gradient(
        jnp.full(grid_shape, -jnp.inf, dtype=energy.dtype).at[cell_indices].max(exponent)
    )
    term = jnp.where(
        can_take,
        weight
        * jnp.exp(
            jnp.where(can_take, exponent - largest_exponent[cell_indices], 0.0)
        ),
        0.0,
    )
    normalization = jnp.zeros(grid_shape, dtype=weight.dtype).at[cell_indices].add(term)
    return _safe_divide(term, normalization[cell_indices])


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _accrete_gas(
    primitive_state: STATE_TYPE,
    sink_particles: SinkParticles,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> tuple[STATE_TYPE, SinkParticles]:
    """
    Move gas above the density threshold within reach of each active slot
    onto it, and update the gas and the slots.

    Slot i takes ΔM = F e g e^(−β η) / Σ_j g_j e^(−β η_j) from a cell. The
    cell's density drops by Σ_i ΔM / ΔV; for an ideal gas its pressure and
    dual-energy g drop with it, which keeps c_s² unchanged, and its velocity
    is unchanged. The slot's mass, position and velocity become (Appendix B)

        M' = M + Σ ΔM,
        x' = x + Σ ΔM r / (M' + ε ρ_res ΔV),
        v' = v + Σ ΔM (v_gas − v) / M',

    which conserves mass and momentum, with ρ_res read at the slot with the
    cell weights. The mass ε ρ_res ΔV keeps a sink with almost no mass from
    jumping to the centre of mass of the first gas it accretes. A slot that
    takes nothing keeps its position and velocity, also when its mass is 0.
    The position is wrapped back along periodic axes. Then the velocity of
    every forming seed is tied to that of its gas (``_couple_to_gas``).

    Args:
        primitive_state: The (padded) primitive state.
        sink_particles: The sink particles, including the new massless ones.
            Empty slots take no gas.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        ``(primitive_state, sink_particles)`` after accretion.
    """
    cell_volume = config.grid_spacing**3

    cells = _gas_quantities(primitive_state, config, params, registered_variables)
    density = cells.density
    grid_shape = density.shape
    neighbourhood = _slot_neighbourhood(sink_particles.position, grid_shape, config)
    cell_indices = neighbourhood.cell_indices

    # -------------------------------------------------------------
    # ================== ↓ Accreted masses ↓ ======================
    # -------------------------------------------------------------

    weight, energy = _pair_weights(
        sink_particles,
        neighbourhood,
        cells,
        config,
        params,
    )
    weight = jnp.where(sink_particles.activation[:, None] > 0.0, weight, 0.0)

    excess = (
        jnp.maximum(
            0.0,
            cells.density[cell_indices]
            - _accreted_cell_density(cells.density_threshold, config, params)[
                cell_indices
            ],
        )
        * cell_volume
    )
    accreted_mass = (
        _taken_fraction(weight, cell_indices, grid_shape)
        * excess
        * _shares(weight, energy, cell_indices, grid_shape, config)
    )

    # -------------------------------------------------------------
    # ================== ↑ Accreted masses ↑ ======================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ==================== ↓ Gas update ↓ =========================
    # -------------------------------------------------------------

    new_density = density - (
        jnp.zeros(grid_shape, dtype=density.dtype).at[cell_indices].add(accreted_mass)
        / cell_volume
    )
    primitive_state = primitive_state.at[registered_variables.density_index].set(
        new_density
    )
    if config.equation_of_state == IDEAL_GAS:
        # Only the cells that lost gas are scaled, so that the others keep
        # their pressure to the last bit.
        has_lost_gas = new_density != density
        energy_rows = [registered_variables.pressure_index]
        if registered_variables.internal_energy_active:
            energy_rows.append(registered_variables.internal_energy_index)
        for row in energy_rows:
            primitive_state = primitive_state.at[row].set(
                jnp.where(
                    has_lost_gas,
                    primitive_state[row] * (new_density / density),
                    primitive_state[row],
                )
            )

    # -------------------------------------------------------------
    # ==================== ↑ Gas update ↑ =========================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ==================== ↓ Sink update ↓ ========================
    # -------------------------------------------------------------

    slot_density_threshold = jnp.sum(
        neighbourhood.cell_weight * cells.density_threshold[cell_indices],
        axis=-1,
    )
    position_update_mass = (
        config.sink_particle_config.position_update_mass
        * slot_density_threshold
        * cell_volume
    )
    new_mass = sink_particles.mass + jnp.sum(accreted_mass, axis=-1)
    position_shift = _safe_divide(
        jnp.sum(accreted_mass[..., None] * neighbourhood.separation, axis=1),
        (new_mass + position_update_mass)[:, None],
    )
    velocity_shift = _safe_divide(
        jnp.sum(
            accreted_mass[..., None]
            * (cells.velocity[cell_indices] - sink_particles.velocity[:, None, :]),
            axis=1,
        ),
        new_mass[:, None],
    )

    is_periodic, box_length = _periodic_box(config)
    new_position = sink_particles.position + position_shift
    new_position = jnp.where(
        is_periodic,
        jnp.mod(new_position, box_length),
        new_position,
    )

    sink_particles = sink_particles._replace(
        mass=new_mass,
        position=new_position,
        velocity=sink_particles.velocity + velocity_shift,
    )

    # -------------------------------------------------------------
    # ==================== ↑ Sink update ↑ ========================
    # -------------------------------------------------------------

    return _couple_to_gas(primitive_state, sink_particles, config, registered_variables)
