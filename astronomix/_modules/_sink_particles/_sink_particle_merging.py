"""
Merging of seeds into more active seeds within their reach, so that the
seeds of one collapse become one sink.

Seeds of one collapse that activate at different times hold each other back
through f_prox, so a collapse can leave several partly active seeds where
the checks of Federrath et al. (2010) form one sink. Once per time step, at
the end of the formation, seed j gives seed i the fraction

    μ_ij = K(|y_i − y_j| − ℓ Δx) step_δ(1 − a_j / a_i)

of its mass and momentum, with K the radial weight of f_prox and δ the
merging width: only a less active seed gives, so two seeds of activation 1
never merge, and seeds merge exactly where they hold each other back. Seed j
gives F_j = 1 − Π_i (1 − μ_ij) in all, shared among its receivers in
proportion to μ_ij, and keeps (1 − F_j) of its mass, momentum, activation
and claim on its home cell; a seed that gives everything leaves its slot
empty. A seed that gives everything takes nothing in, so that no mass is
left in an empty slot. The receivers keep their activations, so the
activation of a group of merged seeds is that of its most active seed.

Each receiver moves to the centre of mass of what it keeps and what it
takes in, with the mass ε ρ_res ΔV added to its own as in the accretion, and
its velocity is that of the total momentum; mass and momentum are conserved.

At the hard limit every activation is 0 or 1, so no seed merges.
"""

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.data_classes.simulation_state_struct import SinkParticles
from astronomix.option_classes.simulation_config import SimulationConfig

# astronomix functions
from astronomix._modules._sink_particles._sink_particle_helpers import _periodic_box
from astronomix._modules._sink_particles._sink_particle_smoothing import (
    _radial_weight,
    _safe_divide,
    _smooth_step,
)


def _merge_fractions(
    distance_squared: jax.Array,
    activation: jax.Array,
    config: SimulationConfig,
) -> jax.Array:
    """
    The fraction μ_ij = K(|y_i − y_j| − ℓ Δx) step_δ(1 − a_j / a_i) of seed j
    that merges into seed i in one step, between active seeds.

    Args:
        distance_squared: The squared distance between every two slots,
            shape (num_slots, num_slots).
        activation: The activation of every slot, shape (num_slots,).
        config: The simulation configuration.

    Returns:
        μ, shape (num_slots, num_slots), the receiver i along the first axis
        and the giving seed j along the second.
    """
    sink_particle_config = config.sink_particle_config
    spatial_width = sink_particle_config.spatial_width_in_cells * config.grid_spacing
    accretion_radius = sink_particle_config.accretion_radius_in_cells * config.grid_spacing

    is_active = activation > 0.0
    is_pair = (
        is_active[:, None]
        & is_active[None, :]
        & ~jnp.eye(activation.shape[0], dtype=bool)
    )
    reach = _radial_weight(
        distance_squared,
        accretion_radius + spatial_width,
        spatial_width,
    )
    order = _smooth_step(
        1.0 - _safe_divide(activation[None, :], activation[:, None]),
        sink_particle_config.merging_width,
    )
    return jnp.where(is_pair, reach * order, 0.0)


def _given_fraction(merge_fraction: jax.Array) -> jax.Array:
    """
    The fraction F_j = 1 − Π_i (1 − μ_ij) that each seed gives in all.

    F is −expm1 of a sum of logarithms, which keeps its relative precision
    when every μ is small; a seed with μ = 1 towards any receiver gives
    everything.

    Args:
        merge_fraction: μ, shape (num_slots, num_slots), receivers along the
            first axis.

    Returns:
        F, shape (num_slots,).
    """
    is_saturated = merge_fraction >= 1.0
    log_remaining = jnp.where(
        is_saturated,
        0.0,
        jnp.log1p(-jnp.where(is_saturated, 0.0, merge_fraction)),
    )
    return jnp.where(
        jnp.any(is_saturated, axis=0),
        1.0,
        -jnp.expm1(jnp.sum(log_remaining, axis=0)),
    )


def _merge_seeds(
    sink_particles: SinkParticles,
    separation: jax.Array,
    density_threshold: jax.Array,
    config: SimulationConfig,
) -> SinkParticles:
    """
    Merge every seed into the more active seeds within its reach.

    Args:
        sink_particles: The sink particles.
        separation: y_j − y_i between every two slots, taken to the nearest
            periodic copy, shape (num_slots, num_slots, 3), i along the first
            axis.
        density_threshold: ρ_res at every slot, shape (num_slots,).
        config: The simulation configuration.

    Returns:
        The sink particles after merging; the slots of the seeds that gave
        everything are empty.
    """
    activation = sink_particles.activation
    mass = sink_particles.mass
    velocity = sink_particles.velocity

    merge_fraction = _merge_fractions(
        jnp.sum(separation**2, axis=-1),
        activation,
        config,
    )
    merge_fraction = merge_fraction * (1.0 - _given_fraction(merge_fraction))[:, None]
    given_fraction = _given_fraction(merge_fraction)
    given_mass = (
        given_fraction[None, :]
        * _safe_divide(merge_fraction, jnp.sum(merge_fraction, axis=0)[None, :])
        * mass[None, :]
    )

    new_mass = (1.0 - given_fraction) * mass + jnp.sum(given_mass, axis=1)
    position_update_mass = (
        config.sink_particle_config.position_update_mass
        * density_threshold
        * config.grid_spacing**3
    )
    position_shift = _safe_divide(
        jnp.sum(given_mass[..., None] * separation, axis=1),
        (new_mass + position_update_mass)[:, None],
    )
    velocity_shift = _safe_divide(
        jnp.sum(
            given_mass[..., None] * (velocity[None, :, :] - velocity[:, None, :]),
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

    new_activation = (1.0 - given_fraction) * activation
    is_emptied = (activation > 0.0) & ~(new_activation > 0.0)

    def kept(field):
        """The field, 0 in the slots emptied by the merging."""
        return jnp.where(
            is_emptied.reshape(is_emptied.shape + (1,) * (field.ndim - 1)),
            jnp.zeros_like(field),
            field,
        )

    return SinkParticles(
        mass=kept(new_mass),
        position=kept(new_position),
        velocity=kept(velocity + velocity_shift),
        activation=new_activation,
        home_cell=kept(sink_particles.home_cell),
        generation=kept(sink_particles.generation),
        home_claim=(1.0 - given_fraction) * sink_particles.home_claim,
    )
