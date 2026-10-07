"""
The shock-history label ``entropy_initial`` is reset where it is non-finite or
runs away (``sanitize_entropy_label``, applied in ``update_shock_history``).

The label is the only unbounded library scalar. The ratio recovery of
near-collapsed cells can grow odd-even pairs of runaway labels (O(1e3) against
current entropies of O(1)); once one of them overflows, the NaN spreads over the
whole label field through the WENO stencil, every gradient becomes NaN and the
shock latch freezes.

Checked (x64, NATIVE_JAX, the 16^3 blast of ``_blast_setup``):

* in-window labels pass bit for bit, NaN / +-inf / runaway ones become the
  current entropy, with a finite (zero) tangent and cotangent;
* ``update_shock_history`` with poisoned labels: identical everywhere else,
  finite, and the latch cannot fire on a reset label;
* a forward run seeded with a NaN label heals (every field finite, every label
  within the window);
* the reverse-mode gradient through a runaway label is finite.

Run on the CPU::

    JAX_PLATFORMS=cpu python -m pytest pytests/differentiability/test_entropy_label_sanitize.py
"""

# ==== GPU selection ====
import os
if os.environ.get("JAX_PLATFORMS") == "cpu":
    # XLA:CPU compiles the unrolled step in seconds at optimisation level 0
    # (minutes and tens of GB otherwise); the numerics differ at round-off only.
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true",
    )
elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# jax
import jax
import jax.numpy as jnp

# The bitwise comparisons and the runaway label values need double precision.
jax.config.update("jax_enable_x64", True)

# astronomix constants
from astronomix import BACKWARDS
from astronomix._fluid_equations._passive_scalars import ENTROPY_LABEL_WINDOW
from astronomix.variable_registry.registered_variables import (
    ENTROPY_INITIAL_SLOT,
    NUM_SHOCK_HISTORY_SCALARS,
    SHOCKED_FRACTION_SLOT,
)

# astronomix functions
from astronomix import time_integration
from astronomix._fluid_equations._passive_scalars import (
    sanitize_entropy_label,
    specific_entropy,
    update_shock_history,
)
from astronomix._stencil_operations._stencil_operations import _shift

# the shared differentiability blast
from _blast_setup import (
    DT_FIXED,
    GAMMA,
    NUM_FIXED_STEPS,
    blast_params,
    fluid_only_registry,
    get_fluid_state,
    get_shock_history,
    setup_blast,
)


def _bits(array):
    """The float64 array reinterpreted as integers, for bitwise comparisons."""
    return np.asarray(array).view(np.int64)


def _entropy_label_row(registered_variables):
    """The state row of the ``entropy_initial`` label."""
    return registered_variables.shock_history_index + ENTROPY_INITIAL_SLOT


def test_sanitize_unit():
    """In-window labels pass bit for bit; NaN, +-inf and runaway labels become
    the current entropy, with zero tangent and finite, zero cotangent."""
    current_entropy = jnp.linspace(-30.0, 5.0, 9)
    window = ENTROPY_LABEL_WINDOW
    labels = jnp.array([jnp.nan, jnp.inf, -jnp.inf, -3805.75, 800.81, 0.0, 0.0, -20.75, 5.0])
    labels = labels.at[6].set(current_entropy[6] - (window - 1e-9))  # just inside the window
    sanitized = sanitize_entropy_label(labels, current_entropy)
    reset = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], bool)
    np.testing.assert_array_equal(_bits(sanitized)[reset], _bits(current_entropy)[reset])
    np.testing.assert_array_equal(_bits(sanitized)[~reset], _bits(labels)[~reset])

    # The tangent is the label's own inside the window and zero on a reset
    # cell, whatever the current entropy's tangent.
    _, tangent = jax.jvp(
        sanitize_entropy_label,
        (labels, current_entropy),
        (jnp.ones_like(labels), 2.0 * jnp.ones_like(current_entropy)),
    )
    np.testing.assert_array_equal(np.asarray(tangent), np.where(reset, 0.0, 1.0))

    # The cotangent stays finite through NaN / inf labels.
    def summed_label(label, entropy):
        return jnp.sum(sanitize_entropy_label(label, entropy))

    label_gradient, entropy_gradient = jax.grad(summed_label, argnums=(0, 1))(
        labels,
        current_entropy,
    )
    assert np.all(np.isfinite(label_gradient)) and np.all(np.isfinite(entropy_gradient))
    np.testing.assert_array_equal(np.asarray(label_gradient), np.where(reset, 0.0, 1.0))
    np.testing.assert_array_equal(np.asarray(entropy_gradient), 0.0)


def test_update_shock_history_poisoned_label():
    """Poisoned labels leave every other cell unchanged, stay finite, and a
    reset label cannot fire the shock latch."""
    config, registered_variables, state = setup_blast()
    fluid_registry = fluid_only_registry(registered_variables)

    # Two steps in, so that there is a real shock to latch on.
    evolved_state = time_integration(
        state,
        config,
        blast_params(t_end=2 * DT_FIXED),
        registered_variables,
    )
    primitive_state = get_fluid_state(evolved_state, registered_variables)
    history = get_shock_history(evolved_state, registered_variables)
    current_entropy = specific_entropy(primitive_state, GAMMA, fluid_registry)

    def updated_history(shock_history):
        return update_shock_history(
            shock_history,
            primitive_state,
            1e-3,
            GAMMA,
            config.shock_entropy_jump,
            config,
            fluid_registry,
        )

    clean = updated_history(history)
    label = history[ENTROPY_INITIAL_SLOT]
    assert np.all(np.abs(np.asarray(current_entropy - label)) <= ENTROPY_LABEL_WINDOW)
    # The clean label is returned as is.
    np.testing.assert_array_equal(_bits(clean[ENTROPY_INITIAL_SLOT]), _bits(label))

    # Poison with NaN, -inf and two runaway values; the latter in converging,
    # not yet shocked cells, where an unsanitized label would fire the latch.
    velocities = [
        primitive_state[fluid_registry.velocity_index.x],
        primitive_state[fluid_registry.velocity_index.y],
        primitive_state[fluid_registry.velocity_index.z],
    ]
    velocity_divergence = sum(
        _shift(velocity, -1, axis=axis) - _shift(velocity, 1, axis=axis)
        for axis, velocity in enumerate(velocities)
    )
    unshocked_converging = np.argwhere(
        np.asarray((velocity_divergence < 0) & (history[SHOCKED_FRACTION_SLOT] == 0.0))
    )
    assert len(unshocked_converging) >= 2
    poisoned_cells = [
        (3, 3, 3),
        (12, 4, 9),
        tuple(unshocked_converging[0]),
        tuple(unshocked_converging[len(unshocked_converging) // 2]),
    ]
    poison_values = [jnp.nan, -jnp.inf, -3805.75, -180.32]
    poisoned_label = label
    for cell, value in zip(poisoned_cells, poison_values):
        poisoned_label = poisoned_label.at[cell].set(value)
    updated = updated_history(history.at[ENTROPY_INITIAL_SLOT].set(poisoned_label))
    assert bool(jnp.all(jnp.isfinite(updated)))

    poisoned_mask = np.zeros(poisoned_label.shape, bool)
    for cell in poisoned_cells:
        poisoned_mask[cell] = True
    for slot in range(NUM_SHOCK_HISTORY_SCALARS):
        np.testing.assert_array_equal(
            _bits(updated[slot])[~poisoned_mask],
            _bits(clean[slot])[~poisoned_mask],
        )
    np.testing.assert_array_equal(
        _bits(updated[ENTROPY_INITIAL_SLOT])[poisoned_mask],
        _bits(current_entropy)[poisoned_mask],
    )
    # A reset label has entropy rise 0, so it cannot latch.
    np.testing.assert_array_equal(
        np.asarray(updated[SHOCKED_FRACTION_SLOT])[poisoned_mask],
        np.asarray(history[SHOCKED_FRACTION_SLOT])[poisoned_mask],
    )
    # Unsanitized, the two runaway labels would have shown an entropy rise
    # above the shock threshold in converging cells, i.e. latched.
    unsanitized_rise = (
        np.asarray(current_entropy)[poisoned_mask] - np.asarray(poisoned_label)[poisoned_mask]
    )
    assert np.sum(unsanitized_rise[2:] > float(config.shock_entropy_jump)) == 2


def test_forward_heals_nan_label():
    """A forward run seeded with a NaN and a runaway label stays finite and
    ends with every label inside the window."""
    config, registered_variables, state = setup_blast()
    label_row = _entropy_label_row(registered_variables)
    poisoned = state.at[label_row, 8, 8, 8].set(jnp.nan).at[label_row, 9, 8, 8].set(-3805.75)
    final_state = time_integration(poisoned, config, blast_params(), registered_variables)
    assert bool(jnp.all(jnp.isfinite(final_state)))
    current_entropy = specific_entropy(
        get_fluid_state(final_state, registered_variables),
        GAMMA,
        fluid_only_registry(registered_variables),
    )
    assert float(jnp.max(jnp.abs(current_entropy - final_state[label_row]))) <= ENTROPY_LABEL_WINDOW


def test_grad_finite_through_runaway_label():
    """The reverse-mode gradient through runaway labels is finite."""
    config, registered_variables, state = setup_blast(
        fixed_timestep=True,
        num_timesteps=NUM_FIXED_STEPS,
        differentiation_mode=BACKWARDS,
        ad_remat="stage",
    )
    params = blast_params(t_end=NUM_FIXED_STEPS * DT_FIXED)
    label_row = _entropy_label_row(registered_variables)
    poisoned = state.at[label_row, 8, 8, 8].set(-3805.75).at[label_row, 9, 8, 8].set(800.81)
    cotangent = jnp.asarray(np.random.default_rng(3).normal(size=state.shape))

    def projected_final_state(initial_state):
        return jnp.vdot(
            cotangent,
            time_integration(initial_state, config, params, registered_variables),
        )

    gradient = jax.jit(jax.grad(projected_final_state))(poisoned)
    assert bool(jnp.all(jnp.isfinite(gradient)))
