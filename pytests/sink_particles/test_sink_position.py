"""
The position of the smooth candidates: the move of light candidates toward
the potential minimum (step 7), the ranking of the proximity check of
recycling (step 6), and a candidate that activates at the same position
whatever the step in which it was created.
"""

# numerics
import numpy as np

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix import FINITE_VOLUME, NATIVE_JAX

# astronomix containers
from astronomix import (
    BackendConfig,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
    SinkParticleConfig,
)
from astronomix.data_classes.simulation_state_struct import SinkParticles

# astronomix functions
from astronomix import get_registered_variables, time_integration
from astronomix._modules._sink_particles import _sink_particle_update
from astronomix._modules._sink_particles._sink_particle_formation import (
    SlotChecks,
    _cell_quantities,
    _slot_checks,
    _slot_neighbourhood,
    _slot_separation_squared,
)
from astronomix._modules._sink_particles._sink_particle_helpers import (
    _empty_sink_particles,
)
from astronomix._modules._sink_particles._sink_particle_update import (
    _recycle_slots,
    _relocate_candidates,
    _soft_minimum_position,
    _update_sink_particles,
)
from astronomix.test_setups.self_gravity.bonnor_ebert_sphere3D import (
    BonnorEbertSettings,
    setup_bonnor_ebert_sphere,
)

# sink particle test helpers
from _sink_helpers import (
    CELL_SIZE,
    CLUMP_CENTER,
    SETTINGS,
    _density_threshold,
    _setup,
    _setup_hard_limit,
    _update_sinks_once,
    _with_sink,
)

#: A clump shallow enough that a candidate a few cells off its centre needs
#: more than one move to reach it.
SHALLOW_CLUMP = SETTINGS._replace(peak_overdensity=100.0)


def _with_candidate(sink_particles, slot, offset_in_cells, activation=0.0, mass=0.0):
    """The sink particles with a candidate at rest in ``slot``, offset from the
    clump centre by ``offset_in_cells``, with the given activation and mass."""
    position = [CLUMP_CENTER + offset * CELL_SIZE for offset in offset_in_cells]
    sink_particles = _with_sink(sink_particles, slot, mass, position)
    return sink_particles._replace(
        activation=sink_particles.activation.at[slot].set(activation)
    )


def _relocate(primitive_state, sink_particles, config, params, registered_variables):
    """Step 7 alone on the given gas, with the quantities of step 2 at the
    slots' positions."""
    cells = _cell_quantities(
        primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )
    neighbourhood = _slot_neighbourhood(
        sink_particles.position,
        primitive_state.shape[1:],
        config,
    )
    checks = _slot_checks(neighbourhood, sink_particles.mass, cells, config)
    minimum_position = _soft_minimum_position(
        sink_particles.position,
        neighbourhood,
        cells,
        config,
    )
    return _relocate_candidates(sink_particles, minimum_position, checks, config)


def _distance_from_center_in_cells(sink_particles, num_slots):
    """The distance of the first ``num_slots`` slots from the clump centre,
    in cells."""
    offset = (sink_particles.position[:num_slots] - CLUMP_CENTER) / CELL_SIZE
    return np.asarray(jnp.linalg.norm(offset, axis=-1))


# -------------------------------------------------------------
# ======================= ↓ Relocation ↓ ======================
# -------------------------------------------------------------


#: Offsets from the clump centre, in cells, all within r_acc = 2.5 cells.
OFFSETS_WITHIN_REACH = (
    (1.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (1.5, 1.0, 0.5),
    (0.3, -0.4, 0.2),
    (2.2, 0.8, 0.0),
)


def test_massless_candidate_reaches_isolated_minimum_within_three_moves():
    """Massless candidates placed anywhere within r_acc of the centre of a
    shallow clump, the only potential minimum in their reach, are within
    1e-4 cells of it after three moves of step 7. After the first move the
    farthest is still about 0.5 cells off."""
    state, config, params, registered_variables = _setup(settings=SHALLOW_CLUMP)
    sink_particles = _empty_sink_particles(config, state.primitive_state.dtype)
    for slot, offset in enumerate(OFFSETS_WITHIN_REACH):
        sink_particles = _with_candidate(sink_particles, slot, offset)

    for _ in range(3):
        sink_particles = _relocate(
            state.primitive_state,
            sink_particles,
            config,
            params,
            registered_variables,
        )

    distance = _distance_from_center_in_cells(sink_particles, len(OFFSETS_WITHIN_REACH))
    assert np.all(distance < 1e-4), (
        f"after three moves the candidates are {distance} cells from the potential "
        "minimum, not within 1e-4 cells"
    )


def test_sinks_and_slots_holding_minimum_mass_do_not_move():
    """Of four slots 1.5 cells off the centre of a shallow clump, a massless
    candidate moves to within 0.5 cells of the centre in one move of step 7,
    while a massless sink (a = 1) and candidates holding 1.01 and 2 m_min
    ρ_thr ΔV stay exactly where they are."""
    state, config, params, registered_variables = _setup(settings=SHALLOW_CLUMP)
    minimum_mass = (
        config.sink_particle_config.recycling_mass
        * _density_threshold(config, params)
        * CELL_SIZE**3
    )
    offset = (1.5, 0.0, 0.0)
    sink_particles = _empty_sink_particles(config, state.primitive_state.dtype)
    sink_particles = _with_candidate(sink_particles, 0, offset)
    sink_particles = _with_candidate(sink_particles, 1, offset, activation=1.0)
    sink_particles = _with_candidate(sink_particles, 2, offset, mass=1.01 * minimum_mass)
    sink_particles = _with_candidate(sink_particles, 3, offset, mass=2.0 * minimum_mass)

    relocated = _relocate(
        state.primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )

    distance = _distance_from_center_in_cells(relocated, 1)[0]
    assert distance < 0.5, (
        f"the massless candidate is {distance:.3f} cells from the centre after one move, "
        "not within 0.5 cells"
    )
    np.testing.assert_array_equal(
        relocated.position[1:4],
        sink_particles.position[1:4],
        err_msg="the sink or a candidate holding at least m_min moved",
    )


def test_no_slot_moves_at_hard_limit():
    """With every parameter at its hard limit, the clump forms its sink with
    a = 1, so after two updates every occupied slot is a sink, and step 7
    leaves each where it is."""
    state, config, params, registered_variables = _setup_hard_limit()
    primitive_state, sink_particles = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
    )
    primitive_state, sink_particles = _update_sink_particles(
        primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )
    occupied = np.asarray(sink_particles.occupied)
    activation = np.asarray(sink_particles.activation)
    assert occupied.any(), "no sink formed at the hard limit"
    assert np.all(activation[occupied] == 1.0), (
        f"occupied slots at the hard limit have activations {activation[occupied]}, not 1"
    )

    relocated = _relocate(
        primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )
    np.testing.assert_array_equal(
        relocated.position,
        sink_particles.position,
        err_msg="step 7 moved a slot at the hard limit",
    )


# -------------------------------------------------------------
# ======================= ↑ Relocation ↑ ======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================== ↓ Recycling ↓ ======================
# -------------------------------------------------------------


def _passing_checks(num_slots):
    """Quantities of the creation checks that every slot passes by a wide
    margin, with ρ_thr = 1, so that m_min ρ_thr ΔV ≈ 3e-7."""
    return SlotChecks(
        has_cells=jnp.ones(num_slots, dtype=bool),
        density_threshold=jnp.ones(num_slots),
        density_ratio=jnp.full(num_slots, 10.0),
        converging_velocity=jnp.ones((num_slots, 6)),
        potential_height=jnp.full(num_slots, -1.0),
        jeans_ratio=jnp.full(num_slots, 10.0),
        bound_ratio=jnp.full(num_slots, 10.0),
        center_of_mass_velocity=jnp.zeros((num_slots, 3)),
    )


def _recycle_two_candidates(activation, mass, distance_in_cells):
    """Recycle two candidates at rest, ``distance_in_cells`` apart along x,
    with the given activations and masses, whose gas passes every creation
    check.

    Returns:
        Whether each of the two slots is freed.
    """
    state, config, _, registered_variables = _setup()
    position = jnp.array(
        [
            [CLUMP_CENTER, CLUMP_CENTER, CLUMP_CENTER],
            [CLUMP_CENTER + distance_in_cells * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER],
        ]
    )
    sink_particles = SinkParticles(
        mass=jnp.array(mass),
        position=position,
        velocity=jnp.zeros((2, 3)),
        activation=jnp.array(activation),
        occupied=jnp.ones(2, dtype=bool),
    )
    _, _, is_freed = _recycle_slots(
        state.primitive_state,
        sink_particles,
        _passing_checks(2),
        _slot_separation_squared(position, config),
        config,
        registered_variables,
    )
    return np.asarray(is_freed)


@pytest.mark.parametrize(
    ("activation", "mass", "kept_slot"),
    [
        ((0.0, 0.005), (1e-7, 0.0), 1),
        ((0.0, 0.0), (0.0, 1e-7), 1),
        ((0.0, 0.0), (0.0, 0.0), 0),
    ],
    ids=["higher_activation_kept", "higher_mass_kept", "lower_index_kept"],
)
def test_recycling_frees_lower_ranked_of_two_close_candidates(activation, mass, kept_slot):
    """Of two candidates below a_min and m_min, one cell apart, whose gas
    passes every creation check, only the one that ranks lower is freed: by
    activation, at equal activations by mass, and at equal masses by index.

    Args:
        activation: The activations of the two candidates.
        mass: Their masses.
        kept_slot: The slot that ranks higher.
    """
    is_freed = _recycle_two_candidates(activation, mass, distance_in_cells=1.0)
    expected = [slot != kept_slot for slot in range(2)]
    assert list(is_freed) == expected, (
        f"recycling freed slots {is_freed} of two close candidates with activations "
        f"{activation} and masses {mass}; only slot {1 - kept_slot} should be freed"
    )


def test_recycling_keeps_candidates_farther_apart_than_accretion_radius():
    """Two candidates below a_min and m_min, 3 cells apart, beyond r_acc =
    2.5 cells, whose gas passes every creation check, are both kept."""
    is_freed = _recycle_two_candidates((0.0, 0.0), (0.0, 0.0), distance_in_cells=3.0)
    assert not is_freed.any(), f"recycling freed slots {is_freed} of two distant candidates"


#: Two clumps of this peak density, this many cells apart along x, keep a
#: massless candidate at the centre of each inactive, and each passes every
#: relaxed creation check.
TWO_CLUMPS_PEAK_OVERDENSITY = 150.0
TWO_CLUMPS_SEPARATION_IN_CELLS = 10


def _two_clumps():
    """Two Gaussian clumps along x, with a massless candidate at the centre
    of each.

    Returns:
        ``(primitive_state, sink_particles, config, params,
        registered_variables)``.
    """
    centers = [
        (CLUMP_CENTER + offset * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER)
        for offset in (0, TWO_CLUMPS_SEPARATION_IN_CELLS)
    ]
    states = []
    for center in centers:
        state, config, params, registered_variables = _setup(
            settings=SETTINGS._replace(
                peak_overdensity=TWO_CLUMPS_PEAK_OVERDENSITY,
                overdensity_center=center,
            )
        )
        states.append(state)
    uniform_state, _, _, _ = _setup(settings=SETTINGS._replace(peak_overdensity=0.0))
    # Density and pressure are linear in the overdensity, and the gas is at rest.
    primitive_state = (
        states[0].primitive_state + states[1].primitive_state - uniform_state.primitive_state
    )

    sink_particles = _empty_sink_particles(config, primitive_state.dtype)
    for slot, center in enumerate(centers):
        offset_in_cells = [(c - CLUMP_CENTER) / CELL_SIZE for c in center]
        sink_particles = _with_candidate(sink_particles, slot, offset_in_cells)
    return primitive_state, sink_particles, config, params, registered_variables


def test_candidates_brought_within_accretion_radius_are_ranked_in_same_update(monkeypatch):
    """Two massless candidates at the centres of two clumps 10 cells apart
    stay inactive and are both kept by an update. When step 7 is made to put
    the second 1 cell from the first, the same update frees the second (the
    lower-ranked, by index) and keeps the first: the proximity check of
    recycling sees the positions after step 7."""
    primitive_state, sink_particles, config, params, registered_variables = _two_clumps()

    _, updated = _update_sink_particles(
        primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )
    occupied = np.asarray(updated.occupied[:2])
    activation = np.asarray(updated.activation[:2])
    assert occupied.all() and np.all(activation == 0.0), (
        f"the candidates of the two clumps end with occupied {occupied} and activations "
        f"{activation}, not both occupied and inactive"
    )

    relocate_candidates = _sink_particle_update._relocate_candidates

    def relocate_next_to_first(sink_particles, minimum_position, checks, config):
        moved = relocate_candidates(sink_particles, minimum_position, checks, config)
        return moved._replace(
            position=moved.position.at[1].set(moved.position[0] + jnp.array([CELL_SIZE, 0.0, 0.0]))
        )

    monkeypatch.setattr(_sink_particle_update, "_relocate_candidates", relocate_next_to_first)
    # Without jit, the update calls the patched step 7.
    with jax.disable_jit():
        _, updated = _update_sink_particles(
            primitive_state,
            sink_particles,
            config,
            params,
            registered_variables,
        )
    occupied = np.asarray(updated.occupied[:2])
    assert list(occupied) == [True, False], (
        f"after step 7 put the candidates 1 cell apart, the occupied slots are {occupied}, "
        "not only the first"
    )


# -------------------------------------------------------------
# ======================== ↑ Recycling ↑ ======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ====================== ↓ Creation step ↓ ====================
# -------------------------------------------------------------


#: The runs of the Bonnor–Ebert sphere off the cell centre: fixed steps to
#: t = 0.1, about twice the adaptive steps over that time, as in the gradient
#: runs of ``hard_baseline.py``, with the sinks recorded at every step.
BONNOR_EBERT_NUM_CELLS = 32
BONNOR_EBERT_NUM_STEPS = 300
BONNOR_EBERT_END_TIME = 0.1

#: The offset of the sphere's centre from the centre of the central cell
#: along x, in cells.
BONNOR_EBERT_CENTER_OFFSET = 0.25


def _bonnor_ebert_off_center(creation_margin):
    """Run the Bonnor–Ebert sphere with its centre off the central cell
    centre along x, with the given creation margin μ.

    Returns:
        The sink particles at every step, each field with a leading axis of
        steps.
    """
    num_cells = BONNOR_EBERT_NUM_CELLS
    central_cell_center = (num_cells // 2 + 0.5) / num_cells
    settings = BonnorEbertSettings(
        center=(
            central_cell_center + BONNOR_EBERT_CENTER_OFFSET / num_cells,
            central_cell_center,
            central_cell_center,
        ),
        t_end=BONNOR_EBERT_END_TIME,
    )
    config = SimulationConfig(
        num_cells=num_cells,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(),
        progress_bar=False,
        fixed_timestep=True,
        num_timesteps=BONNOR_EBERT_NUM_STEPS,
        return_snapshots=True,
        num_snapshots=BONNOR_EBERT_NUM_STEPS,
        sink_particle_config=SinkParticleConfig(
            sink_particles=True,
            creation_margin=creation_margin,
        ),
    )
    state, config, params = setup_bonnor_ebert_sphere(config, SimulationParams(), settings)
    snapshots = time_integration(state, config, params, get_registered_variables(config))
    return jax.tree.map(np.asarray, snapshots.sink_particles)


def test_candidate_activates_at_same_position_whatever_its_creation_step():
    """In the Bonnor–Ebert sphere off the cell centre, a creation margin of
    0.45 instead of 0.5 creates the candidate about 80 steps earlier. μ
    enters only creation and recycling, and a candidate with a = m = 0
    neither accretes, nor exchanges momentum with the gas, nor adds to the
    potential, so the two runs have the same gas until the candidate
    activates. Step 7 moves the candidate to the same position before then:
    it becomes a sink in the same step, at the same position, and the two
    runs end with the same sinks, to rounding. Without step 7 the positions
    at activation differ by 2.6e-4 cells and the final masses by 0.12%."""
    late = _bonnor_ebert_off_center(creation_margin=0.5)
    early = _bonnor_ebert_off_center(creation_margin=0.45)

    created = [int(np.argmax(sinks.occupied.any(axis=1))) for sinks in (late, early)]
    assert created[0] - created[1] > 10, (
        f"the candidate is created at steps {created} with μ = 0.5 and 0.45, not at "
        "least 10 steps apart, so the test does not change the creation step"
    )
    formed = [int(np.argmax((sinks.activation > 0.5).any(axis=1))) for sinks in (late, early)]
    assert formed[0] == formed[1] and formed[0] > 0, (
        f"the first sink forms at steps {formed} with μ = 0.5 and 0.45, not in the "
        "same step"
    )

    position_in_cells = [
        sinks.position[formed[0]][sinks.occupied[formed[0]]] * BONNOR_EBERT_NUM_CELLS
        for sinks in (late, early)
    ]
    assert position_in_cells[0].shape == position_in_cells[1].shape == (1, 3), (
        f"the runs hold {[p.shape[0] for p in position_in_cells]} candidates when the "
        "sink forms, not one each"
    )
    np.testing.assert_allclose(
        position_in_cells[1],
        position_in_cells[0],
        rtol=0.0,
        atol=1e-5,
        err_msg="the candidates created in different steps activate at different "
        "positions, in cells",
    )
    np.testing.assert_allclose(
        early.mass[-1],
        late.mass[-1],
        rtol=1e-5,
        atol=0.0,
        err_msg="the sink masses at the end differ between the two creation steps",
    )
    np.testing.assert_allclose(
        early.position[-1] * BONNOR_EBERT_NUM_CELLS,
        late.position[-1] * BONNOR_EBERT_NUM_CELLS,
        rtol=0.0,
        atol=1e-5,
        err_msg="the sink positions at the end, in cells, differ between the two "
        "creation steps",
    )


# -------------------------------------------------------------
# ====================== ↑ Creation step ↑ ====================
# -------------------------------------------------------------
