"""
Pallas implementations of the fifth-order WENO interface flux.

This module is the Pallas backend of the WENO interface-flux step. All
native-JAX implementations live in ``_weno.py``; its dispatchers call into this
module when ``config.backend_config.backend == PALLAS`` and the per-flavour
``_*_pallas_flux_supported`` predicate accepts. A developer who only changes
the native JAX never needs to touch this file: the kernels here are
translations of the native stencils (see
``agent_guides/pallas_backend_implementation_guide.md``, §2 for the kernel
skeleton and §4 for the per-flavour recipes, and the ``pallasify`` skill).

The module covers

- the ideal-gas hydrodynamic WENO flux (``_weno_flux_hydro_pallas``),
- the ideal-gas MHD WENO flux (``_weno_flux_mhd_pallas``), including the
  x-direction variants with a kept x halo for the multi-GPU fast path of
  ``_lsrk4_with_ct``,
- the isothermal MHD WENO flux (``_weno_flux_mhd_iso_pallas``),
- a fused WENO flux + divergence kernel (``_weno_flux_hydro_pallas_rhs``).

All kernels support the admissible face state of the characteristic basis,
the positivity-preserving flux splitting (``weno_positivity_preserving``)
where the native kernels do, and the hydro and ideal-MHD kernels the
dual-energy pressure switch. Their AD tangents are the native kernels
(``diffable_pallas_call`` in ``_weno.py``). The shared block-shape,
compiler-parameter and multi-GPU helpers live in ``astronomix._pallas_helpers``.
"""

# typing
from typing import NamedTuple

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    IDEAL_GAS,
    ISOTHERMAL,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._pallas_helpers import (
    _as_3tuple_block_shape,
    _backend_is_pallas,
    _pallas_call_sharded,
    _pallas_compiler_params,
    pl,
)
from astronomix._finite_difference._interface_fluxes._weno_positivity import (
    mass_free_modes,
    mhd_physical_flux,
    positivity_preserving_flux_local,
    positivity_preserving_interface_flux,
)
from astronomix._finite_difference._interface_fluxes._weno_weights import (
    _weno_omega_weights,
    _weno_omega_weights_z,
)


# -------------------------------------------------------------
# ============ ↓ Shared per-interface building blocks ↓ =======
# -------------------------------------------------------------
#
# The kernels work on tuples of the local conserved components of the six
# cells i - 2 ... i + 3 of the stencil of the interface i + 1/2, so the
# stencil position k holds the cell at offset k - 2 (k = 2 is cell i, k = 3 is
# cell i + 1). The helpers below are plain Python and are inlined into every
# kernel at trace time.


class _FlooredHydroCell(NamedTuple):
    """Primitive and wave-speed data of one cell after the density and
    pressure floors (hydrodynamics, local component order)."""

    density: object
    normal_momentum: object
    transverse_momentum_1: object
    transverse_momentum_2: object
    energy: object
    normal_velocity: object
    transverse_velocity_1: object
    transverse_velocity_2: object
    velocity_squared: object
    pressure: object
    specific_enthalpy: object
    sound_speed: object


class _FlooredMhdCell(NamedTuple):
    """Primitive and wave-speed data of one cell after the density and
    pressure floors (ideal MHD, local component order)."""

    density: object
    normal_momentum: object
    transverse_momentum_1: object
    transverse_momentum_2: object
    normal_magnetic_field: object
    transverse_magnetic_field_1: object
    transverse_magnetic_field_2: object
    energy: object
    normal_velocity: object
    transverse_velocity_1: object
    transverse_velocity_2: object
    velocity_squared: object
    magnetic_field_squared: object
    pressure: object
    specific_enthalpy: object
    sound_speed: object
    sound_speed_squared: object
    fast_speed: object
    alfven_speed: object
    slow_speed: object


class _FlooredIsothermalMhdCell(NamedTuple):
    """Primitive and wave-speed data of one cell after the density floor
    (isothermal MHD, local component order)."""

    density: object
    normal_momentum: object
    transverse_momentum_1: object
    transverse_momentum_2: object
    normal_magnetic_field: object
    transverse_magnetic_field_1: object
    transverse_magnetic_field_2: object
    normal_velocity: object
    transverse_velocity_1: object
    transverse_velocity_2: object
    fast_speed: object
    alfven_speed: object
    slow_speed: object


class _SplitFaceFluxes(NamedTuple):
    """
    The two upwind-split face fluxes of the positivity-preserving WENO flux,
    kept apart until the recombination.

    ``plus`` / ``minus`` are the reconstructed split fluxes F^+ and F^- at the
    interface (central part included), ``plus_shift`` / ``minus_shift`` the
    corrections z^+- of the upwind split states for fields that keep their own
    splitting speed, ``common_speed`` the splitting speed of all mass-carrying
    fields and ``safe_speed`` its positive floor.
    """

    plus: list
    minus: list
    plus_shift: list
    minus_shift: list
    common_speed: object
    safe_speed: object


def _central_face_average(minus_one, center, plus_one, plus_two):
    """
    The fourth-order central interface value (-v_{i-1} + 7 v_i + 7 v_{i+1} -
    v_{i+2}) / 12 at i + 1/2 from four cell values.
    """
    return (-minus_one + 7.0 * center + 7.0 * plus_one - plus_two) * (1.0 / 12.0)


def _central_face_values(stencil, num_components):
    """
    The central interface value of every local component of a six-cell
    stencil of component tuples.
    """
    return [
        _central_face_average(stencil[1][slot], stencil[2][slot], stencil[3][slot], stencil[4][slot])
        for slot in range(num_components)
    ]


def _start_split_face_fluxes(central_flux, conserved_stencil, common_speed, num_components):
    """
    Start the split face fluxes of the positivity-preserving WENO flux from
    their central parts, 0.5 * (F_c +- alpha q_c), with the common splitting
    speed alpha.

    Args:
        central_flux: The central interface flux, one tile per component.
        conserved_stencil: The six-cell stencil of local conserved tuples.
        common_speed: The splitting speed of all mass-carrying fields.
        num_components: The number of local components.

    Returns:
        The ``_SplitFaceFluxes`` before the characteristic corrections.
    """
    central_state = _central_face_values(conserved_stencil, num_components)
    plus = [
        0.5 * (central_flux[slot] + common_speed * central_state[slot])
        for slot in range(num_components)
    ]
    minus = [
        0.5 * (central_flux[slot] - common_speed * central_state[slot])
        for slot in range(num_components)
    ]
    return _SplitFaceFluxes(
        plus=plus,
        minus=minus,
        plus_shift=[central_flux[0] * 0.0 for _ in range(num_components)],
        minus_shift=[central_flux[0] * 0.0 for _ in range(num_components)],
        common_speed=common_speed,
        safe_speed=jnp.maximum(common_speed, 1e-30),
    )


def _add_mode_to_split_face_fluxes(
    split,
    add_right_correction,
    mode,
    plus_correction,
    minus_correction,
    *,
    own_speed=None,
    projected_state=None,
):
    """
    Add the WENO correction of one characteristic field to the split face
    fluxes, as in the native ``_weno_flux_x_native``.

    A field that carries no mass keeps its own splitting speed ``own_speed``
    instead of the common one; its central part and the upwind cells' split
    states are then shifted along its right eigenvector by the speed
    difference.

    Args:
        split: The current ``_SplitFaceFluxes``.
        add_right_correction: The kernel's ``(fluxes, mode, amplitude)`` ->
            fluxes + amplitude * right eigenvector of ``mode``.
        mode: The characteristic field.
        plus_correction: The upwind-biased reconstruction of the positive
            split flux of the field (enters with a minus sign).
        minus_correction: The reconstruction of the negative split flux.
        own_speed: The field's own splitting speed, or None for the common one.
        projected_state: The six-cell stencil of the field's characteristic
            variable (needed with ``own_speed``).

    Returns:
        The updated ``_SplitFaceFluxes``.
    """
    if own_speed is None:
        return split._replace(
            plus=add_right_correction(split.plus, mode, -plus_correction),
            minus=add_right_correction(split.minus, mode, minus_correction),
        )

    num_components = len(split.plus)
    zeros = [split.plus[0] * 0.0 for _ in range(num_components)]
    plus = add_right_correction(split.plus, mode, -plus_correction)
    minus = add_right_correction(split.minus, mode, minus_correction)
    speed_offset = own_speed - split.common_speed
    central_projection = _central_face_average(
        projected_state[1],
        projected_state[2],
        projected_state[3],
        projected_state[4],
    )
    central_shift = add_right_correction(zeros, mode, 0.5 * speed_offset * central_projection)
    plus = [plus[slot] + central_shift[slot] for slot in range(num_components)]
    minus = [minus[slot] - central_shift[slot] for slot in range(num_components)]
    relative_offset = speed_offset / split.safe_speed
    return split._replace(
        plus=plus,
        minus=minus,
        plus_shift=add_right_correction(split.plus_shift, mode, relative_offset * projected_state[2]),
        minus_shift=add_right_correction(
            split.minus_shift,
            mode,
            relative_offset * projected_state[3],
        ),
    )


def _weno5_shard_wrap(
    kernel_local,
    conserved_state,
    config,
    axis,
    extra_state_inputs=(),
    halo_cells=3,
):
    """
    Multi-GPU wrap of a per-axis fifth-order WENO Pallas kernel.

    The WENO5 stencil reads the offsets -2..+3 along the flux axis only, so a
    halo of 3 cells on that axis suffices; off-axis the kernel reads only its
    own cells, so no halo is needed there even if those axes are sharded. All
    WENO kernels of this module share this reach and funnel through this
    helper. Without an active ``pallas_mesh_context`` it simply calls
    ``kernel_local``.

    Args:
        kernel_local: The single-shard kernel build,
            ``(state_local, *extra_local) -> flux``.
        conserved_state: The conserved state.
        config: The simulation configuration.
        axis: The flux axis.
        extra_state_inputs: Additional state-shaped arrays (leading variable
            axis, same spatial shape and sharding as the state, e.g. the
            dual-energy field as ``g[None]``) that ride the same halo exchange;
            they are passed to ``kernel_local`` after the state.
        halo_cells: The halo along the flux axis; wider for kernels whose
            result also depends on the neighbouring interfaces (the paired
            positivity-preserving recombination of ideal MHD reads -3..+4).

    Returns:
        The flux returned by ``kernel_local``.
    """
    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape,
        ndim,
        spatial_shape=conserved_state.shape[1:],
    )
    halo_list = [0, 0, 0]
    if 0 <= int(axis) < ndim:
        halo_list[int(axis)] = halo_cells
    halo = tuple(halo_list[:ndim])

    return _pallas_call_sharded(
        kernel_local,
        state_inputs=(conserved_state,) + tuple(extra_state_inputs),
        halo=halo,
        block_shape=block_shape[:ndim],
    )


def _blocks_divide_state(conserved_state, config: SimulationConfig) -> bool:
    """Whether the configured Pallas block shape divides every spatial dimension."""
    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape,
        ndim,
        spatial_shape=conserved_state.shape[1:],
    )
    for extent, block_size in zip(conserved_state.shape[1:], block_shape[:ndim], strict=True):
        if int(extent) % int(block_size) != 0:
            return False
    return True


def _native_weno_flux(axis: int):
    """
    The native-JAX WENO flux along ``axis`` (the fallback of the Pallas entry
    points when their support predicate rejects the configuration).
    """
    # Imported lazily: ``_weno.py`` imports this module at load time.
    from astronomix._finite_difference._interface_fluxes._weno import (
        _weno_flux_x_native,
        _weno_flux_y_native,
        _weno_flux_z_native,
    )

    return (_weno_flux_x_native, _weno_flux_y_native, _weno_flux_z_native)[axis]


# -------------------------------------------------------------
# ============ ↑ Shared per-interface building blocks ↑ =======
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============ ↓ Ideal-gas hydrodynamics ↓ ====================
# -------------------------------------------------------------


def _hydro_pallas_flux_supported(conserved_state, config: SimulationConfig) -> bool:
    """
    Whether the Pallas hydro WENO kernel can be used: ideal-gas
    hydrodynamics in 1, 2 or 3 dimensions with a block shape that divides the
    grid. MHD has its own kernels; isothermal hydrodynamics uses the native
    flux.

    Args:
        conserved_state: The conserved state.
        config: The simulation configuration.

    Returns:
        True if the kernel can be used.
    """
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if config.mhd:
        return False
    if config.equation_of_state != IDEAL_GAS:
        return False
    ndim = int(config.dimensionality)
    if ndim not in (1, 2, 3):
        return False
    if conserved_state.ndim != ndim + 1:
        return False
    return _blocks_divide_state(conserved_state, config)


def _hydro_indices_for_axis(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    axis: int,
):
    """
    Return the local Euler component indices for a flux normal to ``axis``.

    The order is the local characteristic order of the Euler eigenvectors:
    density, normal momentum, first transverse momentum, optional second
    transverse momentum, energy. The indices refer to the conserved-state
    variable axis.

    Args:
        config: The simulation configuration.
        registered_variables: The registered variables.
        axis: The flux axis.

    Returns:
        The tuple of state indices in local order.
    """
    density_index = int(registered_variables.density_index)
    energy_index = int(registered_variables.energy_index)
    ndim = int(config.dimensionality)

    if ndim == 1:
        momentum_x = int(registered_variables.momentum_index)
        return (density_index, momentum_x, energy_index)

    momentum_x = int(registered_variables.momentum_index.x)
    momentum_y = int(registered_variables.momentum_index.y)
    if ndim == 2:
        if axis == 0:
            return (density_index, momentum_x, momentum_y, energy_index)
        return (density_index, momentum_y, momentum_x, energy_index)

    momentum_z = int(registered_variables.momentum_index.z)
    if axis == 0:
        return (density_index, momentum_x, momentum_y, momentum_z, energy_index)
    if axis == 1:
        return (density_index, momentum_y, momentum_x, momentum_z, energy_index)
    return (density_index, momentum_z, momentum_y, momentum_x, energy_index)


def _weno_flux_hydro_pallas(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
    internal_energy_density=None,
):
    """
    Pallas implementation of the ideal-gas hydrodynamic WENO flux.

    Public entry point: checks the support predicate (falling back to the
    native flux) and applies the multi-GPU ``shard_map`` + halo wrap. The
    arithmetic lives in ``_weno_flux_hydro_pallas_local``, so the same kernel
    build runs on the global state (single device) or on a halo-padded local
    shard (multi device).

    Args:
        conserved_state: The conserved state.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        axis: The flux axis.
        internal_energy_density: Optional dual-energy field ``g`` (cell
            centred, the state's spatial shape, not transposed: the kernel is
            axis aware). It replaces the pressure recovery from the total
            energy in the flux and the eigenstructure wherever that recovery
            is unreliable, and rides the same halo exchange as the state.

    Returns:
        The interface flux F_{i+1/2} along ``axis``.
    """
    if not _hydro_pallas_flux_supported(conserved_state, config):
        return _native_weno_flux(axis)(
            conserved_state,
            params,
            config,
            registered_variables,
            internal_energy_density=internal_energy_density,
        )

    if internal_energy_density is None:

        def build_local(state_local):
            return _weno_flux_hydro_pallas_local(
                state_local,
                params,
                config,
                registered_variables,
                axis=axis,
            )

        return _weno5_shard_wrap(build_local, conserved_state, config, axis)

    # With dual energy, g rides the halo exchange as a state-shaped (1, ...)
    # array.
    internal_energy_channel = internal_energy_density[None]

    def build_local_dual(state_local, internal_energy_local):
        return _weno_flux_hydro_pallas_local(
            state_local,
            params,
            config,
            registered_variables,
            axis=axis,
            internal_energy_density=internal_energy_local,
        )

    return _weno5_shard_wrap(
        build_local_dual,
        conserved_state,
        config,
        axis,
        extra_state_inputs=(internal_energy_channel,),
    )


def _weno_flux_hydro_pallas_local(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
    internal_energy_density=None,
):
    """Single-shard hydro-WENO kernel build.  When called from inside a
    ``shard_map`` body, ``conserved_state.shape`` is the local halo-padded
    shape and the kernel's grid / modular indexing wrap within that shape.
    Outside ``shard_map`` (single-device path) the shape is global.

    ``internal_energy_density`` (dual-energy ``g``), when given, is a
    ``(1, *spatial)`` array matching the (possibly halo-padded) state's
    spatial shape; the kernel applies the Bryan+95 switch to the pressure
    recovery in both the physical flux and the eigenstructure, mirroring
    the native ``dual_switched_pressure_hydro`` / eigen threading."""
    has_g = internal_energy_density is not None
    ndim = int(config.dimensionality)
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx, by, bz = _as_3tuple_block_shape(config.backend_config.pallas_block_shape, ndim, spatial_shape=spatial_shape)
    grid = (nx // bx, ny // by, nz // bz)

    local_indices = _hydro_indices_for_axis(config, registered_variables, axis)
    ncomp = len(local_indices)
    num_modes = ndim + 2
    epsilon = config.weno_epsilon
    # WENO-Z is a static config choice, so the weight function can be bound
    # here and inlined into the Pallas kernel body below.
    omega_weights = _weno_omega_weights_z if config.weno_z else _weno_omega_weights
    tiny = 1e-14
    admissible_face_state = config.weno_admissible_face_state
    positivity_preserving = config.weno_positivity_preserving
    # fields that carry no mass keep their own splitting speed
    own_speed_modes = mass_free_modes(config) if positivity_preserving else ()

    # Output block specs keep the conserved-variable axis complete and block only
    # the spatial dimensions.
    if ndim == 1:
        block_shape = (nvars, bx)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        block_shape = (nvars, bx, by)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        block_shape = (nvars, bx, by, bz)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(*refs):
        if has_g:
            q_ref, g_ref, gamma_ref, rhomin_ref, pgmin_ref, eta_ref, flux_out_ref = refs
        else:
            q_ref, gamma_ref, rhomin_ref, pgmin_ref, flux_out_ref = refs
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)

        if ndim == 1:
            ii = (bi * bx + jnp.arange(bx)) % nx
        elif ndim == 2:
            ii = (bi * bx + jnp.arange(bx)[:, None]) % nx
            jj = (bj * by + jnp.arange(by)[None, :]) % ny
        else:
            ii = (bi * bx + jnp.arange(bx)[:, None, None]) % nx
            jj = (bj * by + jnp.arange(by)[None, :, None]) % ny
            kk = (bk * bz + jnp.arange(bz)[None, None, :]) % nz

        gamma = gamma_ref[()]
        gm1 = gamma - 1.0
        rhomin = rhomin_ref[()]
        pgmin = pgmin_ref[()]
        if has_g:
            dual_eta = eta_ref[()]
            # typed tiny floor for the reliability test (x64/Triton dtype hygiene)
            e_floor = (gamma - gamma) + 1e-30

        def q_at(var_index: int, offset: int):
            if ndim == 1:
                return q_ref[var_index, (ii + offset) % nx]
            if ndim == 2:
                if axis == 0:
                    return q_ref[var_index, (ii + offset) % nx, jj]
                return q_ref[var_index, ii, (jj + offset) % ny]
            if axis == 0:
                return q_ref[var_index, (ii + offset) % nx, jj, kk]
            if axis == 1:
                return q_ref[var_index, ii, (jj + offset) % ny, kk]
            return q_ref[var_index, ii, jj, (kk + offset) % nz]

        def g_at(offset: int):
            if ndim == 1:
                return g_ref[0, (ii + offset) % nx]
            if ndim == 2:
                if axis == 0:
                    return g_ref[0, (ii + offset) % nx, jj]
                return g_ref[0, ii, (jj + offset) % ny]
            if axis == 0:
                return g_ref[0, (ii + offset) % nx, jj, kk]
            if axis == 1:
                return g_ref[0, ii, (jj + offset) % ny, kk]
            return g_ref[0, ii, jj, (kk + offset) % nz]

        def q_local(offset: int):
            return tuple(q_at(idx, offset) for idx in local_indices)

        def primitive_from_q(q, g=None):
            rho = q[0]
            mn = q[1]
            if ncomp == 3:
                mt1 = 0.0
                mt2 = 0.0
                energy = q[2]
            elif ncomp == 4:
                mt1 = q[2]
                mt2 = 0.0
                energy = q[3]
            else:
                mt1 = q[2]
                mt2 = q[3]
                energy = q[4]

            inv_rho = 1.0 / rho
            vn = mn * inv_rho
            vt1 = mt1 * inv_rho
            vt2 = mt2 * inv_rho
            v2 = vn * vn + vt1 * vt1 + vt2 * vt2
            e_E = energy - 0.5 * rho * v2
            if g is None:
                pressure = gm1 * e_E
            else:
                # Dual-energy switch (Bryan+95): use the advected g where the
                # total-energy internal energy is cancellation-unreliable.
                reliable = (e_E > dual_eta * jnp.maximum(energy, e_floor)) & (e_E == e_E)
                pressure = gm1 * jnp.where(reliable, e_E, g)
            return rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure

        def floored_cell(q, g=None):
            rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure = primitive_from_q(q, g)
            troubled = (rho < rhomin) | (pressure < pgmin)
            rho_f = jnp.where(troubled, jnp.maximum(rho, rhomin), rho)
            pressure_f = jnp.where(troubled, jnp.maximum(pressure, pgmin), pressure)
            energy_f = jnp.where(troubled, pressure_f / gm1 + 0.5 * rho_f * v2, energy)
            specific_enthalpy = (energy_f + pressure_f) / rho_f
            sound_speed = jnp.sqrt(jnp.maximum(gamma * jnp.abs(pressure_f / rho_f), 1e-12))
            return rho_f, mn, mt1, mt2, energy_f, vn, vt1, vt2, v2, pressure_f, specific_enthalpy, sound_speed

        def flux_from_q(q, g=None):
            rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure = primitive_from_q(q, g)
            if ncomp == 3:
                return (mn, mn * vn + pressure, (energy + pressure) * vn)
            if ncomp == 4:
                return (mn, mn * vn + pressure, mt1 * vn, (energy + pressure) * vn)
            return (mn, mn * vn + pressure, mt1 * vn, mt2 * vn, (energy + pressure) * vn)

        qm2 = q_local(-2)
        qm1 = q_local(-1)
        q0 = q_local(0)
        qp1 = q_local(1)
        qp2 = q_local(2)
        qp3 = q_local(3)
        q_stencil = (qm2, qm1, q0, qp1, qp2, qp3)
        if has_g:
            g_stencil = tuple(g_at(off) for off in range(-2, 4))
        else:
            g_stencil = (None,) * 6
        f_stencil = tuple(flux_from_q(q, g) for q, g in zip(q_stencil, g_stencil))

        # Compute floored primitive/eigenvalue data once for the six cells used
        # by the local Lax-Friedrichs alpha.  The earlier version recomputed this
        # data inside every characteristic mode; keeping it local here avoids both
        # global eigenvalue arrays and repeated per-mode work.
        floored_stencil = tuple(floored_cell(q, g) for q, g in zip(q_stencil, g_stencil))

        # Interface eigenvector building blocks at i + 1/2, following
        # _eigenvector_building_blocks in _eigen_hydro.py.
        cell_l = floored_stencil[2]
        cell_r = floored_stencil[3]
        rho_i, mn_i, mt1_i, mt2_i, energy_i, vn_i, vt1_i, vt2_i, v2_i, p_i, h_i, c_i = cell_l
        rho_j, mn_j, mt1_j, mt2_j, energy_j, vn_j, vt1_j, vt2_j, v2_j, p_j, h_j, c_j = cell_r
        rho_face = jnp.maximum(0.5 * (jnp.maximum(rho_i, rhomin) + jnp.maximum(rho_j, rhomin)), rhomin)
        vn_face = 0.5 * (mn_i + mn_j) / rho_face
        vt1_face = 0.5 * (mt1_i + mt1_j) / rho_face
        vt2_face = 0.5 * (mt2_i + mt2_j) / rho_face
        v2_face = vn_face * vn_face + vt1_face * vt1_face + vt2_face * vt2_face
        if admissible_face_state:
            # sound speed from the averaged pressure (see the native
            # _eigenvector_building_blocks): positive and frame independent
            c2_face = gamma * (0.5 * (p_i + p_j)) / rho_face
            h_face = c2_face / gm1 + 0.5 * v2_face
        else:
            h_face = 0.5 * (h_i + h_j)
            c2_face = gm1 * (h_face - 0.5 * v2_face)
        c_face = jnp.sqrt(jnp.maximum(c2_face, 1e-12))
        inv_c2 = jnp.where(c2_face > 0.0, 1.0 / c2_face, 0.0)

        def left_project(mode: int, values):
            """Project one local vector onto one Euler left eigenvector.

            This is the local Pallas replacement for materialising
            ``_eigen_L_row_hydro(..., mode)`` followed by a full-array einsum.
            ``values`` is either a local conserved-state vector or a local flux
            vector in the normal/tangential component order used by this axis.
            """
            if mode == 0:
                acc = (0.5 * gm1 * v2_face + vn_face * c_face) * values[0]
                acc = acc - (gm1 * vn_face + c_face) * values[1]
                if ncomp == 3:
                    acc = acc + gm1 * values[2]
                elif ncomp == 4:
                    acc = acc - gm1 * vt1_face * values[2] + gm1 * values[3]
                else:
                    acc = (
                        acc
                        - gm1 * vt1_face * values[2]
                        - gm1 * vt2_face * values[3]
                        + gm1 * values[4]
                    )
                return 0.5 * inv_c2 * acc

            if mode == 1:
                acc = (c2_face - 0.5 * gm1 * v2_face) * values[0]
                acc = acc + gm1 * vn_face * values[1]
                if ncomp == 3:
                    acc = acc - gm1 * values[2]
                elif ncomp == 4:
                    acc = acc + gm1 * vt1_face * values[2] - gm1 * values[3]
                else:
                    acc = (
                        acc
                        + gm1 * vt1_face * values[2]
                        + gm1 * vt2_face * values[3]
                        - gm1 * values[4]
                    )
                return inv_c2 * acc

            if mode == 2 and ncomp >= 4:
                return -vt1_face * values[0] + values[2]

            if mode == 3 and ncomp == 5:
                return -vt2_face * values[0] + values[3]

            # Right acoustic wave.
            acc = (0.5 * gm1 * v2_face - vn_face * c_face) * values[0]
            acc = acc - (gm1 * vn_face - c_face) * values[1]
            if ncomp == 3:
                acc = acc + gm1 * values[2]
            elif ncomp == 4:
                acc = acc - gm1 * vt1_face * values[2] + gm1 * values[3]
            else:
                acc = (
                    acc
                    - gm1 * vt1_face * values[2]
                    - gm1 * vt2_face * values[3]
                    + gm1 * values[4]
                )
            return 0.5 * inv_c2 * acc

        def add_right_correction(flux_acc, mode: int, Fs):
            """Add Fs times one local Euler right eigenvector to flux_acc.

            This is the local Pallas replacement for materialising
            ``_eigen_R_col_hydro(..., mode)`` followed by an outer-product style
            einsum.  Returning a Python list keeps the component axis static and
            avoids building small dense eigenvector arrays inside the kernel.
            """
            if mode == 0:
                if ncomp == 3:
                    R = (1.0, vn_face - c_face, h_face - vn_face * c_face)
                elif ncomp == 4:
                    R = (1.0, vn_face - c_face, vt1_face, h_face - vn_face * c_face)
                else:
                    R = (1.0, vn_face - c_face, vt1_face, vt2_face, h_face - vn_face * c_face)
            elif mode == 1:
                if ncomp == 3:
                    R = (1.0, vn_face, 0.5 * v2_face)
                elif ncomp == 4:
                    R = (1.0, vn_face, vt1_face, 0.5 * v2_face)
                else:
                    R = (1.0, vn_face, vt1_face, vt2_face, 0.5 * v2_face)
            elif mode == 2 and ncomp >= 4:
                if ncomp == 4:
                    R = (0.0, 0.0, 1.0, vt1_face)
                else:
                    R = (0.0, 0.0, 1.0, 0.0, vt1_face)
            elif mode == 3 and ncomp == 5:
                R = (0.0, 0.0, 0.0, 1.0, vt2_face)
            else:
                if ncomp == 3:
                    R = (1.0, vn_face + c_face, h_face + vn_face * c_face)
                elif ncomp == 4:
                    R = (1.0, vn_face + c_face, vt1_face, h_face + vn_face * c_face)
                else:
                    R = (1.0, vn_face + c_face, vt1_face, vt2_face, h_face + vn_face * c_face)
            return [flux_acc[slot] + R[slot] * Fs for slot in range(ncomp)]

        def lambda_from_floored_cell(cell, mode: int):
            vn = cell[5]
            c = cell[11]
            if mode == 0:
                return vn - c
            if mode == num_modes - 1:
                return vn + c
            return vn

        def alpha_for_mode(mode: int):
            amx = jnp.abs(lambda_from_floored_cell(floored_stencil[0], mode))
            for k in range(1, 6):
                amx = jnp.maximum(
                    amx,
                    jnp.abs(lambda_from_floored_cell(floored_stencil[k], mode)),
                )
            return amx

        flux_acc = [
            (-f_stencil[1][slot] + 7.0 * f_stencil[2][slot] + 7.0 * f_stencil[3][slot] - f_stencil[4][slot]) * (1.0 / 12.0)
            for slot in range(ncomp)
        ]

        if positivity_preserving:
            # One splitting speed (the stencil's spectral radius |v_n| + c) for
            # every field that carries mass, and the two split fluxes kept
            # apart, as in the native _weno_flux_x_native.
            common_speed = jnp.abs(floored_stencil[0][5]) + floored_stencil[0][11]
            for k in range(1, 6):
                common_speed = jnp.maximum(
                    common_speed, jnp.abs(floored_stencil[k][5]) + floored_stencil[k][11]
                )
            safe_speed = jnp.maximum(common_speed, 1e-30)
            central_state = [
                (-q_stencil[1][slot] + 7.0 * q_stencil[2][slot] + 7.0 * q_stencil[3][slot] - q_stencil[4][slot]) * (1.0 / 12.0)
                for slot in range(ncomp)
            ]
            plus_acc = [0.5 * (flux_acc[slot] + common_speed * central_state[slot]) for slot in range(ncomp)]
            minus_acc = [0.5 * (flux_acc[slot] - common_speed * central_state[slot]) for slot in range(ncomp)]
            plus_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]
            minus_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]


        for mode in range(num_modes):
            s = tuple(left_project(mode, f_stencil[k]) for k in range(6))
            qproj = tuple(left_project(mode, q_stencil[k]) for k in range(6))

            d0 = s[1] - s[0]
            d1 = s[2] - s[1]
            d2 = s[3] - s[2]
            d3 = s[4] - s[3]
            d4 = s[5] - s[4]

            dq0 = qproj[1] - qproj[0]
            dq1 = qproj[2] - qproj[1]
            dq2 = qproj[3] - qproj[2]
            dq3 = qproj[4] - qproj[3]
            dq4 = qproj[5] - qproj[4]

            # (under PP the mass-carrying fields use the common speed: skip their
            # own stencil maximum, which would only be computed and discarded)
            if positivity_preserving and mode not in own_speed_modes:
                amx = common_speed
            else:
                amx = alpha_for_mode(mode)

            aterm_p = 0.5 * (d0 + amx * dq0)
            bterm_p = 0.5 * (d1 + amx * dq1)
            cterm_p = 0.5 * (d2 + amx * dq2)
            dterm_p = 0.5 * (d3 + amx * dq3)

            IS0_p = 13.0 * (aterm_p - bterm_p) ** 2 + 3.0 * (aterm_p - 3.0 * bterm_p) ** 2
            IS1_p = 13.0 * (bterm_p - cterm_p) ** 2 + 3.0 * (bterm_p + cterm_p) ** 2
            IS2_p = 13.0 * (cterm_p - dterm_p) ** 2 + 3.0 * (3.0 * cterm_p - dterm_p) ** 2
            omega0_p, omega2_p = omega_weights(IS0_p, IS1_p, IS2_p, epsilon, tiny)
            second = (
                omega0_p * (aterm_p - 2.0 * bterm_p + cterm_p) * (1.0 / 3.0)
                + (omega2_p - 0.5) * (bterm_p - 2.0 * cterm_p + dterm_p) * (1.0 / 6.0)
            )

            aterm_m = 0.5 * (d4 - amx * dq4)
            bterm_m = 0.5 * (d3 - amx * dq3)
            cterm_m = 0.5 * (d2 - amx * dq2)
            dterm_m = 0.5 * (d1 - amx * dq1)

            IS0_m = 13.0 * (aterm_m - bterm_m) ** 2 + 3.0 * (aterm_m - 3.0 * bterm_m) ** 2
            IS1_m = 13.0 * (bterm_m - cterm_m) ** 2 + 3.0 * (bterm_m + cterm_m) ** 2
            IS2_m = 13.0 * (cterm_m - dterm_m) ** 2 + 3.0 * (3.0 * cterm_m - dterm_m) ** 2
            omega0_m, omega2_m = omega_weights(IS0_m, IS1_m, IS2_m, epsilon, tiny)
            third = (
                omega0_m * (aterm_m - 2.0 * bterm_m + cterm_m) * (1.0 / 3.0)
                + (omega2_m - 0.5) * (bterm_m - 2.0 * cterm_m + dterm_m) * (1.0 / 6.0)
            )

            if positivity_preserving:
                zero_acc = [plus_acc[0] * 0.0 for _ in range(ncomp)]
                plus_acc = add_right_correction(plus_acc, mode, -second)
                minus_acc = add_right_correction(minus_acc, mode, third)
                # a field on its own speed also shifts the central part and
                # the upwind cells' split states along its eigenvector
                if mode in own_speed_modes:
                    speed_offset = amx - common_speed
                    central_projection = (
                        -qproj[1] + 7.0 * qproj[2] + 7.0 * qproj[3] - qproj[4]
                    ) * (1.0 / 12.0)
                    central_shift = add_right_correction(zero_acc, mode, 0.5 * speed_offset * central_projection)
                    plus_acc = [plus_acc[slot] + central_shift[slot] for slot in range(ncomp)]
                    minus_acc = [minus_acc[slot] - central_shift[slot] for slot in range(ncomp)]
                    relative_offset = speed_offset / safe_speed
                    plus_shift = add_right_correction(plus_shift, mode, relative_offset * qproj[2])
                    minus_shift = add_right_correction(minus_shift, mode, relative_offset * qproj[3])
                continue

            Fs = -second + third
            flux_acc = add_right_correction(flux_acc, mode, Fs)

        if positivity_preserving:
            flux_acc = positivity_preserving_flux_local(
                q_stencil[2], q_stencil[3], f_stencil[2], f_stencil[3],
                plus_acc, minus_acc, plus_shift, minus_shift,
                common_speed, gm1, rhomin, pgmin,
            )

        # Set every output component.  Hydro should fill all components, but the
        # explicit zeroing makes failures obvious if a future registry adds fields.
        zero = flux_acc[0] * 0.0
        for var in range(nvars):
            flux_out_ref[var, ...] = zero
        for slot, var in enumerate(local_indices):
            flux_out_ref[var, ...] = flux_acc[slot]

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    if has_g:
        if ndim == 1:
            in_g_spec = pl.BlockSpec(internal_energy_density.shape, lambda bi, bj, bk: (0, 0))
        elif ndim == 2:
            in_g_spec = pl.BlockSpec(internal_energy_density.shape, lambda bi, bj, bk: (0, 0, 0))
        else:
            in_g_spec = pl.BlockSpec(internal_energy_density.shape, lambda bi, bj, bk: (0, 0, 0, 0))
        in_specs = [in_state_spec, in_g_spec, scalar_spec, scalar_spec, scalar_spec, scalar_spec]
        args = (
            conserved_state,
            internal_energy_density,
            jnp.asarray(params.gamma, dtype=conserved_state.dtype),
            jnp.asarray(params.minimum_density, dtype=conserved_state.dtype),
            jnp.asarray(params.minimum_pressure, dtype=conserved_state.dtype),
            jnp.asarray(config.dual_energy_eta, dtype=conserved_state.dtype),
        )
    else:
        in_specs = [in_state_spec, scalar_spec, scalar_spec, scalar_spec]
        args = (
            conserved_state,
            jnp.asarray(params.gamma, dtype=conserved_state.dtype),
            jnp.asarray(params.minimum_density, dtype=conserved_state.dtype),
            jnp.asarray(params.minimum_pressure, dtype=conserved_state.dtype),
        )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name=f"hydro_weno_flux_axis_{axis}" + ("_dual" if has_g else ""),
        **kwargs,
    )(*args)


# -----------------------------------------------------------------------------
# Pallas WENO for the ideal-gas MHD equations.
# -----------------------------------------------------------------------------


def _mhd_pallas_flux_supported(conserved_state, config: SimulationConfig) -> bool:
    """Whether the Pallas MHD ideal-gas WENO kernel can be used."""
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if not config.mhd:
        return False
    if config.equation_of_state != IDEAL_GAS:
        return False  # isothermal MHD WENO Pallas kernel still TODO (guide §4.2)
    ndim = int(config.dimensionality)
    if ndim != 3:  # MHD WENO is 3D-only in this codebase
        return False
    if conserved_state.ndim != 4:
        return False
    block_shape = _as_3tuple_block_shape(config.backend_config.pallas_block_shape, ndim, spatial_shape=conserved_state.shape[1:])
    for n, b in zip(conserved_state.shape[1:], block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _mhd_indices_for_axis(config: SimulationConfig, registered_variables: RegisteredVariables, axis: int):
    """Local conserved-variable order used by the MHD eigenvectors for a flux
    normal to ``axis``: (density, p_normal, p_trans1, p_trans2, B_normal,
    B_trans1, B_trans2, energy).  Returns the 8 indices into the original
    conserved-state component axis in that order.
    """
    density_index = int(registered_variables.density_index)
    energy_index = int(registered_variables.energy_index)
    mx = int(registered_variables.momentum_index.x)
    my = int(registered_variables.momentum_index.y)
    mz = int(registered_variables.momentum_index.z)
    bx = int(registered_variables.magnetic_index.x)
    by = int(registered_variables.magnetic_index.y)
    bz = int(registered_variables.magnetic_index.z)

    if axis == 0:
        return (density_index, mx, my, mz, bx, by, bz, energy_index)
    if axis == 1:
        # Matches native ``_weno_flux_y_native``: swap mom_x↔mom_y, B_x↔B_y.
        return (density_index, my, mx, mz, by, bx, bz, energy_index)
    # axis == 2 — matches native ``_weno_flux_z_native`` transpose
    # (0, 3, 2, 1) followed by mom_x↔mom_z and B_x↔B_z swap.
    return (density_index, mz, my, mx, bz, by, bx, energy_index)


def _weno_flux_mhd_pallas(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
    internal_energy_density=None,
    inflow_reference=None,
):
    """Pallas implementation of the ideal-gas MHD WENO interface flux.

    Public entry point: dispatches the supported-predicate check and the
    multi-GPU ``shard_map`` + halo wrap.  Kernel arithmetic in
    ``_weno_flux_mhd_pallas_local``.

    ``internal_energy_density`` (dual-energy ``g``, cell-centred scalar
    field, UNTRANSPOSED) switches the pressure recovery in the flux and
    eigenstructure; it rides the same halo exchange as the state.
    """
    if not _mhd_pallas_flux_supported(conserved_state, config):
        # Lazy import to break the circular dependency with _weno.py.
        from astronomix._finite_difference._interface_fluxes._weno import (
            _weno_flux_x_native, _weno_flux_y_native, _weno_flux_z_native,
        )
        native = [_weno_flux_x_native, _weno_flux_y_native, _weno_flux_z_native][axis]
        return native(conserved_state, params, config, registered_variables,
                      internal_energy_density=internal_energy_density,
                      inflow_reference=inflow_reference)

    # the paired positivity-preserving recombination reads interfaces i +- 1
    halo_cells = 4 if config.weno_positivity_preserving else 3
    if internal_energy_density is not None:
        g4 = internal_energy_density[None]

        def _local_dual(state_local, g_local):
            return _weno_flux_mhd_pallas_local(
                state_local, params, config, registered_variables, axis=axis,
                internal_energy_density=g_local,
            )
        return _weno5_shard_wrap(_local_dual, conserved_state, config, axis,
                                 extra_state_inputs=(g4,), halo_cells=halo_cells)

    if inflow_reference is not None:
        # the joint inflow limiting reads the reference at cells i and i + 1
        reference_state, speed_sum = inflow_reference

        def _local_joint(state_local, reference_local, speed_sum_local):
            return _weno_flux_mhd_pallas_local(
                state_local, params, config, registered_variables, axis=axis,
                inflow_reference=(reference_local, speed_sum_local[0]),
            )
        return _weno5_shard_wrap(_local_joint, conserved_state, config, axis,
                                 extra_state_inputs=(reference_state, speed_sum[None]),
                                 halo_cells=halo_cells)

    def _local(state_local):
        return _weno_flux_mhd_pallas_local(
            state_local, params, config, registered_variables, axis=axis
        )
    return _weno5_shard_wrap(_local, conserved_state, config, axis, halo_cells=halo_cells)


def _weno_flux_mhd_pallas_keep_halo_x(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The x-direction ideal-MHD WENO flux for a mesh that splits x, returned
    twice: once with one kept x-halo cell (for the x divergence, which then
    needs no second exchange) and once stripped (for the local consumers: the
    CT magnetic-flux slices and the density flux).

    Only the plain WENO flux is implemented (no positivity-preserving
    recombination, no dual energy); the multi-GPU fast path of
    ``_lsrk4_with_ct`` is gated accordingly.

    Args:
        conserved_state: The conserved state.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        ``(flux_with_x_halo, flux)``.
    """
    if not _mhd_pallas_flux_supported(conserved_state, config):
        raise RuntimeError(
            "_weno_flux_mhd_pallas_keep_halo_x needs the Pallas MHD WENO kernel."
        )

    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape,
        ndim,
        spatial_shape=conserved_state.shape[1:],
    )
    # The WENO5 stencil reaches three cells along the flux axis; the kept left
    # face F_{-1/2} of a shard needs cells -3..2, all inside that halo.
    halo = (3, 0, 0)

    def _local(state_local):
        flux = _weno_flux_mhd_pallas_local(
            state_local, params, config, registered_variables, axis=0
        )
        return flux, flux

    return _pallas_call_sharded(
        _local,
        state_inputs=(conserved_state,),
        halo=halo,
        block_shape=block_shape[:ndim],
        num_state_outputs=2,
        output_halo=((1, 0, 0), (0, 0, 0)),
    )


def _update_cell_center_and_weno_flux_mhd_pallas_keep_halo_x_with_ct_mod(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Fused first part of a multi-GPU constrained-transport stage along a split
    x axis: the cell-centred magnetic field update from the interface fields,
    the x-direction WENO flux (with and without one kept x-halo cell, see
    ``_weno_flux_mhd_pallas_keep_halo_x``) and the two x-direction modified
    fluxes of constrained transport, in one shard_map, so the x halo of the
    stage is exchanged once.

    The cell-centred update ``B = f2c(B_face)``, ``E += (B_new^2 - B_old^2)/2``
    and the modified fluxes ``F(B_t) + c2f_x(B_x v_t)`` are the expressions of
    ``_ct_update_cell_center_fields_pallas_local`` and
    ``_ct_modified_flux_pallas_local``, evaluated in JAX on the halo-padded
    shard. Plain WENO flux only (no positivity-preserving recombination, no
    dual energy).

    Args:
        conserved_state: The conserved state (cell-centred B not yet updated).
        bx_interface: The x-face magnetic field.
        by_interface: The y-face magnetic field.
        bz_interface: The z-face magnetic field.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        ``(updated_state, flux_with_x_halo, flux, flux_x_modified)`` with
        ``flux_x_modified`` the stacked (B_y, B_z) modified x fluxes.
    """
    if not _mhd_pallas_flux_supported(conserved_state, config):
        raise RuntimeError(
            "The fused cell-centre update and x-WENO flux need the Pallas MHD WENO kernel."
        )

    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape,
        ndim,
        spatial_shape=conserved_state.shape[1:],
    )
    density = int(registered_variables.density_index)
    mom_y = int(registered_variables.momentum_index.y)
    mom_z = int(registered_variables.momentum_index.z)
    mag_x = int(registered_variables.magnetic_index.x)
    mag_y = int(registered_variables.magnetic_index.y)
    mag_z = int(registered_variables.magnetic_index.z)
    energy = int(registered_variables.pressure_index)

    def _local(state_local, bx_local, by_local, bz_local):
        bx_i = bx_local[0]
        by_i = by_local[0]
        bz_i = bz_local[0]

        def f2c_x(a):
            return (
                3.0 * jnp.roll(a, 3, axis=0)
                - 25.0 * jnp.roll(a, 2, axis=0)
                + 150.0 * jnp.roll(a, 1, axis=0)
                + 150.0 * a
                - 25.0 * jnp.roll(a, -1, axis=0)
                + 3.0 * jnp.roll(a, -2, axis=0)
            ) / 256.0

        def f2c_y(a):
            return (
                3.0 * jnp.roll(a, 3, axis=1)
                - 25.0 * jnp.roll(a, 2, axis=1)
                + 150.0 * jnp.roll(a, 1, axis=1)
                + 150.0 * a
                - 25.0 * jnp.roll(a, -1, axis=1)
                + 3.0 * jnp.roll(a, -2, axis=1)
            ) / 256.0

        def f2c_z(a):
            return (
                3.0 * jnp.roll(a, 3, axis=2)
                - 25.0 * jnp.roll(a, 2, axis=2)
                + 150.0 * jnp.roll(a, 1, axis=2)
                + 150.0 * a
                - 25.0 * jnp.roll(a, -1, axis=2)
                + 3.0 * jnp.roll(a, -2, axis=2)
            ) / 256.0

        bx_center = f2c_x(bx_i)
        by_center = f2c_y(by_i)
        bz_center = f2c_z(bz_i)
        b2_old = (
            state_local[mag_x] * state_local[mag_x]
            + state_local[mag_y] * state_local[mag_y]
            + state_local[mag_z] * state_local[mag_z]
        )
        b2_new = bx_center * bx_center + by_center * by_center + bz_center * bz_center
        state_updated = state_local.at[mag_x].set(bx_center)
        state_updated = state_updated.at[mag_y].set(by_center)
        state_updated = state_updated.at[mag_z].set(bz_center)
        state_updated = state_updated.at[energy].set(
            state_local[energy] + 0.5 * (b2_new - b2_old)
        )

        flux = _weno_flux_mhd_pallas_local(
            state_updated, params, config, registered_variables, axis=0
        )
        rho = state_updated[density]
        bx = state_updated[mag_x]
        bx_vy = bx * state_updated[mom_y] / rho
        bx_vz = bx * state_updated[mom_z] / rho

        def c2f_x(a):
            return (
                -jnp.roll(a, 1, axis=0)
                + 9.0 * a
                + 9.0 * jnp.roll(a, -1, axis=0)
                - jnp.roll(a, -2, axis=0)
            ) / 16.0

        flux_x_mod = jnp.stack([
            flux[mag_y] + c2f_x(bx_vy),
            flux[mag_z] + c2f_x(bx_vz),
        ])
        return state_updated, flux, flux, flux_x_mod

    q_halo = (3, 0, 0)[:ndim]
    bx_halo = (6, 0, 0)[:ndim]
    by_halo = (3, 3, 0)[:ndim]
    bz_halo = (3, 0, 3)[:ndim]
    shape_halo = tuple(
        max(vals) for vals in zip(q_halo, bx_halo, by_halo, bz_halo, strict=True)
    )

    return _pallas_call_sharded(
        _local,
        state_inputs=(
            conserved_state,
            bx_interface[None],
            by_interface[None],
            bz_interface[None],
        ),
        halo=shape_halo,
        block_shape=block_shape[:ndim],
        input_halos=(q_halo, bx_halo, by_halo, bz_halo),
        num_state_outputs=4,
        output_halo=((0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 0)),
    )


def _weno_mhd_flux_from_window(q_stencil, gamma, rhomin, pgmin, b_eps, sqrt_floor,
                              ncomp, num_modes, use_approx_rsqrt=False,
                              g_stencil=None, dual_eta=None,
                              admissible_face_state=False, positivity_preserving=False,
                              return_split=False):
    """Pure per-interface ideal-gas MHD WENO flux from a gathered 6-cell stencil.

    With ``positivity_preserving`` and ``return_split`` it returns the two
    split face fluxes and the splitting speed instead, for the paired
    recombination, which needs the neighbouring interfaces too.

    ``q_stencil`` is the tuple ``(q[-2], q[-1], q[0], q[+1], q[+2], q[+3])`` where
    each entry is a length-8 tuple of the local conserved components in per-axis
    characteristic order ``(rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy)``.  Returns
    the length-8 list of WENO interface fluxes ``flux_acc`` at ``i + 1/2``.

    Single source of truth for the MHD WENO arithmetic: the forward Pallas kernel
    gathers ``q_stencil`` from ``q_ref`` and calls this; the adjoint kernel
    gathers the same stencil and calls ``jax.vjp`` of this, so the Pallas
    backward is the exact transpose of the Pallas forward by construction (no
    separately-derived adjoint math).  Every operation is elementwise on the
    gathered arrays — no ref reads, slices or rolls — which is what lets
    ``jax.vjp`` lower inside the Triton kernel (validated bit-exact and
    compile-at-parity on jax >= 0.10; the old auto-VJP Triton miscompile that
    forced the hydro hand-derivation is gone).  ``b_eps`` and ``sqrt_floor`` are
    passed in as already-typed scalars (x64 + Triton dtype hygiene)."""
    gm1 = gamma - 1.0
    gam0 = 1.0 - gamma   # = -gm1
    gam1 = 0.5 * (gamma - 1.0)
    gam2 = (gamma - 2.0) / (gamma - 1.0)
    epsilon = 1e-7
    tiny = 1e-14
    # Properly-typed literal scalars derived from gamma so the dtype follows the
    # working dtype (bare 1.0 / -1.0 / 1/sqrt(2) arrive as f32 under x64 + Triton
    # and trip a ('f64','f32') assertion in _truediv_lowering_rule).
    zero_typed = gamma - gamma
    one_typed = zero_typed + 1.0
    neg_one_typed = zero_typed - 1.0
    inv_sqrt_two_typed = zero_typed + (1.0 / 2.0 ** 0.5)
    # AD-safe sqrt for non-negative-clamped quantities — mirrors the native
    # ``_eigen_mhd.diff_safe_sqrt``: a *positive* floor so the reverse pass never
    # forms ``sqrt'(0) = inf``.  ``jnp.sqrt(jnp.maximum(x, 0.0))`` is value-safe
    # but its gradient is ``inf`` at the clamp; under reverse-mode that ``inf``
    # meets a ``0`` cotangent and the in-kernel/multi-step backward yields NaN
    # (XLA folds inf*0->0 for a single call, which is why a one-shot VJP looked
    # clean).  The floor is below any physical value, so the forward stays
    # bit-exact with the native flux (which floors the same way).
    sqrt_eps = zero_typed + (1e-30 if jax.config.jax_enable_x64 else 1e-20)

    # ``use_approx_rsqrt`` is a static (Python-bool) build flag, so this is a
    # trace-time branch, not a runtime one.  When on, sqrt(s) is computed as
    # ``s * jax.lax.rsqrt(s)`` -> ``rsqrt.approx.f64`` (refined to ~1 ULP by
    # __nv_rsqrt), ~1.6x cheaper than ``sqrt.rn.f64``'s IEEE expansion and, by
    # halving spill traffic, ~1.77x on the full dp step (A100).  s is floored
    # > 0 so ``s * rsqrt(s) == sqrt(s)``.  Forward only: the hand adjoint below
    # keeps IEEE sqrt (finalize_config warns about the ~1 ULP AD mismatch).
    def ssqrt(x):
        s = jnp.maximum(x, sqrt_eps)
        return s * jax.lax.rsqrt(s) if use_approx_rsqrt else jnp.sqrt(s)

    # Dual-energy (Bryan+95): ``g_stencil`` pairs each stencil cell with its
    # advected internal-energy density; the switch replaces the total-energy
    # internal energy where it is cancellation-unreliable, mirroring the
    # native ``dual_switched_pressure`` / ``_eigen_mhd`` threading.
    has_g = g_stencil is not None
    if has_g:
        e_floor = zero_typed + 1e-30
    else:
        g_stencil = (None,) * len(q_stencil)

    def primitive_from_q(q, g=None):
        rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy = q
        inv_rho = 1.0 / rho
        vn = mn * inv_rho
        vt1 = mt1 * inv_rho
        vt2 = mt2 * inv_rho
        v2 = vn * vn + vt1 * vt1 + vt2 * vt2
        b2 = Bn * Bn + Bt1 * Bt1 + Bt2 * Bt2
        e_E = energy - 0.5 * (rho * v2 + b2)
        if g is None:
            p = gm1 * e_E
        else:
            reliable = (e_E > dual_eta * jnp.maximum(energy, e_floor)) & (e_E == e_E)
            p = gm1 * jnp.where(reliable, e_E, g)
        return rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy, vn, vt1, vt2, v2, b2, p

    def floored_cell(q, g=None):
        rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy, vn, vt1, vt2, v2, b2, p = primitive_from_q(q, g)
        troubled = (rho < rhomin) | (p < pgmin)
        rho_f = jnp.where(troubled, jnp.maximum(rho, rhomin), rho)
        p_f = jnp.where(troubled, jnp.maximum(p, pgmin), p)
        energy_f = jnp.where(troubled, p_f / gm1 + 0.5 * (rho_f * v2 + b2), energy)
        # MHD enthalpy includes the magnetic contribution implicitly via
        # the (energy + p_gas) / rho average used in the native code.
        specific_enthalpy = (energy_f + p_f) / rho_f
        sound_speed_sq = jnp.maximum(0.0, gamma * jnp.abs(p_f / rho_f))
        sound_speed = jnp.sqrt(jnp.maximum(sound_speed_sq, sqrt_floor))
        # MHD characteristic speeds (cell-centered — used for the local
        # Lax-Friedrichs alpha; the FACE eigenstructure is computed
        # separately further down).
        bn2_over_rho = (Bn * Bn) / rho_f
        disc_root = ssqrt(
            (b2 / rho_f + sound_speed_sq) ** 2 - 4.0 * bn2_over_rho * sound_speed_sq
        )
        c_fast = ssqrt(0.5 * (b2 / rho_f + sound_speed_sq + disc_root))
        c_alfven = ssqrt(bn2_over_rho)
        c_slow = ssqrt(0.5 * (b2 / rho_f + sound_speed_sq - disc_root))
        return (rho_f, mn, mt1, mt2, Bn, Bt1, Bt2, energy_f,
                vn, vt1, vt2, v2, b2, p_f, specific_enthalpy,
                sound_speed, sound_speed_sq, c_fast, c_alfven, c_slow)

    def flux_from_q(q, g=None):
        """MHD flux along the normal direction (local x).  B_normal flux
        is identically zero (see ``_mhd_flux_x``)."""
        rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy, vn, vt1, vt2, v2, b2, p = primitive_from_q(q, g)
        p_total = p + 0.5 * b2
        v_dot_B = vn * Bn + vt1 * Bt1 + vt2 * Bt2
        return (
            mn,                                  # density flux: rho * vn
            rho * vn * vn + p_total - Bn * Bn,    # normal momentum
            rho * vn * vt1 - Bn * Bt1,            # transverse 1
            rho * vn * vt2 - Bn * Bt2,            # transverse 2
            0.0,                                  # normal B flux is 0
            Bt1 * vn - Bn * vt1,                  # transverse 1 B
            Bt2 * vn - Bn * vt2,                  # transverse 2 B
            (energy + p_total) * vn - v_dot_B * Bn,  # energy
        )

    def lambda_from_floored_cell(cell, mode):
        vn = cell[8]; c_fast = cell[17]; c_alfven = cell[18]; c_slow = cell[19]
        if mode == 0:
            return vn - c_fast
        if mode == 1:
            return vn - c_alfven
        if mode == 2:
            return vn - c_slow
        if mode == 3:
            return vn
        if mode == 4:
            return vn + c_slow
        if mode == 5:
            return vn + c_alfven
        return vn + c_fast

    f_stencil = tuple(flux_from_q(q, g) for q, g in zip(q_stencil, g_stencil))
    floored_stencil = tuple(floored_cell(q, g) for q, g in zip(q_stencil, g_stencil))
    cell_l = floored_stencil[2]  # offset 0  (cell i)
    cell_r = floored_stencil[3]  # offset 1  (cell i+1)

    rho_i = cell_l[0]; mn_i = cell_l[1]; mt1_i = cell_l[2]; mt2_i = cell_l[3]
    Bn_i = cell_l[4]; Bt1_i = cell_l[5]; Bt2_i = cell_l[6]
    h_i = cell_l[14]
    rho_j = cell_r[0]; mn_j = cell_r[1]; mt1_j = cell_r[2]; mt2_j = cell_r[3]
    Bn_j = cell_r[4]; Bt1_j = cell_r[5]; Bt2_j = cell_r[6]
    h_j = cell_r[14]

    rho_face = jnp.maximum(
        0.5 * (jnp.maximum(rho_i, rhomin) + jnp.maximum(rho_j, rhomin)),
        rhomin,
    )
    vn_face = 0.5 * (mn_i + mn_j) / rho_face
    vt1_face = 0.5 * (mt1_i + mt1_j) / rho_face
    vt2_face = 0.5 * (mt2_i + mt2_j) / rho_face
    Bn_face = 0.5 * (Bn_i + Bn_j)
    Bt1_face = 0.5 * (Bt1_i + Bt1_j)
    Bt2_face = 0.5 * (Bt2_i + Bt2_j)
    h_face = 0.5 * (h_i + h_j)

    v2_face = vn_face * vn_face + vt1_face * vt1_face + vt2_face * vt2_face
    b2_face = Bn_face * Bn_face + Bt1_face * Bt1_face + Bt2_face * Bt2_face
    b2_over_rho_face = b2_face / rho_face
    bn2_over_rho_face = (Bn_face * Bn_face) / rho_face

    c_sq_face = gm1 * (h_face - 0.5 * (v2_face + b2_over_rho_face))
    if admissible_face_state:
        # sound speed from the averaged pressure (see the native
        # _eigenvector_building_blocks): positive and frame independent
        c_sq_face = gamma * (0.5 * (cell_l[13] + cell_r[13])) / rho_face
        h_face = c_sq_face / gm1 + 0.5 * (v2_face + b2_over_rho_face)
    c_sq_face = jnp.maximum(c_sq_face, 0.0)
    c_face = jnp.sqrt(jnp.maximum(c_sq_face, sqrt_floor))
    c_sq_safe = jnp.where(c_sq_face > 0.0, c_sq_face, one_typed)
    inv_c_sq = jnp.where(c_sq_face > 0.0, 1.0 / c_sq_safe, 0.0)

    ms_disc = (b2_over_rho_face + c_sq_face) ** 2 - 4.0 * bn2_over_rho_face * c_sq_face
    ms_disc_root = ssqrt(ms_disc)

    lambda_fast = ssqrt(0.5 * (b2_over_rho_face + c_sq_face + ms_disc_root))
    lambda_alfven = ssqrt(bn2_over_rho_face)
    lambda_slow = ssqrt(0.5 * (b2_over_rho_face + c_sq_face - ms_disc_root))

    # Tangential normalisation with the degeneracy fix.
    bt_sq = Bt1_face * Bt1_face + Bt2_face * Bt2_face
    bt_sq_safe = jnp.maximum(bt_sq, b_eps)
    bt_n1 = jnp.where(
        bt_sq >= b_eps,
        Bt1_face / jnp.sqrt(bt_sq_safe),
        inv_sqrt_two_typed,
    )
    bt_n2 = jnp.where(
        bt_sq >= b_eps,
        Bt2_face / jnp.sqrt(bt_sq_safe),
        inv_sqrt_two_typed,
    )

    sgn_bn = jnp.where(Bn_face >= 0.0, one_typed, neg_one_typed)
    sgn_bt = jnp.where(
        Bt1_face != 0.0,
        jnp.where(Bt1_face >= 0.0, one_typed, neg_one_typed),
        jnp.where(Bt2_face >= 0.0, one_typed, neg_one_typed),
    )

    # Fast / slow mode weighting; same algebra as the native helper.
    denom = lambda_fast * lambda_fast - lambda_slow * lambda_slow
    denom_safe = jnp.maximum(denom, b_eps)
    am_fast = jnp.where(
        denom >= b_eps,
        ssqrt(c_sq_face - lambda_slow * lambda_slow) / jnp.sqrt(denom_safe),
        1.0,
    )
    am_slow = jnp.where(
        denom >= b_eps,
        ssqrt(lambda_fast * lambda_fast - c_sq_face) / jnp.sqrt(denom_safe),
        1.0,
    )

    sqrt_rho_face = jnp.sqrt(jnp.maximum(rho_face, rhomin))
    cs_geq_alfven = c_face >= lambda_alfven

    def left_project(mode, values):
        """L_row[mode] · values.  ``values`` is an 8-tuple in local order:
        (rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy)."""
        rho_v, mn_v, mt1_v, mt2_v, Bn_v, Bt1_v, Bt2_v, e_v = values
        if mode == 0:  # fast-
            L_rho = (
                am_fast * (gam1 * v2_face + lambda_fast * vn_face)
                - am_slow * lambda_slow * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
            )
            L_mn = am_fast * (gam0 * vn_face - lambda_fast)
            L_mt1 = gam0 * am_fast * vt1_face + am_slow * lambda_slow * bt_n1 * sgn_bn
            L_mt2 = gam0 * am_fast * vt2_face + am_slow * lambda_slow * bt_n2 * sgn_bn
            L_Bt1 = gam0 * am_fast * Bt1_face + c_face * am_slow * bt_n1 * sqrt_rho_face
            L_Bt2 = gam0 * am_fast * Bt2_face + c_face * am_slow * bt_n2 * sqrt_rho_face
            L_E = -gam0 * am_fast
            acc = (
                L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v + L_E * e_v
            )
            acc = 0.5 * acc * inv_c_sq
            return jnp.where(~cs_geq_alfven, acc * sgn_bt, acc)
        if mode == 1:  # alfvén-
            L_rho = bt_n2 * vt1_face - bt_n1 * vt2_face
            L_mt1 = -bt_n2
            L_mt2 = bt_n1
            L_Bt1 = -bt_n2 * sgn_bn * sqrt_rho_face
            L_Bt2 = bt_n1 * sgn_bn * sqrt_rho_face
            acc = (
                L_rho * rho_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v
            )
            return 0.5 * acc
        if mode == 2:  # slow-
            L_rho = (
                am_slow * (gam1 * v2_face + lambda_slow * vn_face)
                + am_fast * lambda_fast * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
            )
            L_mn = am_slow * (gam0 * vn_face) - am_slow * lambda_slow
            L_mt1 = gam0 * am_slow * vt1_face - am_fast * lambda_fast * bt_n1 * sgn_bn
            L_mt2 = gam0 * am_slow * vt2_face - am_fast * lambda_fast * bt_n2 * sgn_bn
            L_Bt1 = gam0 * am_slow * Bt1_face - c_face * am_fast * bt_n1 * sqrt_rho_face
            L_Bt2 = gam0 * am_slow * Bt2_face - c_face * am_fast * bt_n2 * sqrt_rho_face
            L_E = -gam0 * am_slow
            acc = (
                L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v + L_E * e_v
            )
            acc = 0.5 * acc * inv_c_sq
            return jnp.where(cs_geq_alfven, acc * sgn_bt, acc)
        if mode == 3:  # entropy
            L_rho = -c_sq_face / gam0 - 0.5 * v2_face
            L_mn = vn_face
            L_mt1 = vt1_face
            L_mt2 = vt2_face
            L_Bt1 = Bt1_face
            L_Bt2 = Bt2_face
            L_E = -1.0
            acc = (
                L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v + L_E * e_v
            )
            return -gam0 * acc * inv_c_sq
        if mode == 4:  # slow+
            L_rho = (
                am_slow * (gam1 * v2_face - lambda_slow * vn_face)
                - am_fast * lambda_fast * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
            )
            L_mn = am_slow * (gam0 * vn_face + lambda_slow)
            L_mt1 = gam0 * am_slow * vt1_face + am_fast * lambda_fast * bt_n1 * sgn_bn
            L_mt2 = gam0 * am_slow * vt2_face + am_fast * lambda_fast * bt_n2 * sgn_bn
            L_Bt1 = gam0 * am_slow * Bt1_face - c_face * am_fast * bt_n1 * sqrt_rho_face
            L_Bt2 = gam0 * am_slow * Bt2_face - c_face * am_fast * bt_n2 * sqrt_rho_face
            L_E = -gam0 * am_slow
            acc = (
                L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v + L_E * e_v
            )
            acc = 0.5 * acc * inv_c_sq
            return jnp.where(cs_geq_alfven, acc * sgn_bt, acc)
        if mode == 5:  # alfvén+
            L_rho = bt_n2 * vt1_face - bt_n1 * vt2_face
            L_mt1 = -bt_n2
            L_mt2 = bt_n1
            L_Bt1 = bt_n2 * sgn_bn * sqrt_rho_face
            L_Bt2 = -bt_n1 * sgn_bn * sqrt_rho_face
            acc = (
                L_rho * rho_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v
            )
            return 0.5 * acc
        # mode 6 — fast+
        L_rho = (
            am_fast * (gam1 * v2_face - lambda_fast * vn_face)
            + am_slow * lambda_slow * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
        )
        L_mn = am_fast * (gam0 * vn_face + lambda_fast)
        L_mt1 = gam0 * am_fast * vt1_face - am_slow * lambda_slow * bt_n1 * sgn_bn
        L_mt2 = gam0 * am_fast * vt2_face - am_slow * lambda_slow * bt_n2 * sgn_bn
        L_Bt1 = gam0 * am_fast * Bt1_face + c_face * am_slow * bt_n1 * sqrt_rho_face
        L_Bt2 = gam0 * am_fast * Bt2_face + c_face * am_slow * bt_n2 * sqrt_rho_face
        L_E = -gam0 * am_fast
        acc = (
            L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
            + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v + L_E * e_v
        )
        acc = 0.5 * acc * inv_c_sq
        return jnp.where(~cs_geq_alfven, acc * sgn_bt, acc)

    def add_right_correction(flux_acc, mode, Fs):
        """flux_acc += Fs * R_col[:, mode] (local order, ncomp=8).
        B_normal slot (index 4) always gets 0."""
        if mode == 0:  # fast-
            R = (
                am_fast,
                am_fast * (vn_face - lambda_fast),
                am_fast * vt1_face + am_slow * lambda_slow * bt_n1 * sgn_bn,
                am_fast * vt2_face + am_slow * lambda_slow * bt_n2 * sgn_bn,
                0.0,
                c_face * am_slow * bt_n1 / sqrt_rho_face,
                c_face * am_slow * bt_n2 / sqrt_rho_face,
                am_fast * (
                    lambda_fast * lambda_fast
                    - lambda_fast * vn_face
                    + 0.5 * v2_face
                    - gam2 * c_sq_face
                )
                + am_slow * lambda_slow * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn,
            )
            scale = jnp.where(~cs_geq_alfven, sgn_bt, 1.0)
        elif mode == 1:  # alfvén-
            R = (
                0.0,
                0.0,
                -bt_n2,
                bt_n1,
                0.0,
                -bt_n2 * sgn_bn / sqrt_rho_face,
                bt_n1 * sgn_bn / sqrt_rho_face,
                bt_n1 * vt2_face - bt_n2 * vt1_face,
            )
            scale = 1.0
        elif mode == 2:  # slow-
            R = (
                am_slow,
                am_slow * (vn_face - lambda_slow),
                am_slow * vt1_face - am_fast * lambda_fast * bt_n1 * sgn_bn,
                am_slow * vt2_face - am_fast * lambda_fast * bt_n2 * sgn_bn,
                0.0,
                -c_face * am_fast * bt_n1 / sqrt_rho_face,
                -c_face * am_fast * bt_n2 / sqrt_rho_face,
                am_slow * (
                    lambda_slow * lambda_slow
                    - lambda_slow * vn_face
                    + 0.5 * v2_face
                    - gam2 * c_sq_face
                )
                - am_fast * lambda_fast * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn,
            )
            scale = jnp.where(cs_geq_alfven, sgn_bt, 1.0)
        elif mode == 3:  # entropy
            R = (
                1.0,
                vn_face,
                vt1_face,
                vt2_face,
                0.0,
                0.0,
                0.0,
                0.5 * v2_face,
            )
            scale = 1.0
        elif mode == 4:  # slow+
            R = (
                am_slow,
                am_slow * (vn_face + lambda_slow),
                am_slow * vt1_face + am_fast * lambda_fast * bt_n1 * sgn_bn,
                am_slow * vt2_face + am_fast * lambda_fast * bt_n2 * sgn_bn,
                0.0,
                -c_face * am_fast * bt_n1 / sqrt_rho_face,
                -c_face * am_fast * bt_n2 / sqrt_rho_face,
                am_slow * (
                    lambda_slow * lambda_slow
                    + lambda_slow * vn_face
                    + 0.5 * v2_face
                    - gam2 * c_sq_face
                )
                + am_fast * lambda_fast * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn,
            )
            scale = jnp.where(cs_geq_alfven, sgn_bt, 1.0)
        elif mode == 5:  # alfvén+
            R = (
                0.0,
                0.0,
                -bt_n2,
                bt_n1,
                0.0,
                bt_n2 * sgn_bn / sqrt_rho_face,
                -bt_n1 * sgn_bn / sqrt_rho_face,
                bt_n1 * vt2_face - bt_n2 * vt1_face,
            )
            scale = 1.0
        else:  # mode == 6 — fast+
            R = (
                am_fast,
                am_fast * (vn_face + lambda_fast),
                am_fast * vt1_face - am_slow * lambda_slow * bt_n1 * sgn_bn,
                am_fast * vt2_face - am_slow * lambda_slow * bt_n2 * sgn_bn,
                0.0,
                c_face * am_slow * bt_n1 / sqrt_rho_face,
                c_face * am_slow * bt_n2 / sqrt_rho_face,
                am_fast * (
                    lambda_fast * lambda_fast
                    + lambda_fast * vn_face
                    + 0.5 * v2_face
                    - gam2 * c_sq_face
                )
                - am_slow * lambda_slow * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn,
            )
            scale = jnp.where(~cs_geq_alfven, sgn_bt, 1.0)
        return [flux_acc[slot] + (R[slot] * scale) * Fs for slot in range(ncomp)]

    def alpha_for_mode(mode):
        amx = jnp.abs(lambda_from_floored_cell(floored_stencil[0], mode))
        for k in range(1, 6):
            amx = jnp.maximum(
                amx, jnp.abs(lambda_from_floored_cell(floored_stencil[k], mode))
            )
        return amx

    # First-order centered part (1/12 stencil), one per component.
    flux_acc = [
        (-f_stencil[1][slot] + 7.0 * f_stencil[2][slot]
         + 7.0 * f_stencil[3][slot] - f_stencil[4][slot]) * (1.0 / 12.0)
        for slot in range(ncomp)
    ]

    own_speed_modes = ()  # every ideal-MHD field carries mass or energy
    if positivity_preserving:
        # The reference splitting speed (the stencil's spectral radius) and
        # the two split fluxes kept apart, as in the native kernel.
        common_speed = alpha_for_mode(0)
        for mode in range(1, num_modes):
            common_speed = jnp.maximum(common_speed, alpha_for_mode(mode))

        central_state = [
            (-q_stencil[1][slot] + 7.0 * q_stencil[2][slot]
             + 7.0 * q_stencil[3][slot] - q_stencil[4][slot]) * (1.0 / 12.0)
            for slot in range(ncomp)
        ]
        plus_acc = [0.5 * (flux_acc[slot] + common_speed * central_state[slot]) for slot in range(ncomp)]
        minus_acc = [0.5 * (flux_acc[slot] - common_speed * central_state[slot]) for slot in range(ncomp)]
        safe_speed = jnp.maximum(common_speed, 1e-30)
        plus_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]
        minus_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]


    for mode in range(num_modes):
        s = tuple(left_project(mode, f_stencil[k]) for k in range(6))
        qproj = tuple(left_project(mode, q_stencil[k]) for k in range(6))

        d0 = s[1] - s[0]; d1 = s[2] - s[1]; d2 = s[3] - s[2]
        d3 = s[4] - s[3]; d4 = s[5] - s[4]
        dq0 = qproj[1] - qproj[0]; dq1 = qproj[2] - qproj[1]
        dq2 = qproj[3] - qproj[2]; dq3 = qproj[4] - qproj[3]
        dq4 = qproj[5] - qproj[4]

        # (under PP the mass-carrying fields use the common speed: skip their
        # own stencil maximum, which would only be computed and discarded)
        if positivity_preserving and mode not in own_speed_modes:
            amx = common_speed
        else:
            amx = alpha_for_mode(mode)

        aterm_p = 0.5 * (d0 + amx * dq0)
        bterm_p = 0.5 * (d1 + amx * dq1)
        cterm_p = 0.5 * (d2 + amx * dq2)
        dterm_p = 0.5 * (d3 + amx * dq3)
        IS0_p = 13.0 * (aterm_p - bterm_p) ** 2 + 3.0 * (aterm_p - 3.0 * bterm_p) ** 2
        IS1_p = 13.0 * (bterm_p - cterm_p) ** 2 + 3.0 * (bterm_p + cterm_p) ** 2
        IS2_p = 13.0 * (cterm_p - dterm_p) ** 2 + 3.0 * (3.0 * cterm_p - dterm_p) ** 2
        omega0_p, omega2_p = _weno_omega_weights(IS0_p, IS1_p, IS2_p, epsilon, tiny)
        second = (omega0_p * (aterm_p - 2.0 * bterm_p + cterm_p) * (1.0 / 3.0)
                  + (omega2_p - 0.5) * (bterm_p - 2.0 * cterm_p + dterm_p) * (1.0 / 6.0))

        aterm_m = 0.5 * (d4 - amx * dq4)
        bterm_m = 0.5 * (d3 - amx * dq3)
        cterm_m = 0.5 * (d2 - amx * dq2)
        dterm_m = 0.5 * (d1 - amx * dq1)
        IS0_m = 13.0 * (aterm_m - bterm_m) ** 2 + 3.0 * (aterm_m - 3.0 * bterm_m) ** 2
        IS1_m = 13.0 * (bterm_m - cterm_m) ** 2 + 3.0 * (bterm_m + cterm_m) ** 2
        IS2_m = 13.0 * (cterm_m - dterm_m) ** 2 + 3.0 * (3.0 * cterm_m - dterm_m) ** 2
        omega0_m, omega2_m = _weno_omega_weights(IS0_m, IS1_m, IS2_m, epsilon, tiny)
        third = (omega0_m * (aterm_m - 2.0 * bterm_m + cterm_m) * (1.0 / 3.0)
                 + (omega2_m - 0.5) * (bterm_m - 2.0 * cterm_m + dterm_m) * (1.0 / 6.0))

        if positivity_preserving:
            zero_acc = [plus_acc[0] * 0.0 for _ in range(ncomp)]
            plus_acc = add_right_correction(plus_acc, mode, -second)
            minus_acc = add_right_correction(minus_acc, mode, third)
            if mode in own_speed_modes:
                speed_offset = amx - common_speed
                central_projection = (
                    -qproj[1] + 7.0 * qproj[2] + 7.0 * qproj[3] - qproj[4]
                ) * (1.0 / 12.0)
                central_shift = add_right_correction(zero_acc, mode, 0.5 * speed_offset * central_projection)
                plus_acc = [plus_acc[slot] + central_shift[slot] for slot in range(ncomp)]
                minus_acc = [minus_acc[slot] - central_shift[slot] for slot in range(ncomp)]
                relative_offset = speed_offset / safe_speed
                plus_shift = add_right_correction(plus_shift, mode, relative_offset * qproj[2])
                minus_shift = add_right_correction(minus_shift, mode, relative_offset * qproj[3])
            continue

        Fs = -second + third
        flux_acc = add_right_correction(flux_acc, mode, Fs)

    if positivity_preserving:
        if return_split:
            return plus_acc, minus_acc, common_speed
        flux_acc = positivity_preserving_flux_local(
            q_stencil[2], q_stencil[3], f_stencil[2], f_stencil[3],
            plus_acc, minus_acc, plus_shift, minus_shift,
            common_speed, gm1, rhomin, pgmin, ideal_gas=True, magnetic_slots=(4, 5, 6),
        )

    return flux_acc


def _weno_flux_mhd_pallas_local(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
    internal_energy_density=None,
    inflow_reference=None,
):
    """Single-shard ideal-gas MHD WENO build.  Mirrors
    ``_weno_flux_hydro_pallas`` but with 8 conserved variables and 7
    characteristic waves (fast-, alfvén-, slow-, entropy, slow+, alfvén+,
    fast+).  The per-interface arithmetic — all face eigenstructure (the body
    of ``_eigen_mhd._eigenvector_building_blocks``) and the ``L_row``/``R_col``/
    ``λ`` projections dispatched at compile time via ``if mode == k`` branches —
    lives in the shared pure :func:`_weno_mhd_flux_from_window`, which the
    kernel calls on the gathered 6-cell stencil.  No full-domain projection
    matrices are ever materialised —
    every component is computed per-tile in registers."""
    has_g = internal_energy_density is not None
    ndim = 3
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx, ny, nz = spatial_shape
    bx, by, bz = _as_3tuple_block_shape(config.backend_config.pallas_block_shape, ndim, spatial_shape=spatial_shape)
    grid = (nx // bx, ny // by, nz // bz)

    local_indices = _mhd_indices_for_axis(config, registered_variables, axis)
    ncomp = 8
    num_modes = 7
    positivity_preserving = config.weno_positivity_preserving
    # positivity preserving: plus and minus split face fluxes, then the speed
    out_channels = 2 * nvars + 1 if positivity_preserving else nvars

    # Tile sizes / specs — identical to the hydro kernel.
    block_shape_out = (out_channels, bx, by, bz)
    out_spec = pl.BlockSpec(block_shape_out, lambda bi, bj, bk: (0, bi, bj, bk))
    in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    # ``b_eps`` and the floors for ``sqrt`` are passed in as scalar kernel
    # arguments so they carry the same dtype as the input state.  This
    # matters under x64 + Triton: an untyped Python ``1e-20`` enters the
    # lowering as f32, which trips a ``('f64','f32')`` assertion in
    # ``_truediv_lowering_rule`` further down (see guide §5 x64 notes).
    b_eps_value = 1e-20
    sqrt_floor_value = 1e-12

    def kernel(*refs):
        if has_g:
            (q_ref, g_ref, gamma_ref, rhomin_ref, pgmin_ref, b_eps_ref,
             sqrt_floor_ref, eta_ref, flux_out_ref) = refs
        else:
            (q_ref, gamma_ref, rhomin_ref, pgmin_ref, b_eps_ref,
             sqrt_floor_ref, flux_out_ref) = refs
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)

        ii = (bi * bx + jnp.arange(bx)[:, None, None]) % nx
        jj = (bj * by + jnp.arange(by)[None, :, None]) % ny
        kk = (bk * bz + jnp.arange(bz)[None, None, :]) % nz

        # Scalars are read here and passed to the shared window function, which
        # derives gm1/gam0/typed-literal scalars internally (the per-interface
        # arithmetic — including all x64/Triton dtype hygiene — lives there so
        # the forward kernel and the jax.vjp adjoint share one source of truth).
        gamma = gamma_ref[()]
        b_eps = b_eps_ref[()]
        sqrt_floor = sqrt_floor_ref[()]
        rhomin = rhomin_ref[()]
        pgmin = pgmin_ref[()]

        def q_at(var_index: int, offset: int):
            if axis == 0:
                return q_ref[var_index, (ii + offset) % nx, jj, kk]
            if axis == 1:
                return q_ref[var_index, ii, (jj + offset) % ny, kk]
            return q_ref[var_index, ii, jj, (kk + offset) % nz]

        def q_local(offset: int):
            # local order: rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy
            return tuple(q_at(idx, offset) for idx in local_indices)

        q_stencil = tuple(q_local(off) for off in range(-2, 4))     # offsets -2..3

        window_kwargs = {}
        if has_g:
            def g_at(offset: int):
                if axis == 0:
                    return g_ref[0, (ii + offset) % nx, jj, kk]
                if axis == 1:
                    return g_ref[0, ii, (jj + offset) % ny, kk]
                return g_ref[0, ii, jj, (kk + offset) % nz]

            window_kwargs["g_stencil"] = tuple(g_at(off) for off in range(-2, 4))
            window_kwargs["dual_eta"] = eta_ref[()]

        window = _weno_mhd_flux_from_window(
            q_stencil, gamma, rhomin, pgmin, b_eps, sqrt_floor,
            ncomp, num_modes,
            use_approx_rsqrt=config.backend_config.use_approximate_rsqrt,
            admissible_face_state=config.weno_admissible_face_state,
            positivity_preserving=positivity_preserving,
            return_split=positivity_preserving,
            **window_kwargs,
        )
        if positivity_preserving:
            # split face fluxes + splitting speed; recombined (paired) outside
            plus_acc, minus_acc, common_speed = window
            zero = common_speed * 0.0
            for var in range(2 * nvars):
                flux_out_ref[var, ...] = zero
            for slot, var in enumerate(local_indices):
                flux_out_ref[var, ...] = plus_acc[slot]
                flux_out_ref[nvars + var, ...] = minus_acc[slot]
            flux_out_ref[2 * nvars, ...] = common_speed
            return
        flux_acc = window

        # Write every output component.  Hydro/MHD covers all conserved
        # variables, but explicitly zero anything not in ``local_indices``
        # (defensive — also makes the B_normal-flux = 0 invariant explicit).
        zero = flux_acc[0] * 0.0
        for var in range(nvars):
            flux_out_ref[var, ...] = zero
        for slot, var in enumerate(local_indices):
            flux_out_ref[var, ...] = flux_acc[slot]

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    out_shape = jax.ShapeDtypeStruct((out_channels,) + spatial_shape, conserved_state.dtype)
    scalars = (
        jnp.asarray(params.gamma, dtype=conserved_state.dtype),
        jnp.asarray(params.minimum_density, dtype=conserved_state.dtype),
        jnp.asarray(params.minimum_pressure, dtype=conserved_state.dtype),
        jnp.asarray(b_eps_value, dtype=conserved_state.dtype),
        jnp.asarray(sqrt_floor_value, dtype=conserved_state.dtype),
    )

    if has_g:
        in_g_spec = pl.BlockSpec(internal_energy_density.shape, lambda bi, bj, bk: (0, 0, 0, 0))
        in_specs = [in_state_spec, in_g_spec] + [scalar_spec] * 6
        args = (conserved_state, internal_energy_density) + scalars + (
            jnp.asarray(config.dual_energy_eta, dtype=conserved_state.dtype),
        )
    else:
        in_specs = [in_state_spec] + [scalar_spec] * 5
        args = (conserved_state,) + scalars

    flux = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name=f"mhd_weno_flux_axis_{axis}" + ("_dual" if has_g else ""),
        **kwargs,
    )(*args)

    if positivity_preserving and inflow_reference is not None:
        # joint per-cell inflow limiting, fused into one kernel (it reads the
        # split fluxes of the interfaces i - 1/2, i + 1/2, i + 3/2)
        from astronomix._finite_difference._interface_fluxes._weno_positivity_pallas import (
            mhd_joint_recombination_pallas,
        )
        return mhd_joint_recombination_pallas(
            conserved_state, flux, inflow_reference[0], inflow_reference[1],
            params, config, registered_variables, axis=axis,
        )
    if positivity_preserving:
        # paired recombination (needs the neighbouring interfaces; see
        # _weno_positivity._paired_scalings), on the split face fluxes
        zero = jnp.zeros_like(conserved_state)
        return positivity_preserving_interface_flux(
            conserved_state,
            mhd_physical_flux(conserved_state, params.gamma, registered_variables, axis),
            flux[2 * nvars], flux[:nvars], flux[nvars:2 * nvars], zero, zero,
            params, config, registered_variables, axis=axis, inflow_reference=inflow_reference,
        )
    return flux


# -----------------------------------------------------------------------------
# Pallas WENO for the isothermal MHD equations.
# -----------------------------------------------------------------------------


def _mhd_iso_pallas_flux_supported(conserved_state, config: SimulationConfig) -> bool:
    """Whether the Pallas isothermal MHD WENO kernel can be used."""
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if not config.mhd:
        return False
    if config.equation_of_state != ISOTHERMAL:
        return False
    ndim = int(config.dimensionality)
    if ndim != 3:
        return False
    if conserved_state.ndim != 4:
        return False
    block_shape = _as_3tuple_block_shape(config.backend_config.pallas_block_shape, ndim, spatial_shape=conserved_state.shape[1:])
    for n, b in zip(conserved_state.shape[1:], block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _mhd_iso_indices_for_axis(config: SimulationConfig, registered_variables: RegisteredVariables, axis: int):
    """Local conserved-variable order for isothermal MHD: (density, p_normal,
    p_trans1, p_trans2, B_normal, B_trans1, B_trans2).  Seven slots — no
    energy.  ``B_normal`` is the 0-coefficient placeholder so the
    L_row/R_col formulas can use the same projection structure as ideal-gas
    MHD, and its output flux slot is zeroed (matching ``_mhd_flux_isothermal_x``).
    """
    density_index = int(registered_variables.density_index)
    mx = int(registered_variables.momentum_index.x)
    my = int(registered_variables.momentum_index.y)
    mz = int(registered_variables.momentum_index.z)
    bx = int(registered_variables.magnetic_index.x)
    by = int(registered_variables.magnetic_index.y)
    bz = int(registered_variables.magnetic_index.z)

    if axis == 0:
        return (density_index, mx, my, mz, bx, by, bz)
    if axis == 1:
        return (density_index, my, mx, mz, by, bx, bz)
    return (density_index, mz, my, mx, bz, by, bx)


def _weno_flux_mhd_iso_pallas(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Pallas implementation of the isothermal MHD WENO interface flux.

    Public entry point: dispatches the supported-predicate check and the
    multi-GPU ``shard_map`` + halo wrap.  Kernel arithmetic in
    ``_weno_flux_mhd_iso_pallas_local``.
    """
    if not _mhd_iso_pallas_flux_supported(conserved_state, config):
        # Lazy import to break the circular dependency with _weno.py.
        from astronomix._finite_difference._interface_fluxes._weno import (
            _weno_flux_x_native, _weno_flux_y_native, _weno_flux_z_native,
        )
        if axis == 0:
            return _weno_flux_x_native(conserved_state, params, config, registered_variables)
        if axis == 1:
            return _weno_flux_y_native(conserved_state, params, config, registered_variables)
        return _weno_flux_z_native(conserved_state, params, config, registered_variables)

    def _local(state_local):
        return _weno_flux_mhd_iso_pallas_local(
            state_local, params, config, registered_variables, axis=axis
        )
    return _weno5_shard_wrap(_local, conserved_state, config, axis)


def _weno_flux_mhd_iso_pallas_local(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Single-shard isothermal MHD WENO build.  Mirrors
    ``_weno_flux_mhd_pallas`` but with 7 conserved-state slots (no
    energy) and 6 characteristic waves (no entropy mode): fast-,
    alfvén-, slow-, slow+, alfvén+, fast+.  Sound speed is the fixed
    ``params.isothermal_sound_speed``.  All face eigenstructure,
    ``L_row``, ``R_col``, and ``λ`` are inlined as kernel-local closures
    mirroring ``_eigen_mhd_iso`` line-for-line."""
    ndim = 3
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx, ny, nz = spatial_shape
    bx_, by_, bz_ = _as_3tuple_block_shape(config.backend_config.pallas_block_shape, ndim, spatial_shape=spatial_shape)
    grid = (nx // bx_, ny // by_, nz // bz_)

    local_indices = _mhd_iso_indices_for_axis(config, registered_variables, axis)
    ncomp = 7
    num_modes = 6
    epsilon = config.weno_epsilon
    # WENO-Z is a static config choice, so the weight function can be bound
    # here and inlined into the Pallas kernel body below.
    omega_weights = _weno_omega_weights_z if config.weno_z else _weno_omega_weights
    tiny = 1e-14
    b_eps_value = 1e-20
    positivity_preserving = config.weno_positivity_preserving
    # fields that carry no mass keep their own splitting speed
    own_speed_modes = mass_free_modes(config) if positivity_preserving else ()

    block_shape_out = (nvars, bx_, by_, bz_)
    out_spec = pl.BlockSpec(block_shape_out, lambda bi, bj, bk: (0, bi, bj, bk))
    in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(q_ref, cs_ref, rhomin_ref, b_eps_ref, flux_out_ref):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)

        ii = (bi * bx_ + jnp.arange(bx_)[:, None, None]) % nx
        jj = (bj * by_ + jnp.arange(by_)[None, :, None]) % ny
        kk = (bk * bz_ + jnp.arange(bz_)[None, None, :]) % nz

        cs = cs_ref[()]
        cs2 = cs * cs
        cs2_inv = jnp.where(cs2 > 0.0, 1.0 / cs2, 0.0)
        rhomin = rhomin_ref[()]
        b_eps = b_eps_ref[()]
        # Properly-typed literal scalars (see x64-Triton workaround in the
        # ideal-gas MHD kernel for the rationale).
        zero_typed = cs - cs
        one_typed = zero_typed + 1.0
        neg_one_typed = zero_typed - 1.0
        inv_sqrt_two_typed = zero_typed + (1.0 / 2.0 ** 0.5)

        def q_at(var_index, offset):
            if axis == 0:
                return q_ref[var_index, (ii + offset) % nx, jj, kk]
            if axis == 1:
                return q_ref[var_index, ii, (jj + offset) % ny, kk]
            return q_ref[var_index, ii, jj, (kk + offset) % nz]

        def q_local(offset):
            return tuple(q_at(idx, offset) for idx in local_indices)

        def primitive_from_q(q):
            rho, mn, mt1, mt2, Bn, Bt1, Bt2 = q
            inv_rho = 1.0 / rho
            vn = mn * inv_rho
            vt1 = mt1 * inv_rho
            vt2 = mt2 * inv_rho
            v2 = vn * vn + vt1 * vt1 + vt2 * vt2
            b2 = Bn * Bn + Bt1 * Bt1 + Bt2 * Bt2
            return rho, mn, mt1, mt2, Bn, Bt1, Bt2, vn, vt1, vt2, v2, b2

        def floored_cell(q):
            rho, mn, mt1, mt2, Bn, Bt1, Bt2, vn, vt1, vt2, v2, b2 = primitive_from_q(q)
            rho_f = jnp.maximum(rho, rhomin)
            # Recompute primitives that depend on the floored density to keep
            # downstream arithmetic consistent.
            inv_rho = 1.0 / rho_f
            vn_f = mn * inv_rho
            vt1_f = mt1 * inv_rho
            vt2_f = mt2 * inv_rho
            bn2_over_rho = (Bn * Bn) / rho_f
            disc_root = jnp.sqrt(jnp.maximum(
                0.0, (b2 / rho_f + cs2) ** 2 - 4.0 * bn2_over_rho * cs2
            ))
            c_fast = jnp.sqrt(jnp.maximum(0.0, 0.5 * (b2 / rho_f + cs2 + disc_root)))
            c_alfven = jnp.sqrt(jnp.maximum(0.0, bn2_over_rho))
            c_slow = jnp.sqrt(jnp.maximum(0.0, 0.5 * (b2 / rho_f + cs2 - disc_root)))
            return (rho_f, mn, mt1, mt2, Bn, Bt1, Bt2,
                    vn_f, vt1_f, vt2_f, c_fast, c_alfven, c_slow)

        def flux_from_q(q):
            """Isothermal MHD x-flux in local order; B_normal flux is 0."""
            rho, mn, mt1, mt2, Bn, Bt1, Bt2, vn, vt1, vt2, v2, b2 = primitive_from_q(q)
            p_iso = cs2 * rho
            p_total = p_iso + 0.5 * b2
            return (
                mn,
                rho * vn * vn + p_total - Bn * Bn,
                rho * vn * vt1 - Bn * Bt1,
                rho * vn * vt2 - Bn * Bt2,
                0.0,
                Bt1 * vn - Bn * vt1,
                Bt2 * vn - Bn * vt2,
            )

        def lambda_from_floored_cell(cell, mode: int):
            vn = cell[7]; c_fast = cell[10]; c_alfven = cell[11]; c_slow = cell[12]
            if mode == 0:
                return vn - c_fast
            if mode == 1:
                return vn - c_alfven
            if mode == 2:
                return vn - c_slow
            if mode == 3:
                return vn + c_slow
            if mode == 4:
                return vn + c_alfven
            return vn + c_fast

        q_stencil = tuple(q_local(off) for off in range(-2, 4))
        f_stencil = tuple(flux_from_q(q) for q in q_stencil)
        floored_stencil = tuple(floored_cell(q) for q in q_stencil)
        cell_l = floored_stencil[2]
        cell_r = floored_stencil[3]

        rho_i, mn_i, mt1_i, mt2_i, Bn_i, Bt1_i, Bt2_i = cell_l[:7]
        rho_j, mn_j, mt1_j, mt2_j, Bn_j, Bt1_j, Bt2_j = cell_r[:7]
        rho_face = jnp.maximum(
            0.5 * (jnp.maximum(rho_i, rhomin) + jnp.maximum(rho_j, rhomin)),
            rhomin,
        )
        vn_face = 0.5 * (mn_i + mn_j) / rho_face
        vt1_face = 0.5 * (mt1_i + mt1_j) / rho_face
        vt2_face = 0.5 * (mt2_i + mt2_j) / rho_face
        Bn_face = 0.5 * (Bn_i + Bn_j)
        Bt1_face = 0.5 * (Bt1_i + Bt1_j)
        Bt2_face = 0.5 * (Bt2_i + Bt2_j)

        b2_face = Bn_face * Bn_face + Bt1_face * Bt1_face + Bt2_face * Bt2_face
        b2_over_rho = b2_face / rho_face
        bn2_over_rho = (Bn_face * Bn_face) / rho_face

        ms_disc = (b2_over_rho + cs2) ** 2 - 4.0 * bn2_over_rho * cs2
        ms_disc_root = jnp.sqrt(jnp.maximum(ms_disc, 0.0))
        lambda_fast = jnp.sqrt(jnp.maximum(0.0, 0.5 * (b2_over_rho + cs2 + ms_disc_root)))
        lambda_alfven = jnp.sqrt(jnp.maximum(0.0, bn2_over_rho))
        lambda_slow = jnp.sqrt(jnp.maximum(0.0, 0.5 * (b2_over_rho + cs2 - ms_disc_root)))

        bt_sq = Bt1_face * Bt1_face + Bt2_face * Bt2_face
        bt_sq_safe = jnp.maximum(bt_sq, b_eps)
        bt_n1 = jnp.where(bt_sq >= b_eps, Bt1_face / jnp.sqrt(bt_sq_safe), inv_sqrt_two_typed)
        bt_n2 = jnp.where(bt_sq >= b_eps, Bt2_face / jnp.sqrt(bt_sq_safe), inv_sqrt_two_typed)

        sgn_bn = jnp.where(Bn_face >= 0.0, one_typed, neg_one_typed)
        sgn_bt = jnp.where(
            Bt1_face != 0.0,
            jnp.where(Bt1_face >= 0.0, one_typed, neg_one_typed),
            jnp.where(Bt2_face >= 0.0, one_typed, neg_one_typed),
        )

        denom = lambda_fast * lambda_fast - lambda_slow * lambda_slow
        denom_safe = jnp.maximum(denom, b_eps)
        am_fast = jnp.where(
            denom >= b_eps,
            jnp.sqrt(jnp.maximum(0.0, cs2 - lambda_slow * lambda_slow)) / jnp.sqrt(denom_safe),
            1.0,
        )
        am_slow = jnp.where(
            denom >= b_eps,
            jnp.sqrt(jnp.maximum(0.0, lambda_fast * lambda_fast - cs2)) / jnp.sqrt(denom_safe),
            1.0,
        )

        sqrt_rho_face = jnp.sqrt(jnp.maximum(rho_face, rhomin))
        cs_geq_alfven = cs >= lambda_alfven

        def left_project(mode: int, values):
            """L_row[mode] · values for iso MHD.  ``values`` is a 7-tuple:
            (rho, mn, mt1, mt2, Bn, Bt1, Bt2)."""
            rho_v, mn_v, mt1_v, mt2_v, Bn_v, Bt1_v, Bt2_v = values
            if mode == 0:  # fast-
                L_rho = (
                    am_fast * (cs2 + lambda_fast * vn_face)
                    - am_slow * lambda_slow * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
                )
                L_mn = -am_fast * lambda_fast
                L_mt1 = am_slow * lambda_slow * bt_n1 * sgn_bn
                L_mt2 = am_slow * lambda_slow * bt_n2 * sgn_bn
                L_Bt1 = cs * am_slow * bt_n1 * sqrt_rho_face
                L_Bt2 = cs * am_slow * bt_n2 * sqrt_rho_face
                acc = (L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                       + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v)
                acc = 0.5 * acc * cs2_inv
                return jnp.where(~cs_geq_alfven, acc * sgn_bt, acc)
            if mode == 1:  # alfvén-
                L_rho = bt_n2 * vt1_face - bt_n1 * vt2_face
                L_mt1 = -bt_n2
                L_mt2 = bt_n1
                L_Bt1 = -bt_n2 * sgn_bn * sqrt_rho_face
                L_Bt2 = bt_n1 * sgn_bn * sqrt_rho_face
                acc = (L_rho * rho_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                       + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v)
                return 0.5 * acc
            if mode == 2:  # slow-
                L_rho = (
                    am_slow * (cs2 + lambda_slow * vn_face)
                    + am_fast * lambda_fast * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
                )
                L_mn = -am_slow * lambda_slow
                L_mt1 = -am_fast * lambda_fast * bt_n1 * sgn_bn
                L_mt2 = -am_fast * lambda_fast * bt_n2 * sgn_bn
                L_Bt1 = -cs * am_fast * bt_n1 * sqrt_rho_face
                L_Bt2 = -cs * am_fast * bt_n2 * sqrt_rho_face
                acc = (L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                       + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v)
                acc = 0.5 * acc * cs2_inv
                return jnp.where(cs_geq_alfven, acc * sgn_bt, acc)
            if mode == 3:  # slow+
                L_rho = (
                    am_slow * (cs2 - lambda_slow * vn_face)
                    - am_fast * lambda_fast * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
                )
                L_mn = am_slow * lambda_slow
                L_mt1 = am_fast * lambda_fast * bt_n1 * sgn_bn
                L_mt2 = am_fast * lambda_fast * bt_n2 * sgn_bn
                L_Bt1 = -cs * am_fast * bt_n1 * sqrt_rho_face
                L_Bt2 = -cs * am_fast * bt_n2 * sqrt_rho_face
                acc = (L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                       + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v)
                acc = 0.5 * acc * cs2_inv
                return jnp.where(cs_geq_alfven, acc * sgn_bt, acc)
            if mode == 4:  # alfvén+
                L_rho = bt_n2 * vt1_face - bt_n1 * vt2_face
                L_mt1 = -bt_n2
                L_mt2 = bt_n1
                L_Bt1 = bt_n2 * sgn_bn * sqrt_rho_face
                L_Bt2 = -bt_n1 * sgn_bn * sqrt_rho_face
                acc = (L_rho * rho_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                       + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v)
                return 0.5 * acc
            # mode 5 — fast+
            L_rho = (
                am_fast * (cs2 - lambda_fast * vn_face)
                + am_slow * lambda_slow * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
            )
            L_mn = am_fast * lambda_fast
            L_mt1 = -am_slow * lambda_slow * bt_n1 * sgn_bn
            L_mt2 = -am_slow * lambda_slow * bt_n2 * sgn_bn
            L_Bt1 = cs * am_slow * bt_n1 * sqrt_rho_face
            L_Bt2 = cs * am_slow * bt_n2 * sqrt_rho_face
            acc = (L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                   + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v)
            acc = 0.5 * acc * cs2_inv
            return jnp.where(~cs_geq_alfven, acc * sgn_bt, acc)

        def add_right_correction(flux_acc, mode: int, Fs):
            if mode == 0:  # fast-
                R = (
                    am_fast,
                    am_fast * (vn_face - lambda_fast),
                    am_fast * vt1_face + am_slow * lambda_slow * bt_n1 * sgn_bn,
                    am_fast * vt2_face + am_slow * lambda_slow * bt_n2 * sgn_bn,
                    0.0,
                    cs * am_slow * bt_n1 / sqrt_rho_face,
                    cs * am_slow * bt_n2 / sqrt_rho_face,
                )
                scale = jnp.where(~cs_geq_alfven, sgn_bt, 1.0)
            elif mode == 1:  # alfvén-
                R = (
                    0.0, 0.0,
                    -bt_n2, bt_n1, 0.0,
                    -bt_n2 * sgn_bn / sqrt_rho_face,
                    bt_n1 * sgn_bn / sqrt_rho_face,
                )
                scale = 1.0
            elif mode == 2:  # slow-
                R = (
                    am_slow,
                    am_slow * (vn_face - lambda_slow),
                    am_slow * vt1_face - am_fast * lambda_fast * bt_n1 * sgn_bn,
                    am_slow * vt2_face - am_fast * lambda_fast * bt_n2 * sgn_bn,
                    0.0,
                    -cs * am_fast * bt_n1 / sqrt_rho_face,
                    -cs * am_fast * bt_n2 / sqrt_rho_face,
                )
                scale = jnp.where(cs_geq_alfven, sgn_bt, 1.0)
            elif mode == 3:  # slow+
                R = (
                    am_slow,
                    am_slow * (vn_face + lambda_slow),
                    am_slow * vt1_face + am_fast * lambda_fast * bt_n1 * sgn_bn,
                    am_slow * vt2_face + am_fast * lambda_fast * bt_n2 * sgn_bn,
                    0.0,
                    -cs * am_fast * bt_n1 / sqrt_rho_face,
                    -cs * am_fast * bt_n2 / sqrt_rho_face,
                )
                scale = jnp.where(cs_geq_alfven, sgn_bt, 1.0)
            elif mode == 4:  # alfvén+
                R = (
                    0.0, 0.0,
                    -bt_n2, bt_n1, 0.0,
                    bt_n2 * sgn_bn / sqrt_rho_face,
                    -bt_n1 * sgn_bn / sqrt_rho_face,
                )
                scale = 1.0
            else:  # mode 5 — fast+
                R = (
                    am_fast,
                    am_fast * (vn_face + lambda_fast),
                    am_fast * vt1_face - am_slow * lambda_slow * bt_n1 * sgn_bn,
                    am_fast * vt2_face - am_slow * lambda_slow * bt_n2 * sgn_bn,
                    0.0,
                    cs * am_slow * bt_n1 / sqrt_rho_face,
                    cs * am_slow * bt_n2 / sqrt_rho_face,
                )
                scale = jnp.where(~cs_geq_alfven, sgn_bt, 1.0)
            return [flux_acc[slot] + (R[slot] * scale) * Fs for slot in range(ncomp)]

        def alpha_for_mode(mode: int):
            amx = jnp.abs(lambda_from_floored_cell(floored_stencil[0], mode))
            for k in range(1, 6):
                amx = jnp.maximum(
                    amx, jnp.abs(lambda_from_floored_cell(floored_stencil[k], mode))
                )
            return amx

        flux_acc = [
            (-f_stencil[1][slot] + 7.0 * f_stencil[2][slot]
             + 7.0 * f_stencil[3][slot] - f_stencil[4][slot]) * (1.0 / 12.0)
            for slot in range(ncomp)
        ]

        if positivity_preserving:
            # One splitting speed (the stencil's spectral radius) for every
            # field that carries mass; the split fluxes kept apart.
            common_speed = alpha_for_mode(0)
            for mode in range(1, num_modes):
                common_speed = jnp.maximum(common_speed, alpha_for_mode(mode))
            safe_speed = jnp.maximum(common_speed, 1e-30)
            central_state = [
                (-q_stencil[1][slot] + 7.0 * q_stencil[2][slot]
                 + 7.0 * q_stencil[3][slot] - q_stencil[4][slot]) * (1.0 / 12.0)
                for slot in range(ncomp)
            ]
            plus_acc = [0.5 * (flux_acc[slot] + common_speed * central_state[slot]) for slot in range(ncomp)]
            minus_acc = [0.5 * (flux_acc[slot] - common_speed * central_state[slot]) for slot in range(ncomp)]
            plus_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]
            minus_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]


        for mode in range(num_modes):
            s = tuple(left_project(mode, f_stencil[k]) for k in range(6))
            qproj = tuple(left_project(mode, q_stencil[k]) for k in range(6))

            d0 = s[1] - s[0]; d1 = s[2] - s[1]; d2 = s[3] - s[2]
            d3 = s[4] - s[3]; d4 = s[5] - s[4]
            dq0 = qproj[1] - qproj[0]; dq1 = qproj[2] - qproj[1]
            dq2 = qproj[3] - qproj[2]; dq3 = qproj[4] - qproj[3]
            dq4 = qproj[5] - qproj[4]

            # (under PP the mass-carrying fields use the common speed: skip their
            # own stencil maximum, which would only be computed and discarded)
            if positivity_preserving and mode not in own_speed_modes:
                amx = common_speed
            else:
                amx = alpha_for_mode(mode)

            aterm_p = 0.5 * (d0 + amx * dq0); bterm_p = 0.5 * (d1 + amx * dq1)
            cterm_p = 0.5 * (d2 + amx * dq2); dterm_p = 0.5 * (d3 + amx * dq3)
            IS0_p = 13.0 * (aterm_p - bterm_p) ** 2 + 3.0 * (aterm_p - 3.0 * bterm_p) ** 2
            IS1_p = 13.0 * (bterm_p - cterm_p) ** 2 + 3.0 * (bterm_p + cterm_p) ** 2
            IS2_p = 13.0 * (cterm_p - dterm_p) ** 2 + 3.0 * (3.0 * cterm_p - dterm_p) ** 2
            omega0_p, omega2_p = omega_weights(IS0_p, IS1_p, IS2_p, epsilon, tiny)
            second = (omega0_p * (aterm_p - 2.0 * bterm_p + cterm_p) * (1.0 / 3.0)
                      + (omega2_p - 0.5) * (bterm_p - 2.0 * cterm_p + dterm_p) * (1.0 / 6.0))

            aterm_m = 0.5 * (d4 - amx * dq4); bterm_m = 0.5 * (d3 - amx * dq3)
            cterm_m = 0.5 * (d2 - amx * dq2); dterm_m = 0.5 * (d1 - amx * dq1)
            IS0_m = 13.0 * (aterm_m - bterm_m) ** 2 + 3.0 * (aterm_m - 3.0 * bterm_m) ** 2
            IS1_m = 13.0 * (bterm_m - cterm_m) ** 2 + 3.0 * (bterm_m + cterm_m) ** 2
            IS2_m = 13.0 * (cterm_m - dterm_m) ** 2 + 3.0 * (3.0 * cterm_m - dterm_m) ** 2
            omega0_m, omega2_m = omega_weights(IS0_m, IS1_m, IS2_m, epsilon, tiny)
            third = (omega0_m * (aterm_m - 2.0 * bterm_m + cterm_m) * (1.0 / 3.0)
                     + (omega2_m - 0.5) * (bterm_m - 2.0 * cterm_m + dterm_m) * (1.0 / 6.0))

            if positivity_preserving:
                zero_acc = [plus_acc[0] * 0.0 for _ in range(ncomp)]
                plus_acc = add_right_correction(plus_acc, mode, -second)
                minus_acc = add_right_correction(minus_acc, mode, third)
                if mode in own_speed_modes:
                    speed_offset = amx - common_speed
                    central_projection = (
                        -qproj[1] + 7.0 * qproj[2] + 7.0 * qproj[3] - qproj[4]
                    ) * (1.0 / 12.0)
                    central_shift = add_right_correction(zero_acc, mode, 0.5 * speed_offset * central_projection)
                    plus_acc = [plus_acc[slot] + central_shift[slot] for slot in range(ncomp)]
                    minus_acc = [minus_acc[slot] - central_shift[slot] for slot in range(ncomp)]
                    relative_offset = speed_offset / safe_speed
                    plus_shift = add_right_correction(plus_shift, mode, relative_offset * qproj[2])
                    minus_shift = add_right_correction(minus_shift, mode, relative_offset * qproj[3])
                continue

            Fs = -second + third
            flux_acc = add_right_correction(flux_acc, mode, Fs)

        if positivity_preserving:
            flux_acc = positivity_preserving_flux_local(
                q_stencil[2], q_stencil[3], f_stencil[2], f_stencil[3],
                plus_acc, minus_acc, plus_shift, minus_shift,
                common_speed, 0.0, rhomin, 0.0, ideal_gas=False, magnetic_slots=(4, 5, 6),
            )

        zero = flux_acc[0] * 0.0
        for var in range(nvars):
            flux_out_ref[var, ...] = zero
        for slot, var in enumerate(local_indices):
            flux_out_ref[var, ...] = flux_acc[slot]

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=[in_state_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name=f"mhd_iso_weno_flux_axis_{axis}",
        **kwargs,
    )(
        conserved_state,
        jnp.asarray(params.isothermal_sound_speed, dtype=conserved_state.dtype),
        jnp.asarray(params.minimum_density, dtype=conserved_state.dtype),
        jnp.asarray(b_eps_value, dtype=conserved_state.dtype),
    )


def _weno_flux_hydro_pallas_rhs(
    conserved_state,
    dt_over_dx,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
    rhs_accumulator=None,
):
    """Fused WENO interface flux + axis-flux-divergence kernel.

    Computes ``rhs_out = (rhs_accumulator if provided else 0) +
    (-dt_over_dx) * d/dx_axis(F_axis(state))`` directly, without ever
    materialising the full-state-sized interface flux ``F_axis``.  Each Pallas
    block evaluates the two interface fluxes ``F_{i+1/2}`` and ``F_{i-1/2}``
    it needs locally and writes the divergence contribution (added to the
    accumulator, when present) into its output tile.

    When ``rhs_accumulator`` is provided, the kernel uses
    ``input_output_aliases`` so XLA can keep a single physical RHS buffer
    across all three axes — eliminating both the materialised ``dF``
    temporaries and the chained ``rhs + ...`` adds that would otherwise
    duplicate full-state buffers.

    The arithmetic matches a single pass through ``_weno_flux_hydro_pallas``
    followed by ``_hydro_flux_divergence_pallas``; the only change is that the
    left interface flux is also computed inside the same program rather than
    being read back from HBM.

    Public entry point: dispatches the supported-predicate check and the
    multi-GPU ``shard_map`` + halo wrap.  The same WENO5 halo as the
    pure-flux variant (3 cells on the active axis) suffices — the fused
    kernel evaluates both ``F_{i+1/2}`` and ``F_{i-1/2}``, and the deepest
    read inside ``F_{i-1/2}`` is at offset ``-3`` from the cell index.
    Arithmetic lives in ``_weno_flux_hydro_pallas_rhs_local``.
    """
    if not _hydro_pallas_flux_supported(conserved_state, config):
        raise RuntimeError(
            "_weno_flux_hydro_pallas_rhs called when Pallas WENO is unsupported."
        )

    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(config.backend_config.pallas_block_shape, ndim, spatial_shape=conserved_state.shape[1:])
    halo_list = [0, 0, 0]
    if 0 <= int(axis) < ndim:
        halo_list[int(axis)] = 3
    halo = tuple(halo_list[:ndim])

    if rhs_accumulator is None:
        def _local(state_local):
            return _weno_flux_hydro_pallas_rhs_local(
                state_local, dt_over_dx, params, config, registered_variables,
                axis=axis, rhs_accumulator=None,
            )
        return _pallas_call_sharded(
            _local,
            state_inputs=(conserved_state,),
            halo=halo,
            block_shape=block_shape[:ndim],
        )

    def _local(rhs_local, state_local):
        return _weno_flux_hydro_pallas_rhs_local(
            state_local, dt_over_dx, params, config, registered_variables,
            axis=axis, rhs_accumulator=rhs_local,
        )
    return _pallas_call_sharded(
        _local,
        state_inputs=(rhs_accumulator, conserved_state),
        halo=halo,
        block_shape=block_shape[:ndim],
    )


def _weno_flux_hydro_pallas_rhs_local(
    conserved_state,
    dt_over_dx,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
    rhs_accumulator=None,
):
    """Single-shard fused WENO + divergence kernel build."""
    accumulate = rhs_accumulator is not None

    ndim = int(config.dimensionality)
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx, by, bz = _as_3tuple_block_shape(config.backend_config.pallas_block_shape, ndim, spatial_shape=spatial_shape)
    grid = (nx // bx, ny // by, nz // bz)

    local_indices = _hydro_indices_for_axis(config, registered_variables, axis)
    ncomp = len(local_indices)
    num_modes = ndim + 2
    epsilon = config.weno_epsilon
    # WENO-Z is a static config choice, so the weight function can be bound
    # here and inlined into the Pallas kernel body below.
    omega_weights = _weno_omega_weights_z if config.weno_z else _weno_omega_weights
    tiny = 1e-14

    if ndim == 1:
        block_shape = (nvars, bx)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        block_shape = (nvars, bx, by)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        block_shape = (nvars, bx, by, bz)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(*refs):
        # The kernel accepts either 5 inputs (no accumulator) or 6 inputs
        # (with accumulator).  Both layouts end with the dt-over-dx scalar and
        # an output ref; the accumulator, when present, comes first so it can
        # be aliased to the output via ``input_output_aliases``.
        if accumulate:
            rhs_in_ref, q_ref, gamma_ref, rhomin_ref, pgmin_ref, dtdx_ref, rhs_out_ref = refs
        else:
            q_ref, gamma_ref, rhomin_ref, pgmin_ref, dtdx_ref, rhs_out_ref = refs
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)

        if ndim == 1:
            ii = (bi * bx + jnp.arange(bx)) % nx
        elif ndim == 2:
            ii = (bi * bx + jnp.arange(bx)[:, None]) % nx
            jj = (bj * by + jnp.arange(by)[None, :]) % ny
        else:
            ii = (bi * bx + jnp.arange(bx)[:, None, None]) % nx
            jj = (bj * by + jnp.arange(by)[None, :, None]) % ny
            kk = (bk * bz + jnp.arange(bz)[None, None, :]) % nz

        gamma = gamma_ref[()]
        gm1 = gamma - 1.0
        rhomin = rhomin_ref[()]
        pgmin = pgmin_ref[()]
        dtdx = dtdx_ref[()]

        def q_at(var_index: int, offset: int):
            if ndim == 1:
                return q_ref[var_index, (ii + offset) % nx]
            if ndim == 2:
                if axis == 0:
                    return q_ref[var_index, (ii + offset) % nx, jj]
                return q_ref[var_index, ii, (jj + offset) % ny]
            if axis == 0:
                return q_ref[var_index, (ii + offset) % nx, jj, kk]
            if axis == 1:
                return q_ref[var_index, ii, (jj + offset) % ny, kk]
            return q_ref[var_index, ii, jj, (kk + offset) % nz]

        def q_local(offset: int):
            return tuple(q_at(idx, offset) for idx in local_indices)

        def primitive_from_q(q):
            rho = q[0]
            mn = q[1]
            if ncomp == 3:
                mt1 = 0.0
                mt2 = 0.0
                energy = q[2]
            elif ncomp == 4:
                mt1 = q[2]
                mt2 = 0.0
                energy = q[3]
            else:
                mt1 = q[2]
                mt2 = q[3]
                energy = q[4]

            inv_rho = 1.0 / rho
            vn = mn * inv_rho
            vt1 = mt1 * inv_rho
            vt2 = mt2 * inv_rho
            v2 = vn * vn + vt1 * vt1 + vt2 * vt2
            pressure = gm1 * (energy - 0.5 * rho * v2)
            return rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure

        def floored_cell(q):
            rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure = primitive_from_q(q)
            troubled = (rho < rhomin) | (pressure < pgmin)
            rho_f = jnp.where(troubled, jnp.maximum(rho, rhomin), rho)
            pressure_f = jnp.where(troubled, jnp.maximum(pressure, pgmin), pressure)
            energy_f = jnp.where(troubled, pressure_f / gm1 + 0.5 * rho_f * v2, energy)
            specific_enthalpy = (energy_f + pressure_f) / rho_f
            sound_speed = jnp.sqrt(jnp.maximum(gamma * jnp.abs(pressure_f / rho_f), 1e-12))
            return rho_f, mn, mt1, mt2, energy_f, vn, vt1, vt2, v2, pressure_f, specific_enthalpy, sound_speed

        def flux_from_q(q):
            rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure = primitive_from_q(q)
            if ncomp == 3:
                return (mn, mn * vn + pressure, (energy + pressure) * vn)
            if ncomp == 4:
                return (mn, mn * vn + pressure, mt1 * vn, (energy + pressure) * vn)
            return (mn, mn * vn + pressure, mt1 * vn, mt2 * vn, (energy + pressure) * vn)

        def lambda_from_floored_cell(cell, mode: int):
            vn = cell[5]
            c = cell[11]
            if mode == 0:
                return vn - c
            if mode == num_modes - 1:
                return vn + c
            return vn

        # Pre-compute the union of the two interface stencils once.  The
        # left interface at ``i - 1/2`` needs cells at offsets ``-3..2`` and
        # the right interface at ``i + 1/2`` needs cells at offsets ``-2..3``,
        # so jointly we need offsets ``-3..3`` — seven cells per output
        # block.  Sharing the heavy ``primitive_from_q`` / ``floored_cell`` /
        # ``flux_from_q`` work across both flux evaluations cuts the
        # per-block compute roughly in half compared to evaluating each
        # interface independently.
        shared_q = tuple(q_local(off) for off in range(-3, 4))
        shared_f = tuple(flux_from_q(q) for q in shared_q)
        shared_floored = tuple(floored_cell(q) for q in shared_q)

        def compute_interface_flux(stencil_offset: int):
            """Compute the WENO interface flux ``F_{i + stencil_offset + 1/2}``.

            ``stencil_offset == 0`` evaluates ``F_{i+1/2}`` (left/right cells at
            offsets 0 and 1); ``stencil_offset == -1`` evaluates ``F_{i-1/2}``
            (left/right cells at offsets -1 and 0).  Returns a tuple of
            ``ncomp`` Pallas tiles, one per local Euler component slot.
            """
            # ``shared_*`` is indexed by absolute offset ``-3..3`` (i.e. slot
            # ``off + 3``).  The WENO stencil for this interface uses the six
            # cells at offsets ``stencil_offset - 2 .. stencil_offset + 3``.
            base = stencil_offset + 3 - 2  # absolute index of the first stencil cell
            q_stencil = tuple(shared_q[base + k] for k in range(6))
            f_stencil = tuple(shared_f[base + k] for k in range(6))
            floored_stencil = tuple(shared_floored[base + k] for k in range(6))

            cell_l = floored_stencil[2]
            cell_r = floored_stencil[3]
            (rho_i, mn_i, mt1_i, mt2_i, energy_i,
             vn_i, vt1_i, vt2_i, v2_i, p_i, h_i, c_i) = cell_l
            (rho_j, mn_j, mt1_j, mt2_j, energy_j,
             vn_j, vt1_j, vt2_j, v2_j, p_j, h_j, c_j) = cell_r
            rho_face = jnp.maximum(
                0.5 * (jnp.maximum(rho_i, rhomin) + jnp.maximum(rho_j, rhomin)),
                rhomin,
            )
            vn_face = 0.5 * (mn_i + mn_j) / rho_face
            vt1_face = 0.5 * (mt1_i + mt1_j) / rho_face
            vt2_face = 0.5 * (mt2_i + mt2_j) / rho_face
            h_face = 0.5 * (h_i + h_j)
            v2_face = vn_face * vn_face + vt1_face * vt1_face + vt2_face * vt2_face
            c2_face = gm1 * (h_face - 0.5 * v2_face)
            c_face = jnp.sqrt(jnp.maximum(c2_face, 1e-12))
            inv_c2 = jnp.where(c2_face > 0.0, 1.0 / c2_face, 0.0)

            def left_project(mode, values):
                if mode == 0:
                    acc = (0.5 * gm1 * v2_face + vn_face * c_face) * values[0]
                    acc = acc - (gm1 * vn_face + c_face) * values[1]
                    if ncomp == 3:
                        acc = acc + gm1 * values[2]
                    elif ncomp == 4:
                        acc = acc - gm1 * vt1_face * values[2] + gm1 * values[3]
                    else:
                        acc = (
                            acc
                            - gm1 * vt1_face * values[2]
                            - gm1 * vt2_face * values[3]
                            + gm1 * values[4]
                        )
                    return 0.5 * inv_c2 * acc

                if mode == 1:
                    acc = (c2_face - 0.5 * gm1 * v2_face) * values[0]
                    acc = acc + gm1 * vn_face * values[1]
                    if ncomp == 3:
                        acc = acc - gm1 * values[2]
                    elif ncomp == 4:
                        acc = acc + gm1 * vt1_face * values[2] - gm1 * values[3]
                    else:
                        acc = (
                            acc
                            + gm1 * vt1_face * values[2]
                            + gm1 * vt2_face * values[3]
                            - gm1 * values[4]
                        )
                    return inv_c2 * acc

                if mode == 2 and ncomp >= 4:
                    return -vt1_face * values[0] + values[2]

                if mode == 3 and ncomp == 5:
                    return -vt2_face * values[0] + values[3]

                acc = (0.5 * gm1 * v2_face - vn_face * c_face) * values[0]
                acc = acc - (gm1 * vn_face - c_face) * values[1]
                if ncomp == 3:
                    acc = acc + gm1 * values[2]
                elif ncomp == 4:
                    acc = acc - gm1 * vt1_face * values[2] + gm1 * values[3]
                else:
                    acc = (
                        acc
                        - gm1 * vt1_face * values[2]
                        - gm1 * vt2_face * values[3]
                        + gm1 * values[4]
                    )
                return 0.5 * inv_c2 * acc

            def add_right_correction(flux_acc, mode, Fs):
                if mode == 0:
                    if ncomp == 3:
                        R = (1.0, vn_face - c_face, h_face - vn_face * c_face)
                    elif ncomp == 4:
                        R = (1.0, vn_face - c_face, vt1_face, h_face - vn_face * c_face)
                    else:
                        R = (1.0, vn_face - c_face, vt1_face, vt2_face, h_face - vn_face * c_face)
                elif mode == 1:
                    if ncomp == 3:
                        R = (1.0, vn_face, 0.5 * v2_face)
                    elif ncomp == 4:
                        R = (1.0, vn_face, vt1_face, 0.5 * v2_face)
                    else:
                        R = (1.0, vn_face, vt1_face, vt2_face, 0.5 * v2_face)
                elif mode == 2 and ncomp >= 4:
                    if ncomp == 4:
                        R = (0.0, 0.0, 1.0, vt1_face)
                    else:
                        R = (0.0, 0.0, 1.0, 0.0, vt1_face)
                elif mode == 3 and ncomp == 5:
                    R = (0.0, 0.0, 0.0, 1.0, vt2_face)
                else:
                    if ncomp == 3:
                        R = (1.0, vn_face + c_face, h_face + vn_face * c_face)
                    elif ncomp == 4:
                        R = (1.0, vn_face + c_face, vt1_face, h_face + vn_face * c_face)
                    else:
                        R = (1.0, vn_face + c_face, vt1_face, vt2_face, h_face + vn_face * c_face)
                return [flux_acc[slot] + R[slot] * Fs for slot in range(ncomp)]

            def alpha_for_mode(mode):
                amx = jnp.abs(lambda_from_floored_cell(floored_stencil[0], mode))
                for k in range(1, 6):
                    amx = jnp.maximum(
                        amx,
                        jnp.abs(lambda_from_floored_cell(floored_stencil[k], mode)),
                    )
                return amx

            flux_acc = [
                (
                    -f_stencil[1][slot]
                    + 7.0 * f_stencil[2][slot]
                    + 7.0 * f_stencil[3][slot]
                    - f_stencil[4][slot]
                )
                * (1.0 / 12.0)
                for slot in range(ncomp)
            ]

            for mode in range(num_modes):
                s = tuple(left_project(mode, f_stencil[k]) for k in range(6))
                qproj = tuple(left_project(mode, q_stencil[k]) for k in range(6))

                d0 = s[1] - s[0]
                d1 = s[2] - s[1]
                d2 = s[3] - s[2]
                d3 = s[4] - s[3]
                d4 = s[5] - s[4]

                dq0 = qproj[1] - qproj[0]
                dq1 = qproj[2] - qproj[1]
                dq2 = qproj[3] - qproj[2]
                dq3 = qproj[4] - qproj[3]
                dq4 = qproj[5] - qproj[4]

                amx = alpha_for_mode(mode)

                aterm_p = 0.5 * (d0 + amx * dq0)
                bterm_p = 0.5 * (d1 + amx * dq1)
                cterm_p = 0.5 * (d2 + amx * dq2)
                dterm_p = 0.5 * (d3 + amx * dq3)

                IS0_p = 13.0 * (aterm_p - bterm_p) ** 2 + 3.0 * (aterm_p - 3.0 * bterm_p) ** 2
                IS1_p = 13.0 * (bterm_p - cterm_p) ** 2 + 3.0 * (bterm_p + cterm_p) ** 2
                IS2_p = 13.0 * (cterm_p - dterm_p) ** 2 + 3.0 * (3.0 * cterm_p - dterm_p) ** 2
                omega0_p, omega2_p = omega_weights(IS0_p, IS1_p, IS2_p, epsilon, tiny)
                second = (
                    omega0_p * (aterm_p - 2.0 * bterm_p + cterm_p) * (1.0 / 3.0)
                    + (omega2_p - 0.5) * (bterm_p - 2.0 * cterm_p + dterm_p) * (1.0 / 6.0)
                )

                aterm_m = 0.5 * (d4 - amx * dq4)
                bterm_m = 0.5 * (d3 - amx * dq3)
                cterm_m = 0.5 * (d2 - amx * dq2)
                dterm_m = 0.5 * (d1 - amx * dq1)

                IS0_m = 13.0 * (aterm_m - bterm_m) ** 2 + 3.0 * (aterm_m - 3.0 * bterm_m) ** 2
                IS1_m = 13.0 * (bterm_m - cterm_m) ** 2 + 3.0 * (bterm_m + cterm_m) ** 2
                IS2_m = 13.0 * (cterm_m - dterm_m) ** 2 + 3.0 * (3.0 * cterm_m - dterm_m) ** 2
                omega0_m, omega2_m = omega_weights(IS0_m, IS1_m, IS2_m, epsilon, tiny)
                third = (
                    omega0_m * (aterm_m - 2.0 * bterm_m + cterm_m) * (1.0 / 3.0)
                    + (omega2_m - 0.5) * (bterm_m - 2.0 * cterm_m + dterm_m) * (1.0 / 6.0)
                )

                Fs = -second + third
                flux_acc = add_right_correction(flux_acc, mode, Fs)

            return flux_acc

        flux_right = compute_interface_flux(0)   # F_{i+1/2}
        flux_left = compute_interface_flux(-1)   # F_{i-1/2}

        # local_indices covers every conserved component for hydro, so a
        # blanket zeroing pass is unnecessary; we set every output slot below.
        if accumulate:
            for slot, var in enumerate(local_indices):
                prior = rhs_in_ref[var, ...]
                rhs_out_ref[var, ...] = prior + (-dtdx) * (flux_right[slot] - flux_left[slot])
        else:
            for slot, var in enumerate(local_indices):
                rhs_out_ref[var, ...] = -dtdx * (flux_right[slot] - flux_left[slot])

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    if accumulate:
        # Same BlockSpec layout as the state/output (full conserved-variable
        # axis, blocked over spatial dims).  XLA is told to reuse the
        # accumulator buffer for the output so the RHS lives in a single
        # physical buffer across all three axis calls.
        rhs_in_spec = pl.BlockSpec(
            block_shape if not isinstance(block_shape, tuple)
            else block_shape,
            (
                (lambda bi, bj, bk: (0, bi))
                if ndim == 1
                else (lambda bi, bj, bk: (0, bi, bj))
                if ndim == 2
                else (lambda bi, bj, bk: (0, bi, bj, bk))
            ),
        )
        in_specs = [rhs_in_spec, in_state_spec, scalar_spec, scalar_spec, scalar_spec, scalar_spec]
        kernel_args = (
            rhs_accumulator,
            conserved_state,
            jnp.asarray(params.gamma, dtype=conserved_state.dtype),
            jnp.asarray(params.minimum_density, dtype=conserved_state.dtype),
            jnp.asarray(params.minimum_pressure, dtype=conserved_state.dtype),
            jnp.asarray(dt_over_dx, dtype=conserved_state.dtype),
        )
        kwargs["input_output_aliases"] = {0: 0}
    else:
        in_specs = [in_state_spec, scalar_spec, scalar_spec, scalar_spec, scalar_spec]
        kernel_args = (
            conserved_state,
            jnp.asarray(params.gamma, dtype=conserved_state.dtype),
            jnp.asarray(params.minimum_density, dtype=conserved_state.dtype),
            jnp.asarray(params.minimum_pressure, dtype=conserved_state.dtype),
            jnp.asarray(dt_over_dx, dtype=conserved_state.dtype),
        )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name=f"hydro_weno_rhs_axis_{axis}",
        **kwargs,
    )(*kernel_args)
