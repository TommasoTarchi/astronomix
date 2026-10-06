"""
Runge-Kutta drivers of the finite-difference solver.

One time step of the hydrodynamics and of the MHD equations with Constrained
Transport (CT), either with the 5-stage, 4th-order Strong Stability Preserving
Runge-Kutta scheme (SSPRK4, Spiteri & Ruuth 2002) or with the 2N-storage
low-storage RK4 (LSRK4, Carpenter & Kennedy 1994). The drivers build the
stage right-hand sides (WENO interface fluxes, optional cold-crush flux
blending, flux divergence, physics sources) and hand them to the generic
schemes in ``_integrators/_explicit_rk.py``; the helpers at the top wrap the
stages for reverse-mode rematerialisation (``SimulationConfig.ad_remat``).

See _magnetic_update/_constrained_transport.py for more details on the
Constrained Transport (CT) implementation following (Seo & Ryu 2023,
https://arxiv.org/abs/2304.04360).
"""

# general
from functools import partial

# typing
from typing import Union

# jax
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

# astronomix constants
from astronomix.option_classes.simulation_config import (
    AD_REMAT_AXIS,
    AD_REMAT_NONE,
    CONSERVATIVE_GAS_STATE,
    GHOST_CELLS,
    IDEAL_GAS,
    MAGNETIC_FIELD_ONLY,
    SIMPLE_SOURCE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_difference._interface_fluxes._weno import (
    _weno_flux_x,
    _weno_flux_y,
    _weno_flux_z,
)
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _hydro_pallas_flux_supported,
    _mhd_pallas_flux_supported,
    _update_cell_center_and_weno_flux_mhd_pallas_keep_halo_x_with_ct_mod,
    _weno_flux_mhd_pallas_keep_halo_x,
)
from astronomix._finite_difference._interface_fluxes._weno_positivity_pallas import (
    mhd_inflow_reference_dispatch,
)
from astronomix._finite_difference._interface_fluxes._flux_blending import (
    _blend_interface_flux,
)
from astronomix._finite_difference._time_integrators._ssprk_pallas import (
    _div_axis_pallas_shape_ok,
    _hydro_flux_div_axis_native,
    _hydro_flux_div_axis_native_from_kept_halo_sharded,
    _hydro_flux_div_axis_pallas,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    _constrained_transport_rhs_from_slices,
    update_cell_center_fields,
)
from astronomix._finite_difference._magnetic_update._constrained_transport_pallas import (
    _ct_rhs_pallas_supported,
    _ct_rhs_pallas_x_precomputed,
)
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._integrators._explicit_rk import lsrk4, ssprk4
from astronomix._modules._time_integrator_sources import _time_integrator_sources
from astronomix._modules._resistivity._resistivity import fd_ohmic_interface_rhs
from astronomix._pallas_helpers import (
    _backend_is_pallas,
    _current_pallas_mesh,
    _current_pallas_spec,
    _pallas_mesh_splits_axis,
    diffable_pallas_call_n,
    pl,
)
from astronomix._stencil_operations._stencil_operations import _shift


def _stage_remat(fn, config):
    """
    Wrap a Runge-Kutta stage function in ``jax.checkpoint`` when reverse-mode
    rematerialisation is requested (``config.ad_remat != "none"``).

    Applied to the right-hand side and to the per-stage boundary and
    cell-centred-field hooks. The backward pass then stores only their inputs
    and recomputes the internals (WENO reconstruction, flux blending,
    divergence, sources) when it needs them: about one extra forward
    evaluation per stage in exchange for not holding every stage's residuals
    across the whole step. Under a plain forward evaluation, or forward-mode
    AD, ``jax.checkpoint`` is inlined and changes nothing.

    Args:
        fn: The stage function.
        config: The simulation configuration.

    Returns:
        ``fn`` itself, or its checkpointed version.
    """
    if config.ad_remat == AD_REMAT_NONE:
        return fn
    return jax.checkpoint(fn)


def _remat_axis_call(
    axis_increment,
    conserved_state,
    dt_over_dx,
    internal_energy_density,
    axis: int,
    chunks: int,
):
    """
    Evaluate the checkpointed increment of one axis (``ad_remat == "axis"``).

    With ``chunks > 1`` (3D only) the increment is evaluated as a
    ``lax.map`` over ``chunks`` slabs of a perpendicular axis, each slab
    checkpointed on its own, so the backward pass holds one slab's WENO and
    blending internals instead of the whole block's. The slabs are cut along
    z for the x and y fluxes and along y for the z flux, never along the
    first spatial axis, which is the axis split over devices.

    The slab evaluation is exact: the WENO flux (its local Lax-Friedrichs
    splitting speed included), the flux blending and the divergence are
    stencils along ``axis`` only, so a slab needs no halo.

    Args:
        axis_increment: ``axis_increment(state, dt_over_dx,
            internal_energy_density) -> (increment, density flux or None)``.
        conserved_state: The conserved state ``(var, x, y, z)``.
        dt_over_dx: The stage time step over the cell size.
        internal_energy_density: The dual-energy ``g`` with shape
            ``(x, y, z)``, or None.
        axis: The spatial axis of the fluxes.
        chunks: The number of slabs.

    Returns:
        The ``(increment, density flux or None)`` pair of the whole block.
    """
    num_spatial_dims = conserved_state.ndim - 1
    slab_axis = 2 if axis != 2 else 1
    if (
        chunks <= 1
        or num_spatial_dims != 3
        or conserved_state.shape[1 + slab_axis] % chunks
    ):
        return jax.checkpoint(axis_increment)(
            conserved_state,
            dt_over_dx,
            internal_energy_density,
        )
    slab_size = conserved_state.shape[1 + slab_axis] // chunks

    # On a multi-device mesh the stacked slabs must keep the (var, x, y, z)
    # sharding of the state, or GSPMD reshards the whole block.
    mesh, spec = _current_pallas_mesh(), _current_pallas_spec()
    state_spec = None
    if mesh is not None and mesh.size > 1 and spec is not None:
        state_spec = tuple(spec) + (None,) * (4 - len(tuple(spec)))

    def constrain(array, num_leading_axes):
        """Keep the (var, x, y, z) layout of the state on the stacked slabs."""
        if state_spec is None:
            return array
        if array.ndim - num_leading_axes == 4:
            slab_spec = (None,) * num_leading_axes + state_spec
        else:
            slab_spec = (None,) * num_leading_axes + state_spec[1:]
        return jax.lax.with_sharding_constraint(
            array,
            NamedSharding(mesh, PartitionSpec(*slab_spec)),
        )

    def split(array):
        """Stack the slabs of ``array`` along a new leading axis."""
        array_slab_axis = array.ndim - 3 + slab_axis
        leading_shape = array.shape[:array_slab_axis]
        trailing_shape = array.shape[array_slab_axis + 1:]
        array = array.reshape(leading_shape + (chunks, slab_size) + trailing_shape)
        return constrain(jnp.moveaxis(array, array_slab_axis, 0), 1)

    def merge(array):
        """Undo ``split`` on one stacked output: move the leading slab index
        back in front of the slab axis and fuse the two."""
        array_slab_axis = array.ndim - 4 + slab_axis
        array = jnp.moveaxis(array, 0, array_slab_axis)
        leading_shape = array.shape[:array_slab_axis]
        trailing_shape = array.shape[array_slab_axis + 2:]
        array = array.reshape(leading_shape + (chunks * slab_size,) + trailing_shape)
        return constrain(array, 0)

    def slab_increment(slab):
        state_slab, internal_energy_slab = slab
        return axis_increment(state_slab, dt_over_dx, internal_energy_slab)

    slab_increment = jax.checkpoint(slab_increment)
    stacked_state = split(conserved_state)
    stacked_internal_energy = (
        None if internal_energy_density is None else split(internal_energy_density)
    )
    stacked_output = jax.lax.map(
        slab_increment,
        (stacked_state, stacked_internal_energy),
    )
    return jax.tree.map(merge, stacked_output)


@partial(
    jax.jit,
    static_argnames=["registered_variables", "config"],
    donate_argnames=["conserved_state", "bx_interface", "by_interface", "bz_interface"],
)
def _ssprk4_with_ct(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params: SimulationParams,
    helper_data: HelperData,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    internal_energy_density=None,
):
    """
    Integrates the MHD equations for one time step using a 5-stage, 4th-order
    Strong Stability Preserving Runge-Kutta (SSPRK) method
    with Constrained Transport (CT).

    Args:
        conserved_state: The conserved state (cell-centred fields included).
        bx_interface: The interface magnetic field B_x.
        by_interface: The interface magnetic field B_y.
        bz_interface: The interface magnetic field B_z.
        gamma: The adiabatic index.
        grid_spacing: The cell size.
        dt: The time step.
        params: The simulation parameters.
        helper_data: The helper data.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The dual-energy ``g`` with shape
            ``(x, y, z)`` (used in the pressure recovery of the fluxes), or
            None.

    Returns:
        The updated ``(conserved_state, bx_interface, by_interface,
        bz_interface)``.
    """

    # Processes with time scales similar to or shorter than the hydrodynamics
    # should be included as source terms in the RK stages; slower ones could be
    # handled outside.

    # The per-axis divergence kernel (with its ``input_output_aliases`` memory
    # saving) only operates on whatever flux tensor it is handed, so it serves
    # the MHD equations as well whenever the Pallas backend is selected.
    use_pallas_div = (
        _backend_is_pallas(config) and pl is not None
        and _div_axis_pallas_shape_ok(conserved_state, config)
    )

    def rhs(u, dt_tilde):
        """
        Computes the right-hand side (RHS) of the MHD equations for a given stage.
        ``dt_tilde`` is the stage-effective step (``k * dt``); the state pytree
        ``u`` is the ``(q, bx, by, bz)`` tuple.
        """

        current_q, bx, by, bz = u

        current_q = update_cell_center_fields(
            current_q, bx, by, bz, config, registered_variables
        )

        # For ideal MHD with the positivity-preserving WENO, compute each
        # cell's axis-summed first-order inflow, so that the inflow faces of a
        # cell can be limited jointly (see _weno_positivity.py).
        inflow_reference = None
        if (
            config.weno_positivity_preserving
            and config.equation_of_state == IDEAL_GAS
            and internal_energy_density is None
        ):
            inflow_reference = mhd_inflow_reference_dispatch(
                current_q,
                params,
                config,
                registered_variables,
            )

        # in the future we might support
        # different grid spacings in each direction
        dtdx = dt_tilde / grid_spacing
        dtdy = dt_tilde / grid_spacing
        dtdz = dt_tilde / grid_spacing

        # Axis-incremental flow: build each axis's full dF, extract the
        # two magnetic-flux slices CT needs (plus the density-flux slice
        # for any physics modules that consume it), consume dF for the
        # divergence step, then free dF.  CT runs at the end on the six
        # small single-channel slices instead of the three full 8-var dF
        # arrays — saves ~7/8 × 3 = 2.6× state-shape buffers at peak.
        magnetic_index = registered_variables.magnetic_index
        density_index = registered_variables.density_index

        # Cold-crush flux blending (radiatively cooled crushes; see _flux_blending):
        # apply to the full interface flux BEFORE the transverse magnetic-flux
        # slices are extracted, so CT consumes the blended (locally-diffusive)
        # induction flux. CT stays div(B)=0 by construction (single-valued edge
        # EMFs from consistent face fluxes).
        blend = config.positivity_config.coldcrush_blend

        # --------------- ↓ x-axis ↓ ----------------
        dF_x = _weno_flux_x(
            current_q,
            params,
            config,
            registered_variables,
            internal_energy_density=internal_energy_density,
            inflow_reference=inflow_reference,
        )
        if blend:
            dF_x = _blend_interface_flux(
                dF_x,
                current_q,
                0,
                params,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
            )
        By_flux_x = dF_x[magnetic_index.y]
        Bz_flux_x = dF_x[magnetic_index.z]
        density_flux_x = dF_x[density_index]
        if use_pallas_div:
            rhs_q = _hydro_flux_div_axis_pallas(dF_x, dtdx, config, axis=0)
        else:
            rhs_q = -dtdx * (dF_x - _shift(dF_x, 1, axis=1))
        del dF_x
        # --------------- ↑ x-axis ↑ ----------------

        # --------------- ↓ y-axis ↓ ----------------
        if config.dimensionality >= 2:
            dF_y = _weno_flux_y(
                current_q,
                params,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
                inflow_reference=inflow_reference,
            )
            if blend:
                dF_y = _blend_interface_flux(
                    dF_y,
                    current_q,
                    1,
                    params,
                    config,
                    registered_variables,
                    internal_energy_density=internal_energy_density,
                )
            Bx_flux_y = dF_y[magnetic_index.x]
            Bz_flux_y = dF_y[magnetic_index.z]
            density_flux_y = dF_y[density_index]
            if use_pallas_div:
                rhs_q = _hydro_flux_div_axis_pallas(
                    dF_y, dtdy, config, axis=1, rhs_accumulator=rhs_q
                )
            else:
                rhs_q = rhs_q - dtdy * (dF_y - _shift(dF_y, 1, axis=2))
            del dF_y
        else:
            Bx_flux_y = 0.0
            Bz_flux_y = 0.0
        # --------------- ↑ y-axis ↑ ----------------

        # --------------- ↓ z-axis ↓ ----------------
        if config.dimensionality == 3:
            dF_z = _weno_flux_z(
                current_q,
                params,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
                inflow_reference=inflow_reference,
            )
            if blend:
                dF_z = _blend_interface_flux(
                    dF_z,
                    current_q,
                    2,
                    params,
                    config,
                    registered_variables,
                    internal_energy_density=internal_energy_density,
                )
            Bx_flux_z = dF_z[magnetic_index.x]
            By_flux_z = dF_z[magnetic_index.y]
            density_flux_z = dF_z[density_index]
            if use_pallas_div:
                rhs_q = _hydro_flux_div_axis_pallas(
                    dF_z, dtdz, config, axis=2, rhs_accumulator=rhs_q
                )
            else:
                rhs_q = rhs_q - dtdz * (dF_z - _shift(dF_z, 1, axis=3))
            del dF_z
        else:
            Bx_flux_z = 0.0
            By_flux_z = 0.0
        # --------------- ↑ z-axis ↑ ----------------

        # CT now runs on the six single-channel B-flux slices only — the
        # three 8-var dF arrays have all been freed by this point.
        rhs_bx, rhs_by, rhs_bz = _constrained_transport_rhs_from_slices(
            current_q,
            By_flux_x,
            Bz_flux_x,
            Bx_flux_y,
            Bz_flux_y,
            Bx_flux_z,
            By_flux_z,
            dtdx,
            dtdy,
            dtdz,
            config,
            registered_variables,
        )
        # Explicit ohmic resistivity: a further curl of an edge EMF on the
        # interface fields, so div(B) = 0 is kept exactly (see _resistivity).
        if config.resistivity:
            resistive_rhs_bx, resistive_rhs_by, resistive_rhs_bz = fd_ohmic_interface_rhs(
                bx,
                by,
                bz,
                params.resistivity,
                dt_tilde,
                grid_spacing,
                config,
            )
            rhs_bx = rhs_bx + resistive_rhs_bx
            rhs_by = rhs_by + resistive_rhs_by
            rhs_bz = rhs_bz + resistive_rhs_bz

        if config.dimensionality == 1:
            density_fluxes = (density_flux_x,)
        elif config.dimensionality == 2:
            density_fluxes = (density_flux_x, density_flux_y)
        else:
            density_fluxes = (density_flux_x, density_flux_y, density_flux_z)

        # Add the physics source terms; the density increment is handed in
        # separately for the modules that need it.
        density_increment = rhs_q[registered_variables.density_index]
        rhs_q += _time_integrator_sources(
            current_q,
            density_fluxes,
            density_increment,
            dt_tilde,
            gamma,
            config,
            params,
            helper_data,
            registered_variables,
        )

        return rhs_q, rhs_bx, rhs_by, rhs_bz

    rhs = _stage_remat(rhs, config)

    def pre_stage(u):
        q, bx, by, bz = u
        if config.boundary_handling == GHOST_CELLS:
            q = _boundary_handler(
                q, config, registered_variables, params, CONSERVATIVE_GAS_STATE
            )
            b_curr = _boundary_handler(
                jnp.stack([bx, by, bz], axis=0),
                config,
                registered_variables,
                params,
                MAGNETIC_FIELD_ONLY,
            )
            bx, by, bz = b_curr[0], b_curr[1], b_curr[2]
        return (q, bx, by, bz)

    def finalize(u):
        q, bx, by, bz = u
        # Update the cell-centered magnetic fields in the conserved state array
        # from the final interface magnetic fields.
        q = update_cell_center_fields(
            q, bx, by, bz, config, registered_variables
        )
        return (q, bx, by, bz)

    def post_stage(u):
        # The increment of a stage is evaluated at the state with its
        # cell-centred B rebuilt from the faces (start of ``rhs``) but is
        # added to the stored state, whose B and E are the previous stage's
        # WENO update. The two have the same pressure, yet the sum moves it by
        # (gamma - 1) lambda dF_B . (B_stored - B_faces), of no definite sign
        # and large at low beta. Rebuilding the stored stage state (pressure
        # held) makes every increment start from the state it was evaluated
        # at, which the positivity-preserving WENO proof needs.
        q, bx, by, bz = u
        q = update_cell_center_fields(q, bx, by, bz, config, registered_variables)
        return (q, bx, by, bz)

    # Only the positivity-preserving ideal-MHD scheme needs the stage states
    # resynchronised; everywhere else ``ssprk4`` keeps its identity hook.
    stage_hooks = {"pre_stage": _stage_remat(pre_stage, config)}
    if config.weno_positivity_preserving and config.equation_of_state == IDEAL_GAS:
        stage_hooks["post_stage"] = _stage_remat(post_stage, config)
    return ssprk4(
        (conserved_state, bx_interface, by_interface, bz_interface),
        dt,
        rhs=rhs,
        **stage_hooks,
        finalize=_stage_remat(finalize, config),
    )


def _hydro_density_fluxes_needed(config) -> bool:
    """
    Whether any FD physics module actually consumes the per-axis density
    flux slices.  Only self-gravity variants other than SIMPLE_SOURCE do,
    so for typical setups (hydrodynamics only / wind / cooling without
    flux-coupled gravity) the standalone density flux arrays can be skipped
    and the fused Pallas WENO+divergence path is safe.

    Args:
        config: The simulation configuration.

    Returns:
        True if the density fluxes must be kept.
    """
    return config.gravity_config.gravity and (
        config.gravity_config.self_gravity_version != SIMPLE_SOURCE
    )


def _hydro_step_rhs(
    current_q,
    dt_tilde,
    *,
    params,
    config,
    registered_variables,
    gamma,
    grid_spacing,
    helper_data,
    density_fluxes_needed: bool,
    internal_energy_density=None,
):
    """
    RHS for one hydro WENO time-step stage (excluding RK coefficient logic),
    ``rhs_q = -dt_tilde * div(F(current_q)) + dt_tilde * S(current_q)``.

    Shared by the SSPRK4 and LSRK4 (low-storage) integrators below; the only
    integrator-specific code is the way ``dt_tilde`` is built and how each
    stage's update accumulates ``rhs_q`` back into the running state.

    Args:
        current_q: The conserved state of the stage.
        dt_tilde: The stage-effective step (``k * dt``).
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        gamma: The adiabatic index.
        grid_spacing: The cell size.
        helper_data: The helper data.
        density_fluxes_needed: Whether a physics module consumes the per-axis
            density fluxes (see ``_hydro_density_fluxes_needed``).
        internal_energy_density: The dual-energy ``g`` with shape
            ``(x, y, z)`` (used in the pressure recovery of the fluxes), or
            None.

    Returns:
        The stage increment ``rhs_q``.
    """
    dtdx = dt_tilde / grid_spacing
    dtdy = dt_tilde / grid_spacing
    dtdz = dt_tilde / grid_spacing

    # Fused WENO + axis-flux-divergence: each axis is built and consumed one
    # at a time, so the full-state-sized ``dF_x/y/z`` temporaries never
    # coexist.  Falls back to the explicit flux + divergence path when (a)
    # Pallas is unavailable / unsupported or (b) a physics module needs the
    # standalone density flux slices.
    # The cold-crush blend post-processes each assembled interface flux before
    # the divergence, so it needs the standalone per-axis flux and cannot use
    # the fused WENO+divergence kernel either.
    use_fused_pallas = (
        _hydro_pallas_flux_supported(current_q, config)
        and not density_fluxes_needed
        and not config.positivity_config.coldcrush_blend
    )

    if use_fused_pallas:
        # Compute each axis flux with the standard (1-flux-per-cell) WENO
        # kernel, then immediately consume it via a per-axis divergence
        # kernel that accumulates into ``rhs_q`` in place (via the kernel's
        # ``input_output_aliases``).  This keeps WENO compute unchanged
        # relative to the original Pallas path while ensuring all three
        # ``dF`` temporaries never coexist and the rhs lives in a single
        # physical buffer across axes.
        dF_x = _weno_flux_x(
            current_q,
            params,
            config,
            registered_variables,
            internal_energy_density=internal_energy_density,
        )
        rhs_q = _hydro_flux_div_axis_pallas(dF_x, dtdx, config, axis=0)
        del dF_x

        if config.dimensionality >= 2:
            dF_y = _weno_flux_y(
                current_q,
                params,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
            )
            rhs_q = _hydro_flux_div_axis_pallas(
                dF_y, dtdy, config, axis=1, rhs_accumulator=rhs_q
            )
            del dF_y

        if config.dimensionality == 3:
            dF_z = _weno_flux_z(
                current_q,
                params,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
            )
            rhs_q = _hydro_flux_div_axis_pallas(
                dF_z, dtdz, config, axis=2, rhs_accumulator=rhs_q
            )
            del dF_z

        density_fluxes = None
    elif config.ad_remat == AD_REMAT_AXIS:
        # Reverse-mode rematerialisation per AXIS (config.ad_remat == "axis"):
        # the same per-axis flux + blend + divergence as the path below, each
        # axis inside its own jax.checkpoint, so the backward of a stage holds
        # one axis' WENO / blend internals at a time instead of all three.
        blend = config.positivity_config.coldcrush_blend
        weno_fluxes = (_weno_flux_x, _weno_flux_y, _weno_flux_z)

        def axis_increment(state, dt_over_dx, internal_energy, axis):
            interface_flux = weno_fluxes[axis](
                state,
                params,
                config,
                registered_variables,
                internal_energy_density=internal_energy,
            )
            if blend:
                interface_flux = _blend_interface_flux(
                    interface_flux,
                    state,
                    axis,
                    params,
                    config,
                    registered_variables,
                    internal_energy_density=internal_energy,
                )
            flux_divergence_increment = -dt_over_dx * (
                interface_flux - _shift(interface_flux, 1, axis=axis + 1)
            )
            if density_fluxes_needed:
                return flux_divergence_increment, interface_flux[registered_variables.density_index]
            return flux_divergence_increment, None

        dt_over_dx_per_axis = (dtdx, dtdy, dtdz)
        rhs_q = None
        density_fluxes = [] if density_fluxes_needed else None
        for axis in range(config.dimensionality):
            flux_divergence_increment, density_flux = _remat_axis_call(
                partial(axis_increment, axis=axis),
                current_q,
                dt_over_dx_per_axis[axis],
                internal_energy_density,
                axis,
                int(config.ad_remat_chunks),
            )
            if rhs_q is None:
                rhs_q = flux_divergence_increment
            else:
                rhs_q = rhs_q + flux_divergence_increment
            if density_fluxes_needed:
                density_fluxes.append(density_flux)
        if density_fluxes_needed:
            density_fluxes = tuple(density_fluxes)
    else:
        # Per-axis flux + divergence path.  Accumulate axis-by-axis rather
        # than holding all three flux arrays live simultaneously, so XLA
        # can reuse buffers between axes.
        blend = config.positivity_config.coldcrush_blend
        dF_x = _weno_flux_x(
            current_q,
            params,
            config,
            registered_variables,
            internal_energy_density=internal_energy_density,
        )
        if blend:
            dF_x = _blend_interface_flux(
                dF_x,
                current_q,
                0,
                params,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
            )
        rhs_q = -dtdx * (dF_x - _shift(dF_x, 1, axis=1))
        if density_fluxes_needed:
            density_fluxes = [dF_x[registered_variables.density_index]]
        else:
            density_fluxes = None
        del dF_x

        if config.dimensionality >= 2:
            dF_y = _weno_flux_y(
                current_q,
                params,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
            )
            if blend:
                dF_y = _blend_interface_flux(
                    dF_y,
                    current_q,
                    1,
                    params,
                    config,
                    registered_variables,
                    internal_energy_density=internal_energy_density,
                )
            rhs_q = rhs_q - dtdy * (dF_y - _shift(dF_y, 1, axis=2))
            if density_fluxes_needed:
                density_fluxes.append(dF_y[registered_variables.density_index])
            del dF_y

        if config.dimensionality == 3:
            dF_z = _weno_flux_z(
                current_q,
                params,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
            )
            if blend:
                dF_z = _blend_interface_flux(
                    dF_z,
                    current_q,
                    2,
                    params,
                    config,
                    registered_variables,
                    internal_energy_density=internal_energy_density,
                )
            rhs_q = rhs_q - dtdz * (dF_z - _shift(dF_z, 1, axis=3))
            if density_fluxes_needed:
                density_fluxes.append(dF_z[registered_variables.density_index])
            del dF_z

        if density_fluxes_needed:
            density_fluxes = tuple(density_fluxes)

    # Add the physics source terms; the density increment is handed in
    # separately for the modules that need it.
    density_increment = rhs_q[registered_variables.density_index]
    rhs_q += _time_integrator_sources(
        current_q,
        density_fluxes,
        density_increment,
        dt_tilde,
        gamma,
        config,
        params,
        helper_data,
        registered_variables,
    )

    return rhs_q


@partial(
    jax.jit,
    static_argnames=["registered_variables", "config"],
    donate_argnames=["conserved_state"],
)
def _ssprk4_hydro(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params: SimulationParams,
    helper_data: HelperData,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    internal_energy_density=None,
):
    """
    Integrates the Euler (hydrodynamics) equations for one time step using a
    5-stage, 4th-order Strong Stability Preserving Runge-Kutta (SSPRK) method.

    Three-register Spiteri-Ruuth scheme: needs ``q0``, ``q_curr`` and
    ``q_final`` simultaneously.  For storage-constrained runs, the
    ``_lsrk4_hydro`` 2-register Carpenter-Kennedy LSRK4 is available below
    via ``time_integrator=RK4_LSRK``.

    Args:
        conserved_state: The conserved state.
        gamma: The adiabatic index.
        grid_spacing: The cell size.
        dt: The time step.
        params: The simulation parameters.
        helper_data: The helper data.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The dual-energy ``g`` with shape
            ``(x, y, z)`` (used in the pressure recovery of the fluxes), or
            None.

    Returns:
        The updated conserved state.
    """

    # Processes with time scales similar to or shorter than the hydrodynamics
    # should be included as source terms in the RK stages; slower ones could be
    # handled outside.

    density_fluxes_needed = _hydro_density_fluxes_needed(config)

    def pre_stage(q):
        if config.boundary_handling == GHOST_CELLS:
            q = _boundary_handler(
                q, config, registered_variables, params, CONSERVATIVE_GAS_STATE
            )
        return q

    def rhs(q, dt_stage):
        return _hydro_step_rhs(
            q,
            dt_stage,
            params=params,
            config=config,
            registered_variables=registered_variables,
            gamma=gamma,
            grid_spacing=grid_spacing,
            helper_data=helper_data,
            density_fluxes_needed=density_fluxes_needed,
            internal_energy_density=internal_energy_density,
        )

    rhs = _stage_remat(rhs, config)

    return ssprk4(
        conserved_state,
        dt,
        rhs=rhs,
        pre_stage=_stage_remat(pre_stage, config),
    )


@partial(
    jax.jit,
    static_argnames=["registered_variables", "config"],
    donate_argnames=["conserved_state"],
)
def _lsrk4_hydro(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params: SimulationParams,
    helper_data: HelperData,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    internal_energy_density=None,
):
    """
    Carpenter-Kennedy 2N-storage, 5-stage, 4th-order low-storage RK4.

    The integrator carries two full-state registers (``q`` and ``dq``)
    instead of the three (``q0``, ``q_curr``, ``q_final``) required by the
    SSPRK4 Spiteri-Ruuth scheme above, which saves one full conserved-state
    buffer at peak on top of the memory savings of the fused WENO /
    divergence Pallas kernels.

    The trade-off is a smaller linear-stability CFL than SSPRK4 (the user
    should expect roughly half of the 1.5 that SSPRK4 tolerates with the
    5th-order WENO scheme); LSRK4 has no SSP property either, so very strong
    shocks may need a slightly tighter limiter / floor than SSPRK4 to avoid
    sporadic non-monotone overshoots.

    Args:
        conserved_state: The conserved state.
        gamma: The adiabatic index.
        grid_spacing: The cell size.
        dt: The time step.
        params: The simulation parameters.
        helper_data: The helper data.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The dual-energy ``g`` with shape
            ``(x, y, z)`` (used in the pressure recovery of the fluxes), or
            None.

    Returns:
        The updated conserved state.
    """

    density_fluxes_needed = _hydro_density_fluxes_needed(config)

    dtdx = dt / grid_spacing
    dtdy = dt / grid_spacing
    dtdz = dt / grid_spacing

    def pre_stage(q):
        if config.boundary_handling == GHOST_CELLS:
            q = _boundary_handler(
                q, config, registered_variables, params, CONSERVATIVE_GAS_STATE
            )
        return q

    def lsrk_increment(q, dq, a_coef, dt_step):
        # Fused path: write the LSRK4 ``dq_new = A[i] * dq + dt * L(q)``
        # update directly into the ``dq`` buffer using the per-axis
        # divergence kernel's ``input_output_aliases``.  The
        # rhs/``L(q)``-sized scratch register is never materialised, which is
        # what gets us below the 3-buffer floor of the explicit
        # rhs-then-update path.
        # The fused divergence path never calls ``_blend_interface_flux``, so
        # the cold-crush blend must force the explicit rhs route.
        use_fused_pallas = (
            _hydro_pallas_flux_supported(q, config)
            and not density_fluxes_needed
            and not config.positivity_config.coldcrush_blend
        )

        if use_fused_pallas:
            dF_x = _weno_flux_x(
                q,
                params,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
            )
            dq = _hydro_flux_div_axis_pallas(
                dF_x,
                dtdx,
                config,
                axis=0,
                rhs_accumulator=dq,
                scale_in=a_coef,
            )
            del dF_x

            if config.dimensionality >= 2:
                dF_y = _weno_flux_y(
                    q,
                    params,
                    config,
                    registered_variables,
                    internal_energy_density=internal_energy_density,
                )
                dq = _hydro_flux_div_axis_pallas(
                    dF_y,
                    dtdy,
                    config,
                    axis=1,
                    rhs_accumulator=dq,
                    scale_in=1.0,
                )
                del dF_y

            if config.dimensionality == 3:
                dF_z = _weno_flux_z(
                    q,
                    params,
                    config,
                    registered_variables,
                    internal_energy_density=internal_energy_density,
                )
                dq = _hydro_flux_div_axis_pallas(
                    dF_z,
                    dtdz,
                    config,
                    axis=2,
                    rhs_accumulator=dq,
                    scale_in=1.0,
                )
                del dF_z

            # Physics source terms.  Sedov-style hydro with no active modules
            # makes this a no-op (``_time_integrator_sources`` returns zeros); for
            # active modules the dt-scaled source is added on top of the
            # already-scaled ``A[i] * dq + dt * L(q)`` value in ``dq``.
            sources = _time_integrator_sources(
                q,
                None,
                dq[registered_variables.density_index],
                dt_step,
                gamma,
                config,
                params,
                helper_data,
                registered_variables,
            )
            if sources is not None:
                dq = dq + sources
            return dq

        # Fallback: explicit ``rhs = dt * L(q)`` then ``dq = A * dq + rhs``.
        rhs = _hydro_step_rhs(
            q,
            dt_step,
            params=params,
            config=config,
            registered_variables=registered_variables,
            gamma=gamma,
            grid_spacing=grid_spacing,
            helper_data=helper_data,
            density_fluxes_needed=density_fluxes_needed,
            internal_energy_density=internal_energy_density,
        )
        return a_coef * dq + rhs

    lsrk_increment = _stage_remat(lsrk_increment, config)

    return lsrk4(
        conserved_state,
        dt,
        pre_stage=_stage_remat(pre_stage, config),
        lsrk_increment=lsrk_increment,
    )


@partial(
    jax.jit,
    static_argnames=["registered_variables", "config"],
    donate_argnames=["conserved_state", "bx_interface", "by_interface", "bz_interface"],
)
def _lsrk4_with_ct(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params: SimulationParams,
    helper_data: HelperData,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    internal_energy_density=None,
):
    """
    Carpenter-Kennedy 2N-storage 5-stage 4th-order LSRK4 for MHD-CT.

    Mirrors ``_lsrk4_hydro`` but carries the four MHD register pairs that
    ``_ssprk4_with_ct``'s Spiteri-Ruuth scheme used as three-register triples:

      * ``(q, dq)`` for the conserved state (8 vars),
      * ``(bx, dbx)``, ``(by, dby)``, ``(bz, dbz)`` for the three interface
        magnetic-field components.

    Compared to the SSPRK4 carry ``(q0, q_curr, q_final)`` plus the three
    ``(bx0, bx_curr, bx_final)`` triples this saves one full conserved
    register plus three interface-B registers.

    Trade-off: linear-stability CFL drops from SSPRK4's ~1.5 to roughly 1.4
    and LSRK4 has no SSP property — same caveats as ``_lsrk4_hydro``.
    Selected via ``config.time_integrator == RK4_LSRK``.

    Args:
        conserved_state: The conserved state (cell-centred fields included).
        bx_interface: The interface magnetic field B_x.
        by_interface: The interface magnetic field B_y.
        bz_interface: The interface magnetic field B_z.
        gamma: The adiabatic index.
        grid_spacing: The cell size.
        dt: The time step.
        params: The simulation parameters.
        helper_data: The helper data.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The dual-energy ``g`` with shape
            ``(x, y, z)`` (used in the pressure recovery of the fluxes), or
            None.

    Returns:
        The updated ``(conserved_state, bx_interface, by_interface,
        bz_interface)``.
    """

    use_pallas_div = (
        _backend_is_pallas(config) and pl is not None
        and _div_axis_pallas_shape_ok(conserved_state, config)
    )

    # Multi-GPU fast path for x split over devices. The x-WENO kernel keeps one
    # x-halo cell of its flux, so the x divergence needs no second halo
    # exchange; the x parts of the constrained-transport modified fluxes are
    # computed in the same shard_map, and with the Pallas CT kernels the
    # cell-centred field update is fused into it as well. The fused kernels
    # implement the plain WENO flux only, so positivity-preserving WENO, dual
    # energy and the cold-crush blend take the standard path.
    use_kept_x_flux_halo = (
        use_pallas_div
        and int(config.dimensionality) == 3
        and _pallas_mesh_splits_axis(conserved_state, 0)
        and _mhd_pallas_flux_supported(conserved_state, config)
        and not config.weno_positivity_preserving
        and internal_energy_density is None
        and not config.positivity_config.coldcrush_blend
    )
    fuse_cell_center_update = (
        use_kept_x_flux_halo and _ct_rhs_pallas_supported(conserved_state, config)
    )

    # With y and z unsplit the y and z divergences are purely shard-local and
    # cheaper in plain JAX than as separate Pallas calls.
    yz_unsplit = not (
        _pallas_mesh_splits_axis(conserved_state, 1)
        or _pallas_mesh_splits_axis(conserved_state, 2)
    )

    def stage_rhs(
        kept_halo_path,
        current_q,
        bx,
        by,
        bz,
        dq,
        a_coef,
        dt,
        gamma,
        grid_spacing,
        params,
        helper_data,
    ):
        """Compute ``dq_new = a_coef * dq + dt * L_q`` (in-place via the
        Pallas div-axis accumulator when available) and the three
        interface-B ``dt * L_b{x,y,z}`` increments.

        The fused conserved-state path matches ``_lsrk4_hydro``: each
        axis's divergence kernel folds ``a_coef * dq + (-dt/dx) * div``
        directly into the ``dq`` register, so the LSRK4 update never
        materialises a separate ``rhs_q``.  When Pallas is unavailable
        we fall back to the explicit
        ``rhs_q`` → ``dq = a_coef * dq + rhs_q`` pattern.

        ``kept_halo_path`` (static) selects the multi-GPU fast path described
        above; both paths compute the same stage.
        """
        dtdx = dt / grid_spacing
        dtdy = dt / grid_spacing
        dtdz = dt / grid_spacing
        fuse_update = kept_halo_path and fuse_cell_center_update

        if not fuse_update:
            current_q = update_cell_center_fields(
                current_q, bx, by, bz, config, registered_variables
            )

        # For ideal MHD with the positivity-preserving WENO, compute each
        # cell's axis-summed first-order inflow, so that the inflow faces of a
        # cell can be limited jointly (see _weno_positivity.py).
        inflow_reference = None
        if (
            config.weno_positivity_preserving
            and config.equation_of_state == IDEAL_GAS
            and internal_energy_density is None
        ):
            inflow_reference = mhd_inflow_reference_dispatch(
                current_q,
                params,
                config,
                registered_variables,
            )

        # Axis-incremental flow — see the matching SSPRK4-with-CT path
        # above for the rationale.  Each axis's full dF is built, the
        # two magnetic-flux slices CT needs are extracted, dF is consumed
        # for the divergence step (folding ``a_coef * dq`` in for the
        # first axis), then freed.  CT runs on the six small slices only.
        magnetic_index = registered_variables.magnetic_index
        density_index = registered_variables.density_index

        # Cold-crush flux blending, as in the SSPRK4-with-CT path: applied to
        # the full interface flux before the magnetic-flux slices are
        # extracted, so CT consumes the blended induction flux.
        blend = config.positivity_config.coldcrush_blend

        # -------------------------------------------------------------
        # ======================= ↓ x-axis ↓ ==========================
        # -------------------------------------------------------------

        # Fold the LSRK4 ``a_coef * dq + ...`` step into the first axis's
        # divergence kernel via ``scale_in`` so ``rhs_q`` is never
        # materialised; subsequent axes accumulate (scale_in = 1.0).  The
        # native fallback path keeps the explicit ``rhs_q`` register.
        flux_x_modified = None
        flux_x_with_halo = None
        if fuse_update:
            current_q, flux_x_with_halo, dF_x, flux_x_modified = (
                _update_cell_center_and_weno_flux_mhd_pallas_keep_halo_x_with_ct_mod(
                    current_q,
                    bx,
                    by,
                    bz,
                    params,
                    config,
                    registered_variables,
                )
            )
        elif kept_halo_path:
            flux_x_with_halo, dF_x = _weno_flux_mhd_pallas_keep_halo_x(
                current_q,
                params,
                config,
                registered_variables,
            )
        else:
            dF_x = _weno_flux_x(
                current_q,
                params,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
                inflow_reference=inflow_reference,
            )
            if blend:
                dF_x = _blend_interface_flux(
                    dF_x,
                    current_q,
                    0,
                    params,
                    config,
                    registered_variables,
                    internal_energy_density=internal_energy_density,
                )
        By_flux_x = dF_x[magnetic_index.y]
        Bz_flux_x = dF_x[magnetic_index.z]
        density_flux_x = dF_x[density_index]
        if kept_halo_path:
            dq = _hydro_flux_div_axis_native_from_kept_halo_sharded(
                flux_x_with_halo,
                dtdx,
                config,
                axis=0,
                rhs_accumulator=dq,
                scale_in=a_coef,
                kept_halo=1,
            )
            rhs_q_for_phys = None
        elif use_pallas_div:
            dq = _hydro_flux_div_axis_pallas(
                dF_x,
                dtdx,
                config,
                axis=0,
                rhs_accumulator=dq,
                scale_in=a_coef,
            )
            rhs_q_for_phys = None
        else:
            rhs_q_for_phys = -dtdx * (dF_x - _shift(dF_x, 1, axis=1))
        del dF_x, flux_x_with_halo

        # Serialize the per-axis flux passes.  ``dF_{x,y,z}`` each depend only
        # on ``current_q`` (not on ``dq`` or each other), so without an explicit
        # dependency XLA is free to schedule all three axis reconstructions —
        # and their halo-padded copies of the 8-var flux and the full state —
        # concurrently, tripling the transient peak.  Routing ``current_q`` into
        # the next axis *through* the just-updated accumulator forces the x-axis
        # flux buffer to be freed before the y-axis one is built, holding a
        # single axis live at a time.  ``optimization_barrier`` is a numerical
        # no-op (bit-identical output); at production grid sizes each axis kernel
        # already saturates the GPU, so the lost cross-axis overlap costs little
        # throughput (a few per cent in multi-GPU runs that are not memory
        # bound) while cutting the RHS peak footprint ~3x.
        if use_pallas_div:
            current_q, dq = jax.lax.optimization_barrier((current_q, dq))
        else:
            current_q, rhs_q_for_phys = jax.lax.optimization_barrier(
                (current_q, rhs_q_for_phys)
            )

        # -------------------------------------------------------------
        # ======================= ↑ x-axis ↑ ==========================
        # -------------------------------------------------------------

        # On the fast path with y and z unsplit, the y and z divergences are
        # shard-local plain-JAX updates of the accumulator.
        native_yz_divergence = kept_halo_path and yz_unsplit

        # -------------------------------------------------------------
        # ======================= ↓ y-axis ↓ ==========================
        # -------------------------------------------------------------

        if config.dimensionality >= 2:
            dF_y = _weno_flux_y(
                current_q,
                params,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
                inflow_reference=inflow_reference,
            )
            if blend:
                dF_y = _blend_interface_flux(
                    dF_y,
                    current_q,
                    1,
                    params,
                    config,
                    registered_variables,
                    internal_energy_density=internal_energy_density,
                )
            Bx_flux_y = dF_y[magnetic_index.x]
            Bz_flux_y = dF_y[magnetic_index.z]
            density_flux_y = dF_y[density_index]
            if native_yz_divergence:
                dq = _hydro_flux_div_axis_native(
                    dF_y,
                    dtdy,
                    axis=1,
                    rhs_accumulator=dq,
                )
            elif use_pallas_div:
                dq = _hydro_flux_div_axis_pallas(
                    dF_y,
                    dtdy,
                    config,
                    axis=1,
                    rhs_accumulator=dq,
                )
            else:
                rhs_q_for_phys = rhs_q_for_phys - dtdy * (dF_y - _shift(dF_y, 1, axis=2))
            del dF_y

            # Same serialization barrier between the y and z axes (see above).
            if use_pallas_div:
                current_q, dq = jax.lax.optimization_barrier((current_q, dq))
            else:
                current_q, rhs_q_for_phys = jax.lax.optimization_barrier(
                    (current_q, rhs_q_for_phys)
                )
        else:
            Bx_flux_y = 0.0
            Bz_flux_y = 0.0

        # -------------------------------------------------------------
        # ======================= ↑ y-axis ↑ ==========================
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        # ======================= ↓ z-axis ↓ ==========================
        # -------------------------------------------------------------

        if config.dimensionality == 3:
            dF_z = _weno_flux_z(
                current_q,
                params,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
                inflow_reference=inflow_reference,
            )
            if blend:
                dF_z = _blend_interface_flux(
                    dF_z,
                    current_q,
                    2,
                    params,
                    config,
                    registered_variables,
                    internal_energy_density=internal_energy_density,
                )
            Bx_flux_z = dF_z[magnetic_index.x]
            By_flux_z = dF_z[magnetic_index.y]
            density_flux_z = dF_z[density_index]
            if native_yz_divergence:
                dq = _hydro_flux_div_axis_native(
                    dF_z,
                    dtdz,
                    axis=2,
                    rhs_accumulator=dq,
                )
            elif use_pallas_div:
                dq = _hydro_flux_div_axis_pallas(
                    dF_z,
                    dtdz,
                    config,
                    axis=2,
                    rhs_accumulator=dq,
                )
            else:
                rhs_q_for_phys = rhs_q_for_phys - dtdz * (dF_z - _shift(dF_z, 1, axis=3))
            del dF_z
        else:
            Bx_flux_z = 0.0
            By_flux_z = 0.0

        # -------------------------------------------------------------
        # ======================= ↑ z-axis ↑ ==========================
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        # ============ ↓ Constrained transport and sources ↓ ==========
        # -------------------------------------------------------------

        if fuse_update:
            rhs_bx, rhs_by, rhs_bz = _ct_rhs_pallas_x_precomputed(
                current_q,
                flux_x_modified[0],
                flux_x_modified[1],
                Bx_flux_y,
                Bz_flux_y,
                Bx_flux_z,
                By_flux_z,
                dtdx,
                dtdy,
                dtdz,
                config,
                registered_variables,
            )
        else:
            rhs_bx, rhs_by, rhs_bz = _constrained_transport_rhs_from_slices(
                current_q,
                By_flux_x,
                Bz_flux_x,
                Bx_flux_y,
                Bz_flux_y,
                Bx_flux_z,
                By_flux_z,
                dtdx,
                dtdy,
                dtdz,
                config,
                registered_variables,
            )
        del flux_x_modified

        # Explicit ohmic resistivity: a further curl of an edge EMF on the
        # interface fields, so div(B) = 0 is kept exactly (see _resistivity).
        if config.resistivity:
            resistive_rhs_bx, resistive_rhs_by, resistive_rhs_bz = fd_ohmic_interface_rhs(
                bx,
                by,
                bz,
                params.resistivity,
                dt,
                grid_spacing,
                config,
            )
            rhs_bx = rhs_bx + resistive_rhs_bx
            rhs_by = rhs_by + resistive_rhs_by
            rhs_bz = rhs_bz + resistive_rhs_bz

        if config.dimensionality == 1:
            density_fluxes = (density_flux_x,)
        elif config.dimensionality == 2:
            density_fluxes = (density_flux_x, density_flux_y)
        else:
            density_fluxes = (density_flux_x, density_flux_y, density_flux_z)

        # Physics source terms.  On the Pallas-fused path the divergence
        # has already been folded into ``dq``; we add ``dt * S`` on top.
        # On the native fallback we still have a standalone ``rhs_q_for_phys``
        # and fold the full LSRK4 update at the end.
        if use_pallas_div:
            sources = _time_integrator_sources(
                current_q,
                density_fluxes,
                dq[registered_variables.density_index],
                dt,
                gamma,
                config,
                params,
                helper_data,
                registered_variables,
            )
            if sources is not None:
                dq = dq + sources
        else:
            rhs_q_for_phys += _time_integrator_sources(
                current_q,
                density_fluxes,
                rhs_q_for_phys[registered_variables.density_index],
                dt,
                gamma,
                config,
                params,
                helper_data,
                registered_variables,
            )
            dq = a_coef * dq + rhs_q_for_phys

        # -------------------------------------------------------------
        # ============ ↑ Constrained transport and sources ↑ ==========
        # -------------------------------------------------------------

        return dq, rhs_bx, rhs_by, rhs_bz

    def compute_lqs(current_q, bx, by, bz, dq, a_coef):
        """The stage increments of ``stage_rhs``, on the fast path if it applies."""
        primals = (current_q, bx, by, bz, dq, a_coef, dt, gamma, grid_spacing, params, helper_data)
        if not use_kept_x_flux_halo:
            return stage_rhs(False, *primals)
        # The kept-halo kernels call Pallas outside the differentiable
        # wrappers, so derivatives are taken through the standard path, which
        # computes the same stage.
        return diffable_pallas_call_n(
            primals,
            pallas_branch=partial(stage_rhs, True),
            native_branch=partial(stage_rhs, False),
        )

    def pre_stage(u):
        q, bx, by, bz = u
        if config.boundary_handling == GHOST_CELLS:
            q = _boundary_handler(
                q, config, registered_variables, params, CONSERVATIVE_GAS_STATE,
            )
            b_curr = _boundary_handler(
                jnp.stack([bx, by, bz], axis=0),
                config,
                registered_variables,
                params,
                MAGNETIC_FIELD_ONLY,
            )
            bx, by, bz = b_curr[0], b_curr[1], b_curr[2]
        return (q, bx, by, bz)

    def lsrk_increment(u, du, a_coef, dt_step):
        # ``compute_lqs`` returns the new ``dq`` already in LSRK4 form
        # (``a_coef * dq_old + dt * L_q``), folding the accumulate into the
        # divergence kernel when Pallas is available.  The interface-B deltas
        # use the explicit ``a_coef * db + dt * L_b`` low-storage update.
        # ``lsrk4`` hands every stage the full step ``dt_step == dt``, which
        # ``compute_lqs`` already closes over.
        q, bx, by, bz = u
        dq, dbx, dby, dbz = du
        dq, rhs_bx, rhs_by, rhs_bz = compute_lqs(q, bx, by, bz, dq, a_coef)
        dbx = a_coef * dbx + rhs_bx
        dby = a_coef * dby + rhs_by
        dbz = a_coef * dbz + rhs_bz
        return (dq, dbx, dby, dbz)

    lsrk_increment = _stage_remat(lsrk_increment, config)

    def finalize(u):
        q, bx, by, bz = u
        q = update_cell_center_fields(
            q, bx, by, bz, config, registered_variables,
        )
        return (q, bx, by, bz)

    return lsrk4(
        (conserved_state, bx_interface, by_interface, bz_interface),
        dt,
        pre_stage=_stage_remat(pre_stage, config),
        finalize=_stage_remat(finalize, config),
        lsrk_increment=lsrk_increment,
    )
