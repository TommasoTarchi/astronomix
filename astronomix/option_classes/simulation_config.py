"""
Static simulation configuration.

Defines :class:`SimulationConfig` — the bundle of options that, unlike the
simulation parameters, necessitate recompilation when changed — together with
the integer-coded enumerations they reference (backends, solver/boundary/Riemann
modes, positivity modes, ...), the small geometry vector helpers, the sub-configs
for gravity and positivity, and the ``finalize_config`` pass that fills in
derived fields and validates the configuration.
"""

# general
import math
import subprocess

# typing
from types import NoneType
from typing import NamedTuple, Optional, Tuple, Union
from jaxtyping import Array, Float

# jax
import jax

# astronomix containers
from astronomix._modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
)
from astronomix._modules._cooling.cooling_options import CoolingConfig
from astronomix._modules._cosmic_rays.cosmic_ray_options import CosmicRayConfig
from astronomix._modules._neural_net_force._neural_net_force_options import (
    NeuralNetForceConfig,
)
from astronomix._modules._stellar_wind.stellar_wind_options import WindConfig
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig

# ===================== constant definition =====================

# backends (very limited support currently)
NATIVE_JAX = 0
PALLAS = 1
#: OPTIMAL_BACKEND is not a backend of its own: it is a request to pick the
#: fastest available one at ``finalize_config`` time.  It resolves to PALLAS
#: when JAX's default backend is a GPU new enough to run the Triton kernels
#: (compute capability >= 8.0, read from ``jax.devices()``) and falls back to
#: NATIVE_JAX everywhere else (older GPUs, CPU -- including ``JAX_PLATFORMS=cpu``
#: on a GPU node).
OPTIMAL_BACKEND = 2

# positivity-enforcement modes (used by ``PositivityConfig.per_stage_mode`` /
# ``per_step_mode``).  HARD_FLOOR clamps density (and, for ideal
# gas, pressure) pointwise — cheap, non-conservative, matches the *adiabatic*
# HOW-MHD ``prot.f``.  REDISTRIBUTE neighbour-averages density+momentum (and
# energy) over the valid 3x3x3 neighbourhood of sub-threshold cells — much
# gentler at strong shocks than a hard floor (no sharp floored cell), matches
# the *isothermal* HOW-MHD ``prot.f`` (not strictly mass-conserving: like
# ``prot.f`` it copies neighbour values without debiting the donors).
POSITIVITY_NONE = 0
POSITIVITY_HARD_FLOOR = 1
POSITIVITY_REDISTRIBUTE = 2
#: CONSERVATIVE: enforce internal-energy positivity by an antisymmetric
#: face-flux diffusion that pulls internal energy into (near-)negative-pressure
#: cells from their hotter neighbours (exact total-energy conservation), plus a
#: density floor / vacuum-rest for voids and a minimal residual pressure floor
#: as the unconditional guarantee. The smooth, conservative cousin of HARD_FLOOR:
#: it keeps the energy-conserving self-gravity scheme stable on violent collapse
#: without the 100%+ energy injection a bare floor causes.
POSITIVITY_CONSERVATIVE = 3

# solver modes
FINITE_VOLUME = 0
FINITE_DIFFERENCE = 1

# differentiation modes
FORWARDS = 0
BACKWARDS = 1

# Passive-scalar sub-cycling loop (``SimulationConfig.passive_scalar_substep_loop``).
#   SUBSTEPS_AUTO     - DYNAMIC under FORWARDS, MASKED under BACKWARDS (default)
#   SUBSTEPS_DYNAMIC  - a traced trip count (lowers to a while loop): exactly the
#                       flow-derived number of sub-steps runs; forward-mode AD
#                       only (reverse mode cannot differentiate a while loop)
#   SUBSTEPS_MASKED   - a static loop over ``max_passive_scalar_substeps`` in
#                       which sub-steps beyond the flow-derived count are masked
#                       out (``lax.cond``): reverse-mode differentiable, same
#                       primal
SUBSTEPS_AUTO = 0
SUBSTEPS_DYNAMIC = 1
SUBSTEPS_MASKED = 2

# Rematerialisation in reverse mode (``SimulationConfig.ad_remat``).
#   "none"  - store every residual of the step (fastest backward, most memory)
#   "stage" - ``jax.checkpoint`` around every Runge-Kutta-stage RHS (hydro WENO
#             fluxes + blends + divergence + sources, and the passive-scalar
#             advection RHS): the backward keeps only the stage inputs and
#             recomputes each stage's internals (about one extra forward)
#   "axis"  - "stage" plus nested checkpoints: each axis' flux + blend +
#             divergence inside the hydro RHS (non-fused flux path), and in the
#             passive-scalar RHS each axis and each scalar separately, so the
#             backward holds one axis' (one scalar's) internals at a time
AD_REMAT_NONE = "none"
AD_REMAT_STAGE = "stage"
AD_REMAT_AXIS = "axis"
AD_REMAT_MODES = (AD_REMAT_NONE, AD_REMAT_STAGE, AD_REMAT_AXIS)

# limiter types
MINMOD = 0
OSHER = 1
DOUBLE_MINMOD = 2
SUPERBEE = 3
VAN_ALBADA = 4
VAN_ALBADA_PP = 5

# splitting modes
UNSPLIT = 0
SPLIT = 1

# Riemann solvers
HLL = 0
HLLC = 1
HLLC_LM = 2
LAX_FRIEDRICHS = 3
HYBRID_HLLC = 4
AM_HLLC = 5

# time integrators
# currently only for finite volume
RK2_SSP = 0
MUSCL = 1
# currently only for finite difference
RK4_SSP = 2
RK4_LSRK = 3

# boundary conditions
OPEN_BOUNDARY = 0
REFLECTIVE_BOUNDARY = 1
PERIODIC_BOUNDARY = 2
FIXED_BOUNDARY = 3
MHD_JET_BOUNDARY = 4
FIXED_BOUNDARY_OPEN_MOMENTUM = 5

PRIMITIVE_GAS_STATE = 0
CONSERVATIVE_GAS_STATE = 1
VELOCITY_ONLY = 2
MAGNETIC_FIELD_ONLY = 3

# geometry types
CARTESIAN = 0
CYLINDRICAL = 1
SPHERICAL = 2

# axes
VARAXIS = 0
XAXIS = 1
YAXIS = 2
ZAXIS = 3

# boundary handling modes
GHOST_CELLS = 0
PERIODIC_ROLL = 1
# OPEN_SHIFT = 2

# self-gravity coupling schemes (FD):
#   SIMPLE_SOURCE              - rho * v * a energy source (non-conservative)
#   SECOND_ORDER_CONSERVATIVE  - flux-based energy source (2nd-order accurate)
#   FOURTH_ORDER_CONSERVATIVE  - corrected flux-based energy source (4th-order,
#                                the energy-conserving high-order scheme)
SIMPLE_SOURCE = 0
SECOND_ORDER_CONSERVATIVE = 1
FOURTH_ORDER_CONSERVATIVE = 2

# Magnetic part integrators for split MHD
IMPLICIT_MIDPOINT = 0
IMPLICIT_EULER = 1

# Numerical precision
SINGLE_PRECISION = 0
DOUBLE_PRECISION = 1

# Viscosity types
KINEMATIC_VISCOSITY = 0
DYNAMIC_VISCOSITY = 1

# Equation of state
IDEAL_GAS = 0
ISOTHERMAL = 1

# Snapshot storage modes
ON_DEVICE = 0
TO_DISK = 1

# ============================================================

# ===================== type definitions =====================

class StaticIntVector(NamedTuple):
    """A static (compile-time) per-axis integer triple (e.g. cells per axis)."""

    x: int = -1
    y: int = -1
    z: int = -1


class StaticFloatVector(NamedTuple):
    """A static (compile-time) per-axis float triple (e.g. box size per axis)."""

    x: float = -1.0
    y: float = -1.0
    z: float = -1.0

    def __truediv__(self, other: StaticIntVector) -> "StaticFloatVector":
        """Divide component-wise by a :class:`StaticIntVector` (e.g. box / cells)."""
        if not isinstance(other, StaticIntVector):
            return NotImplemented
        return StaticFloatVector(
            x=self.x / other.x,
            y=self.y / other.y,
            z=self.z / other.z,
        )

STATE_TYPE = Union[
    Float[Array, "num_vars num_cells_x"],
    Float[Array, "num_vars num_cells_x num_cells_y"],
    Float[Array, "num_vars num_cells_x num_cells_y num_cells_z"],
]

STATE_TYPE_ALTERED = Union[
    Float[Array, "num_vars num_cells_a"],
    Float[Array, "num_vars num_cells_a num_cells_b"],
    Float[Array, "num_vars num_cells_a num_cells_b num_cells_c"],
]

FIELD_TYPE = Union[
    Float[Array, "num_cells_x"],
    Float[Array, "num_cells_x num_cells_y"],
    Float[Array, "num_cells_x num_cells_y num_cells_z"],
]

class SnapshotSettings(NamedTuple):
    """Settings for the snapshot output of the simulation."""

    #: Whether to record the full primitive state at every checkpoint.
    #: This is the single biggest snapshot allocation
    #: (``num_snapshots × num_vars × num_cells^d``); it is **opt-in**.
    #: Set to ``True`` if you actually need the per-snapshot states; for
    #: the common case of only wanting a final state plus integrated
    #: diagnostics (energies, total mass, runtime, num_iterations), the
    #: default ``False`` skips the per-snapshot state allocation entirely.
    return_states: bool = False

    #: Whether to return the final state of the simulation.
    return_final_state: bool = False

    #: Whether to return the total mass at the times the snapshots were taken.
    return_total_mass: bool = False

    #: Whether to return the total energy at the times the snapshots were taken.
    return_total_energy: bool = False

    #: Whether to return internal energy
    return_internal_energy: bool = False

    #: Whether to return kinetic energy
    return_kinetic_energy: bool = False

    #: Whether to return gravitational energy
    return_gravitational_energy: bool = False

    #: Whether to return radial momentum
    return_radial_momentum: bool = False

    #: Whether to return the kinetic energy spectrum
    return_kinetic_energy_spectrum: bool = False

    #: Whether to return the magnetic energy spectrum
    return_magnetic_energy_spectrum: bool = False

    #: Whether to return the helicity spectrum
    return_helicity_spectrum: bool = False

    #: Whether to return the magnetic field divergence
    #: NOTE: currently only implemented for finite difference MHD
    return_magnetic_divergence: bool = False

    #: Whether to return the temperature PDF (dV/dlogT)
    return_temperature_pdf: bool = False
    num_temperature_bins: int = 100
    temperature_pdf_min: float = 1e-10
    temperature_pdf_max: float = 1e10


class BoundarySettings1D(NamedTuple):
    """The boundary-condition type at the left and right end of a single axis."""

    left_boundary: int = OPEN_BOUNDARY
    right_boundary: int = OPEN_BOUNDARY


class BoundarySettings(NamedTuple):
    """Per-axis boundary settings for the simulation."""

    x: BoundarySettings1D = BoundarySettings1D()
    y: BoundarySettings1D = BoundarySettings1D()
    z: BoundarySettings1D = BoundarySettings1D()


class GravityConfig(NamedTuple):
    """Self-gravity and external-potential configuration."""

    #: Self-gravity switch (currently only for periodic / manual-open boundaries).
    self_gravity: bool = False

    #: Coupling of the self-gravity source to the hydrodynamics. One of
    #: ``SIMPLE_SOURCE`` / ``SECOND_ORDER_CONSERVATIVE`` /
    #: ``FOURTH_ORDER_CONSERVATIVE``.
    self_gravity_version: int = FOURTH_ORDER_CONSERVATIVE

    #: Enable an external, static gravitational potential provided via
    #: ``params.gravitational_potential``. It is added to the self-gravity
    #: potential (if any) in ``_compute_total_potential``.
    external_potential: bool = False

    #: Manual open boundary conditions in the Poisson solver.
    poisson_manual_open_boundaries: bool = False

    #: Non-conservative backstop for the energy-conserving coupling (finite
    #: difference, SECOND/FOURTH_ORDER_CONSERVATIVE): the part of the energy
    #: source that is not the kinetic work of the momentum source is scaled
    #: down wherever it would drain the internal energy faster than half of it
    #: per wave-crossing time (a dt-independent rate budget). Use it together
    #: with ``work_flux_correction``: that conservative correction removes the
    #: dominant failure (half the climb of mass entering a cold or tenuous cell
    #: charged to the receiver), and this backstop then only catches the work
    #: no conservative split can pay for -- mass lifted against gravity by
    #: numerical diffusion in cold gas -- so the energy it rejects is confined
    #: to those cells.
    limit_internal_energy_work: bool = False

    #: Flux-corrected gravitational work (finite difference,
    #: SECOND/FOURTH_ORDER_CONSERVATIVE). Every conservative energy coupling
    #: is a choice of the potential-energy flux q at each face,
    #: S_E,i = -(1/dx) sum [(q - F phi_i)_{i+1/2} - (q - F phi_i)_{i-1/2}],
    #: and conserves total energy for ANY q. The scheme's high-order q charges
    #: half the climb of mass crossing a face to each side, so a cold or
    #: tenuous receiver can be driven to negative pressure; the low-order
    #: q = F phi_downwind charges the whole climb to the donor (the cell the
    #: mass leaves). This option blends them face by face,
    #: q = q_low + psi (q_high - q_low), with the largest psi in [0, 1] that
    #: keeps every cell's internal-energy loss rate within half its internal
    #: energy per wave-crossing time (Zalesak limiting with RATE budgets, so
    #: psi does not depend on dt). Exactly conservative; high order wherever
    #: psi = 1.
    work_flux_correction: bool = False

    #: Master gravity switch. Set automatically in ``finalize_config`` to
    #: ``self_gravity or external_potential``; gates the gravity source-term
    #: machinery so an external potential works without self-gravity. Not set
    #: by the user directly.
    gravity: bool = False


class PositivityConfig(NamedTuple):
    """
    Density/pressure positivity-enforcement configuration.
    """

    #: Casual on/off switch for the per-stage / per-step STATE floors. Default
    #: False (no flooring). When True, finalize_config sets per_stage_mode and
    #: per_step_mode to HARD_FLOOR unless explicitly overridden. Does NOT affect
    #: the read-only ``clamp_in_estimates`` (always respected).
    default_positivity_protection: bool = False

    #: Positivity enforcement applied inside every SSPRK/LSRK stage (on the
    #: conserved state — the CFL lever for strong shocks). One of
    #: ``POSITIVITY_{NONE,HARD_FLOOR,REDISTRIBUTE,CONSERVATIVE}``. Default NONE;
    #: set to HARD_FLOOR by finalize when ``default_positivity_protection``.
    per_stage_mode: int = POSITIVITY_NONE

    #: Positivity enforcement applied once per step before the evolve (on the
    #: primitive state). With turbulent forcing + ``vacuum_protection`` the
    #: conservative ``prot`` redistribution already runs once per step, so a
    #: per-step REDISTRIBUTE here is redundant and is auto-skipped.
    per_step_mode: int = POSITIVITY_NONE

    #: Upgrade the per-step HARD_FLOOR pressure clamp to the density-scaled
    #: temperature floor ``p >= max(minimum_pressure,
    #: rho * params.minimum_specific_pressure)`` (Athena-style tfloor). Meant
    #: for RADIATIVE runs: a radiatively cooled shock layer compresses to the
    #: isothermal jump, and without isothermal pressure support (p ∝ rho) the
    #: constant floor leaves it effectively pressureless and it ram-crushes
    #: without bound. Applied ONCE per step (the per-STAGE variant pumps
    #: energy and destabilized adiabatic runs, 2026-07-25); with cooling
    #: active the floor's energy input is radiated away (the isothermal
    #: balance). No-op when ``params.minimum_specific_pressure == 0``.
    per_step_specific_floor: bool = False

    #: Additionally apply the density-scaled temperature floor inside EVERY
    #: RK stage's HARD_FLOOR positivity pass (p >= rho * msp on the conserved
    #: state). A radiative crush can complete within the stages between
    #: per-step floors; this closes that window. CAUTION: in ADIABATIC runs
    #: the per-stage injection was a proven destabilizer (2026-07-25) — use
    #: only with real cooling, which radiates the injected energy away.
    per_stage_specific_floor: bool = False

    #: Read-only density/pressure clamp in the flux / eigenvalue / timestep
    #: estimates (NaN-safety; does NOT modify the evolved state). This is the
    #: role the old ``enforce_positivity`` bool played in those estimators.
    #: DECOUPLED from ``default_positivity_protection`` and ON by default --
    #: cheap insurance that never touches the conserved solution.
    clamp_in_estimates: bool = True

    #: Blend the WENO flux toward HLLC instead of first-order Lax-Friedrichs
    #: in the positivity/FCT limiter (ideal-gas hydro only; MHD and isothermal
    #: keep LLF). Both are positivity preserving under the CFL condition, but
    #: LLF smears the CONTACT wave at first order, so blending toward it
    #: dissolves cold dense condensations — a two-phase medium imported from
    #: AthenaK was fully evaporated in 10 Myr through that path, while the same
    #: run with the limiter disabled kept (and grew) its cold phase but went
    #: numerically unstable. HLLC resolves the contact exactly, so positivity
    #: can be enforced without erasing the structure.
    blend_fallback_hllc: bool = False

    #: Vacuum-rest velocity recovery: zero the momentum in below-floor (vacuum)
    #: cells so the recovered velocity is 0 rather than ``momentum/rho_floored``
    #: (which spikes and drives high-Mach blow-up); lets ``minimum_density`` be
    #: lowered by orders of magnitude without instability.
    vacuum_rest: bool = False

    #: NaN/inf backstop: reset non-finite conserved entries to zero before the
    #: density/pressure floors so they become a valid floored state.
    nan_safe: bool = False

    #: POSITIVITY_CONSERVATIVE-mode parameters (conservative internal-energy
    #: redistribution): per-axis diffusion coefficient (stability needs
    #: < 1/(2*dim)), number of Jacobi passes, and the activation margin in units
    #: of the internal-energy floor (keep ~1 -- genuine near-violations only).
    cons_coeff: float = 0.15
    cons_passes: int = 16
    cons_activate: float = 1.0

    #: Deep-void first-order flux blending (FOFC-style): blend the WENO interface
    #: flux toward LLF in cells near the density floor; the weight ramps from 1
    #: at the floor to 0 at ``deepvoid_blend_factor * minimum_density``.
    deepvoid_blend: bool = False
    deepvoid_blend_factor: float = 8.0

    #: Positivity-preserving (Hu-Adams-Shu / Zalesak FCT) flux limiter: blend the
    #: WENO flux toward LLF by the largest weight keeping the LF-updated density
    #: AND pressure above their floors. Shares the unified flux-blending
    #: infrastructure with ``deepvoid_blend`` (different activation path; both may
    #: be on, the stronger blend wins). Forces the non-fused WENO+divergence path.
    preserving_flux: bool = False

    #: Cold-crush first-order flux blending (the FD counterpart of Athena's
    #: FOFC for radiatively cooled gas): blend the WENO interface flux toward
    #: LLF at interfaces with a COLD side under COMPRESSION. The weight is a
    #: temperature ramp on the COLDER adjacent cell's recovered ``p/rho``
    #: (1 at the effective temperature floor
    #: ``params.minimum_specific_pressure``, 0 at ``coldcrush_blend_factor``
    #: times it) times a compressive-velocity gate (so the freely-expanding
    #: cold ejecta core and the static ambient never activate). Catches both
    #: cold-cold isothermal collapse and the crushing of a cold dense clump
    #: by hot surroundings; the trade is locally first-order shock fronts
    #: into cold gas (classic FOFC behavior). Radiatively cooled,
    #: ram-pressure-crushed cells otherwise collapse without bound once the
    #: grid resolves the cooling layer (the 512^3 blast/shell and jet-cone
    #: blow-ups): the local first-order diffusion saturates the collapse the
    #: way coarse-grid numerical diffusion does at lower resolution. Inert
    #: unless ``params.minimum_specific_pressure > 0``.
    coldcrush_blend: bool = False
    coldcrush_blend_factor: float = 8.0


class BackendConfig(NamedTuple):
    """Compute-backend configuration: which backend runs the kernels and the
    Pallas/Triton knobs that shape them.

    Grouped as its own sub-config (like ``PositivityConfig`` / ``GravityConfig``)
    so backend concerns stay together.  Construct nested, e.g.
    ``SimulationConfig(backend_config=BackendConfig(backend=NATIVE_JAX))``.
    """

    #: Backend. Defaults to OPTIMAL_BACKEND, which ``finalize_config`` resolves
    #: to PALLAS on compute-capability >= 8.0 GPUs and NATIVE_JAX otherwise.
    backend: int = OPTIMAL_BACKEND
    #: Pallas kernel block shape ``(bx, by, bz)``; ``None`` picks the tuned
    #: per-dimensionality default (see ``_default_pallas_block_shape``),
    #: clamped to the grid extents.  With the default ``pallas_num_warps=4``
    #: keep blocks at 128 cells (one element per thread) — larger blocks
    #: register-spill in the f64 WENO kernels, smaller ones idle threads.
    pallas_block_shape: Optional[Tuple[int, int, int]] = None
    pallas_use_triton: bool = True
    pallas_interpret: bool = False
    pallas_num_warps: int = 4
    #: Toggle for the Pallas constrained-transport helpers
    #: (``update_cell_center_fields``, ``constrained_transport_rhs``).
    #: Disabled by default: the staged Pallas-CT pipeline gives a clear
    #: memory win at small grids (~65% temp at N=16 on alfven_wave3D)
    #: but only marginal savings at production scale (~2% temp at N=64)
    #: while adding ~25s of one-time compile cost.  Flip to True if the
    #: small-N memory profile matters; the rest of the Pallas backend
    #: stays on regardless.
    pallas_ct: bool = False
    #: Replace the IEEE ``sqrt`` in the MHD WENO kernel with the refined
    #: approximate ``rsqrt`` path (``x * jax.lax.rsqrt(x)`` -> ``rsqrt.approx.f64``,
    #: still ~1 ULP).  On A100 this cut the dp WENO kernel ~1.6x and the full
    #: dp step ~1.77x (spill loads halved) with Alfvén L1 convergence bit-identical.
    #: NOTE: the *forward* kernel is switched but the hand-written Pallas
    #: *adjoints* still use IEEE ``sqrt``; with this on, reverse-mode AD is
    #: therefore ~1 ULP inconsistent with the forward (``finalize_config`` warns).
    #: There is no equivalent fast f64 path for division, so only ``sqrt`` changes.
    use_approximate_rsqrt: bool = False


class SimulationConfig(NamedTuple):
    """
    Configuration object for the simulation.
    The simulation configuration are parameters defining
    the simulation where changes necessitate recompilation.
    """

    # Static simulation parameters

    #: Compute-backend configuration (see :class:`BackendConfig`): backend
    #: choice + Pallas/Triton kernel knobs.  Accessed as
    #: ``config.backend_config.backend`` / ``.pallas_block_shape`` / etc.
    backend_config: BackendConfig = BackendConfig()

    #: Basic solver mode, either finite volume or finite difference.
    #: Defaults to the finite-difference HOW-MHD scheme (Jeongbhin Seo,
    #: Dongsu Ryu, 2023), which is the recommended solver.
    solver_mode: int = FINITE_DIFFERENCE

    #: Precision mode.
    numerical_precision: int = SINGLE_PRECISION

    #: Debug runtime errors, throws exceptions
    #: on e.g. negative pressure or density.
    #: Significantly reduces performance.
    runtime_debugging: bool = False

    #: Donate the state arrays to the time integration function
    #: to reduce memory allocations. If activated, the
    #: initial state arrays will be invalid after
    #: the simulation.
    donate_state: bool = False

    #: Memory analysis of the main time integration
    #: function
    memory_analysis: bool = False

    #: Build the simulation helper data on the host (CPU) and
    #: only move the fields that are actually needed by the
    #: enabled subsystems onto the accelerator. Useful in
    #: production runs where a large meshgrid like
    #: ``geometric_centers`` is not required on device and the
    #: per-field memory footprint matters.
    host_helper_data: bool = False

    #: Print the elapsed time of the simulation
    print_elapsed_time: bool = False

    #: Activate progress bar
    progress_bar: bool = False

    #: The number of dimensions of the simulation.
    dimensionality: int = 1

    #: Use a struct for the state.
    state_struct: bool = False

    #: The geometry of the simulation.
    geometry: int = CARTESIAN

    #: The random seed for any stochastic processes
    #: in the simulation, e.g. turbulent forcing.
    random_seed: int = 42

    #: The equation of state for the simulation.
    #: NOTE: CURRENTLY ONLY IMPLEMENTED FOR 
    #: FINITE DIFFERENCE MODE.
    equation_of_state: int = IDEAL_GAS

    #: Magnetohydrodynamics switch.
    mhd: bool = False

    #: Integrator used for the magnetic part in the FV MHD scheme.
    fv_magnetic_integrator: int = IMPLICIT_MIDPOINT

    #: Density/pressure positivity-enforcement configuration (see PositivityConfig).
    positivity_config: PositivityConfig = PositivityConfig()

    #: Dual-energy formalism (Bryan et al. 1995 switch) for adiabatic FD
    #: hydro and MHD. When True a separately-advected internal-energy density
    #: ``g`` is carried through the time loop and used in the WENO pressure
    #: recovery wherever ``e_E/E < dual_energy_eta``, so high-Mach / low-beta
    #: float32 cancellation of ``e_int = E - KE - ME`` does not corrupt the
    #: pressure.
    dual_energy: bool = False

    #: Switch threshold for the dual-energy formalism: the fraction of total
    #: energy below which the total-energy internal energy is deemed
    #: cancellation-unreliable and the advected ``g`` is used instead.
    dual_energy_eta: float = 1e-3

    #: Number of user-defined passive scalars: per-parcel labels advected with
    #: the flow (``dC/dt + v.grad C = 0``) that do not act back on it —
    #: composition mass fractions, an ejecta/circumstellar discriminator, and so
    #: on. Finite-difference solver only; they are carried as the last variables
    #: of the state and advected operator-split (WENO5 + SSP-RK3) in
    #: ``_passive_scalars.py``.
    num_passive_scalars: int = 0

    #: Physical bounds for the user's passive scalars, one ``(lo, hi)`` pair per
    #: scalar (``None`` for an unbounded one). Declaring them is strongly
    #: recommended for anything that is a mass fraction: the recovered label is
    #: a ratio ``s / rho~``, and in the near-vacuum interior of a blast wave the
    #: denominator can collapse and the ratio run away (measured: an ejecta
    #: fraction reaching -87 where radiative cooling and a fast ejecta piston
    #: compress the same cells).
    #:
    #: Note this is NOT the same as clipping a scalar to its own current range,
    #: which is destructive: that clips smooth extrema every step and cost a
    #: factor of three in convergence order when tried. A *physical* bound never
    #: activates on smooth data that respects it, so it is free.
    passive_scalar_bounds: tuple = ()

    #: CFL number for the passive-scalar sub-steps. The scalar advection is
    #: operator-split, so it does not inherit the hydro timestep's safety: the
    #: hydro CFL is set by ``|u| + c``, while what constrains this is ``|u|`` and
    #: — the binding one — positivity of the companion density under
    #: ``h |div v| <= 0.5``. The number of sub-steps is derived from the flow at
    #: every step, so a benign step takes exactly one and costs nothing.
    passive_scalar_cfl: float = 0.4

    #: Cap on the derived sub-step count, so a single pathological cell cannot
    #: stall a run. Hitting the cap means the scalars are being integrated
    #: outside their stability limit in some cells; the physical bounds in
    #: ``passive_scalar_bounds`` are the backstop for that.
    max_passive_scalar_substeps: int = 8

    #: How the sub-cycling loop is built (``SUBSTEPS_AUTO`` / ``_DYNAMIC`` /
    #: ``_MASKED``, see the constants). The flow-derived count is a traced
    #: integer, and a loop with a traced trip count is a while loop, which
    #: reverse-mode AD cannot differentiate. The MASKED loop runs to the static
    #: cap ``max_passive_scalar_substeps`` and skips (``lax.cond``) the sub-steps
    #: beyond the flow-derived count, so its primal is the same computation. The
    #: default AUTO keeps the dynamic loop under ``differentiation_mode ==
    #: FORWARDS`` (forward results unchanged, bit for bit) and switches to the
    #: masked one under BACKWARDS. In the masked loop every sub-step (the first
    #: included) is one iteration of a scan whose body is ``jax.checkpoint``-ed
    #: (its RHS too, whatever ``ad_remat`` says), so the backward stores one
    #: scalar-stack copy per iteration and recomputes the rest: reverse-mode
    #: memory grows only mildly with the cap, compute by one extra advection
    #: evaluation. A cap of 1 is a straight-line single sub-step (no loop).
    #: Lower ``max_passive_scalar_substeps`` in reverse-mode runs whose flow
    #: never needs more than one or two sub-steps.
    passive_scalar_substep_loop: int = SUBSTEPS_AUTO

    #: Track when each parcel was shocked, adding three further library-managed
    #: scalars (``entropy_initial``, ``time_since_shock``, ``density_time``)
    #: after the user's. ``density_time`` is the ionization age up to a unit
    #: conversion, and ``time_since_shock`` drives electron/ion temperature
    #: relaxation — the Dwarkadas, Dewey & Bauer (2010) proxy that makes
    #: non-equilibrium-ionization spectral synthesis possible without an
    #: ionization network.
    track_shock_history: bool = False

    #: Entropy rise (in nats, above the parcel's own ``t = 0`` value) that counts
    #: as having been shocked. Because the Rankine-Hugoniot jump fixes the
    #: entropy rise as a function of Mach number alone, this threshold is
    #: equivalent to a minimum shock strength; for ``gamma = 5/3``:
    #:
    #: ===========  ==========
    #: Mach number  rise [nats]
    #: ===========  ==========
    #: 2            0.18
    #: 3            0.57
    #: **3.3**      **0.69**
    #: 5            1.31
    #: 10           2.57
    #: 100          7.12
    #: ===========  ==========
    #:
    #: The default ``log(2)`` therefore flags everything above Mach ~3.3 —
    #: comfortably below a supernova remnant's Mach ~100 forward and reverse
    #: shocks, while ignoring the weak compressions and sound waves that carry
    #: no ionization. Lower it if weak shocks matter for the problem at hand.
    shock_entropy_jump: float = 0.6931471805599453

    #: Give the shock-history latch a derivative (straight-through estimator).
    #: The primal is UNCHANGED: a parcel is still flagged by the boolean test
    #: ``entropy rise > shock_entropy_jump AND div v < 0``. With the switch off
    #: that boolean has zero derivative, so d(shocked_fraction, time_since_shock,
    #: density_time)/d(state) only transports the existing history, and an
    #: adjoint cannot see that moving a shock changes which parcels are freshly
    #: shocked. With it on, the tangent (and cotangent) of the latch is taken
    #: through the smooth surrogate
    #:
    #:     soft = sigmoid((ds - shock_entropy_jump) / ad_shock_latch_entropy_width)
    #:          * sigmoid(-div_v / (ad_shock_latch_compression_width * c_s))
    #:
    #: (the same two criteria; ``div_v`` is the undivided central difference
    #: the latch uses and ``c_s`` the local sound speed, held constant), merged
    #: with the carried fraction as a probabilistic OR:
    #: ``d sf = (1 - latch) d sf_old + (1 - sf_old) d soft``. Away from the
    #: threshold soft saturates and the derivative reduces to transport -- but
    #: only if the entropy sigmoid is narrow enough: quiescent, never-shocked
    #: gas (entropy rise 0, div_v = 0) sits at the steepest point of the
    #: compression sigmoid, so its leak is set by sigmoid(-jump / width) alone,
    #: and it ACCUMULATES step after step in the carried fraction. Measured
    #: (review 2026-09-25, 16^3 blast, 1 % pressure tangent): width 0.25 leaks
    #: 22 % of the at-threshold response per step into every ambient cell,
    #: width 0.1 (the default) 0.4 %. This is a surrogate gradient, not the
    #: derivative of the primal (which is zero almost everywhere); check it
    #: with a finite-difference Taylor test at the amplitudes the optimiser
    #: uses.
    ad_smooth_shock_latch: bool = False

    #: Width in nats of the entropy-rise sigmoid of the smooth latch. Keep it
    #: well below ``shock_entropy_jump`` (see the leak above).
    ad_shock_latch_entropy_width: float = 0.1

    #: Width of the compression sigmoid of the smooth latch, in units of the
    #: local sound speed (``div_v`` is an undivided difference, i.e. a velocity).
    ad_shock_latch_compression_width: float = 0.05

    #: Self-gravity / external-potential configuration (see GravityConfig).
    gravity_config: GravityConfig = GravityConfig()

    #: Explicit diffusion term 
    #: (currently only for finite difference mode)
    diffusion: bool = False

    #: Viscosity type - either kinematic or dynamic viscosity.
    viscosity_type: int = DYNAMIC_VISCOSITY

    #: Explicit ohmic resistivity ``params.resistivity`` in the induction
    #: equation, applied to the interface fields as the curl of an edge EMF
    #: (finite-difference CT MHD, isothermal EOS only: no ohmic heating term).
    resistivity: bool = False

    #: Explicit thermal conduction term div(kappa grad T) in the energy
    #: equation (constant conductivity params.thermal_conductivity,
    #: explicit integration). Currently only for finite difference mode.
    thermal_conduction: bool = False

    #: Interpret ``params.thermal_conductivity`` as the Athena-style
    #: DIFFUSIVITY ``alpha`` instead of a constant conductivity, i.e. use
    #: ``kappa = rho * alpha`` so the heat flux is ``-rho alpha grad T``
    #: (conservative face-flux form). This keeps the TEMPERATURE diffusivity
    #: ``chi = (gamma - 1) alpha`` density-INDEPENDENT — the convention used
    #: for thermal-instability / Field-length studies (AthenaK
    #: ``<hydro> alpha_iso``), where the constant-kappa form would instead
    #: suppress conduction exactly inside the cold dense clumps.
    conduction_density_weighted: bool = False

    #: Formal order of the conduction discretisation: 2 (legacy) or 4. In a
    #: FINITE-DIFFERENCE scheme the state IS the pointwise value, so evaluating
    #: ``T = p/rho`` (and ``kappa = rho*alpha``) pointwise is already exact --
    #: the order is set purely by the derivative stencils. Order 4 uses the
    #: 4th-order central first derivative for the pointwise heat flux and the
    #: 4th-order conservative face interpolation
    #: ``(-F_{i-1} + 7F_i + 7F_{i+1} - F_{i+2})/12`` for its divergence (the
    #: same linear flux the WENO kernel uses), so it is consistent with the
    #: 5th-order hydro rather than throttling it to 2nd order.
    conduction_order: int = 2

    #: The size of the simulation box.
    box_size: Union[float, StaticFloatVector] = 1.0

    #: The number of cells in the simulation.
    num_cells: Union[int, StaticIntVector] = 400

    #: The reconstruction order is the number of
    #: cells on each side of the cell of interest
    #: used to calculate the gradients for the
    #: reconstruction at the interfaces.
    reconstruction_order: int = 1

    #: The limiter for the reconstruction.
    #: Only affects finite volume mode.
    limiter: int = MINMOD

    #: The Riemann solver used
    #: Only for finite volume mode.
    riemann_solver: int = HLL

    #: Dimensional splitting / unsplit mode.
    #: Note that the UNSPLIT scheme currently
    #: interferes with energy conservation in settings
    #: with self-gravity.
    split: int = UNSPLIT

    #: Time integration method.
    time_integrator: int = RK2_SSP

    # Explanation of the ghost cells
    #                                |---------|
    #                           |---------|
    # stencil              |---------|
    # cells            || 1g | 2g | 3c | 4g | 5g ||
    # reconstructions        |L  R|L  R|L  R|    |
    # fluxes                     -->  -->
    # update                      | 3c'|
    # --> all others are ghost cells

    #: The number of ghost cells.
    num_ghost_cells: int = reconstruction_order + 1

    #: Grid spacing.
    grid_spacing: float = box_size / num_cells

    #: Explicit boundary handling mode.
    boundary_handling: int = GHOST_CELLS

    #: Boundary settings for the simulation.
    boundary_settings: Union[NoneType, BoundarySettings1D, BoundarySettings] = None

    #: Enables a fixed timestep for the simulation
    #: based on the specified number of timesteps.
    fixed_timestep: bool = False

    #: Exactly reach the end time. In adaptive timestepping,
    #: one might otherwise overshoot.
    exact_end_time: bool = True

    #: Adds the sources with the current timestep to
    #: a hypothetical state to estimate the actual timestep.
    #: Useful for time-dependent sources, but additional
    #: computational overhead.
    source_term_aware_timestep: bool = False

    #: The number of timesteps for the fixed timestep mode.
    num_timesteps: int = 1000

    #: Use a maximum timestep in adaptive timestep mode.
    use_max_adaptive_timestep: bool = True

    #: The differentiation mode one whats to use
    #: the solver in (forwards or backwards).
    differentiation_mode: int = FORWARDS

    #: The number of checkpoints used in the setup
    #: with backwards differetiability and adaptive
    #: time stepping.
    num_checkpoints: int = 100

    #: Rematerialisation for reverse-mode AD: ``"none"`` (default), ``"stage"``
    #: or ``"axis"`` (see ``AD_REMAT_*``). Changes only what the backward pass
    #: stores versus recomputes; the primal is the same computation, and under
    #: forward-only evaluation or forward-mode AD ``jax.checkpoint`` is inlined
    #: (no cost). Measured for the casa_xfit configuration (FD/WENO, dual
    #: energy, 5 scalars + shock history, max_passive_scalar_substeps = 8),
    #: temp memory of one BACKWARDS gradient in units of the state:
    #:
    #: ==========================================  ======  =======  ======
    #: setting                                     none    stage    axis
    #: ==========================================  ======  =======  ======
    #: XLA:CPU, f32 32^3, 1 fixed step (no FCT)    290x    159x     77x
    #: XLA:CPU, f32 32^3, equinox K = 4 (no FCT)   295x    173x     87x
    #: A100, f64 64^3, equinox K = 8, FCT, Pallas  --      92x      62x
    #: ==========================================  ======  =======  ======
    #:
    #: Each costs roughly one more forward evaluation of the rematerialised
    #: pieces in the backward pass ("axis": two).
    ad_remat: str = AD_REMAT_NONE

    #: ``ad_remat == "axis"`` only: split each axis' flux + blend + divergence
    #: into this many slabs along a PERPENDICULAR axis (z for the x / y fluxes,
    #: y for the z flux; never the first spatial axis, the one a multi-GPU run
    #: splits), a ``lax.map`` with one ``jax.checkpoint`` per slab. The
    #: backward then holds one slab's WENO / blend internals at a time. Exact
    #: (every piece is a stencil along the flux axis only); 1 = off; a count
    #: that does not divide the slab axis falls back to 1.
    ad_remat_chunks: int = 1

    #: Reverse-mode memory of the passive-scalar block (``ad_remat != "none"``
    #: only; default False = unchanged). The scalar advection's backward was the
    #: peak of a Cas A 4D-Var gradient (xprof memory viewer, 128^3: 45 % of the
    #: live set): the masked sub-step scan's stack of ``max_passive_scalar_substeps``
    #: scalar-stack copies, the tie-preserving clips' mask residuals of the ratio
    #: recovery and of the shock history. With True: the sub-steps run in an
    #: equinox checkpointed while loop over the flow-derived count (two scalar-
    #: stack checkpoints instead of the cap's), and the ratio recovery / bounds
    #: and the shock-history update are ``jax.checkpoint``-ed (their masks are
    #: recomputed, not stored). Same arithmetic in the primal; derivatives equal
    #: up to rounding (a different XLA program).
    ad_scalar_lean: bool = False

    #: Return intermediate snapshots of the time evolution
    #: instead of only the final fluid state.
    return_snapshots: bool = False

    #: Snapshot settings
    snapshot_settings: SnapshotSettings = SnapshotSettings()

    #: Where the snapshots are stored. ``ON_DEVICE`` (default) keeps the
    #: snapshot diagnostics in preallocated device buffers and returns them
    #: at the end (the classic behaviour). ``TO_DISK`` instead streams each
    #: snapshot to disk via Orbax: the run is split into segments between the
    #: snapshot times, and the loop carry (primitive state, PRNG key, OU
    #: forcing field) plus the time is written to ``snapshot_storage_path``
    #: after each segment. Each device writes its own shard, so this scales
    #: to multiple devices / nodes. TO_DISK is forward-mode only.
    snapshot_storage_mode: int = ON_DEVICE

    #: Directory the Orbax checkpoints are written to / read from when
    #: ``snapshot_storage_mode == TO_DISK``. Required in that mode.
    snapshot_storage_path: Union[str, NoneType] = None

    #: Call a user given function on the snapshot data,
    #: e.g. for saving or plotting. Must have signature
    #: callback(time, state, registered_variables).
    activate_snapshot_callback: bool = False

    #: Return snapshots at specific time points.
    use_specific_snapshot_timepoints: bool = False

    #: The number of snapshots to return.
    num_snapshots: int = 10

    #: Fallback to the first order Godunov scheme.
    first_order_fallback: bool = False

    #: WENO nonlinear weights: ``False`` = classic Jiang-Shu
    #: ``alpha_k = c_k/(eps+IS_k)^2``, ``True`` = WENO-Z (Borges et al. 2008)
    #: ``alpha_k = c_k (1 + tau_5/(eps+IS_k))``. WENO-Z is scale invariant (the
    #: nonlinearity is a ratio of smoothness indicators, not a comparison
    #: against an absolute epsilon) and keeps the optimal linear weights at
    #: smooth extrema, where JS drops order and damps small-amplitude features.
    #: NOTE: currently implemented for the NATIVE backend only.
    weno_z: bool = False

    #: Absolute floor in the WENO smoothness denominators (JS and Z).
    weno_epsilon: float = 1e-7

    #: Evaluate the characteristic basis of the WENO projection at an
    #: ADMISSIBLE interface state: the interface sound speed comes from the
    #: averaged pressure, c^2 = gamma <p> / <rho>, and the enthalpy is rebuilt
    #: from it. ``False`` restores the previous c^2 = (gamma - 1)(<h> - v^2/2)
    #: from an UNWEIGHTED enthalpy mean and a MASS-WEIGHTED velocity; that
    #: combination is not the state of any gas, is not Galilean invariant, and
    #: at a density jump (ratio >~ 10) carrying a velocity jump of a few sound
    #: speeds its c^2 is negative -- the clamp then zeroes the acoustic upwind
    #: correction exactly at the strongest jumps (a cold dense slab rammed at
    #: Mach ~800 into tenuous gas blows up in two steps with it). Smooth-flow
    #: results agree with the old basis to the WENO dissipation level (the
    #: basis moves by O(dx^2)). Ideal gas only (the isothermal basis has a fixed
    #: sound speed). Native and Pallas.
    weno_admissible_face_state: bool = True

    #: Positivity-preserving WENO (Zhang & Shu 2012, J. Comput. Phys. 231,
    #: 2245), inside the reconstruction. With alpha the splitting speed of an
    #: interface, each split flux is f^+- = +-(alpha/2) w^+- with
    #: w^+- = q +- F/alpha - sum_s (1 - alpha_s/alpha) R_s L_s q. Every field
    #: that carries mass is split with the common alpha (the stencil's spectral
    #: radius), which makes the frozen-basis splitting monotone for every
    #: stencil cell; mass-free fields (hydro shear, isothermal-MHD Alfven) keep
    #: their own speed alpha_s. Each WENO face value is then pulled toward its
    #: upwind split state by the largest theta in [0, 1] keeping it, and its
    #: mirror about q +- F/alpha, admissible (positive density; positive
    #: pressure for an ideal gas; closed form, as pressure is concave). theta = 1
    #: in smooth flow. For ideal MHD a single q +- F/alpha is often NOT
    #: admissible at the fast speed when beta is low (Wu 2018, SIAM J. Numer.
    #: Anal. 56, 2124), so theta acts on the two weighted pairs of each cell's
    #: update instead: its own two mirror states (the cell's flux cancels; base
    #: q_i) and its two inflow states (Wu's generalized splitting: the
    #: magnetic-tension terms of the neighbours cancel). The SSPRK stages then
    #: rebuild the cell-centred B from the faces, pressure held, so each
    #: increment starts from the state it was evaluated at. Each forward-Euler
    #: stage is positivity preserving for C_cfl <= 1/2 (sum-of-speeds CFL),
    #: i.e. 0.75 with SSPRK4 (ideal MHD: up to the rare inadmissible inflow
    #: base, where theta = 0). Implies ``weno_admissible_face_state``. Native
    #: and Pallas (not the fused WENO+divergence kernel, which is then skipped).
    weno_positivity_preserving: bool = False

    #: If > 0, ADD a relative contribution ``weno_epsilon_relative * (amx*|q|)^2``
    #: to the WENO epsilon, where ``q`` is the local characteristic variable and
    #: ``amx`` the family's dissipation coefficient — i.e. compare the
    #: smoothness indicators against the local DATA SCALE instead of an
    #: absolute constant. With ``weno_epsilon`` alone the weights are fully
    #: nonlinear whenever the variables are O(10) or larger, regardless of how
    #: smooth the solution is. NATIVE backend only.
    weno_epsilon_relative: float = 0.0

    #: Differentiate through the WENO reconstruction with its nonlinear
    #: weights, characteristic eigenvectors and Lax-Friedrichs splitting speed
    #: FROZEN (``stop_gradient``): the tangent / adjoint is then that of the
    #: linear scheme the primal step actually used, the usual linearisation for
    #: WENO adjoints. The primal is unchanged. Needed for
    #: forward-mode derivatives through long runs with cold, near-uniform gas:
    #: there IS_k << weno_epsilon, d alpha / d IS ~ 2 / epsilon^3 ~ 1e21, and
    #: the float32 tangent overflows within a few years of a Cas A run.
    weno_ad_frozen_weights: bool = False

    #: If > 0: on interfaces whose colder side is below this factor times
    #: ``params.minimum_specific_pressure``, take the DERIVATIVE of the flux
    #: through the monotone LLF flux (the primal value is unchanged). The
    #: frozen-weight WENO linearisation is unstable at cold dense knots; this
    #: is the tangent-only analogue of the cold-crush LLF blend. Requires one
    #: of the positivity blend paths to be active (it lives in the blend).
    ad_tangent_llf_cold_factor: float = 0.0

    # physical modules

    #: Turbulent forcing configuration.
    turbulent_forcing_config: TurbulentForcingConfig = TurbulentForcingConfig()

    #: The configuration for the stellar wind module.
    wind_config: WindConfig = WindConfig()

    #: Cosmic rays
    cosmic_ray_config: CosmicRayConfig = CosmicRayConfig()

    #: The configuration for the cooling module.
    cooling_config: CoolingConfig = CoolingConfig()

    #: Frame tracking in z-direction
    #: shifting the frame to follow a
    #: turbulent radiative mixing layer
    frame_tracking: bool = False

    #: Configuration of the neural network force module.
    neural_net_force_config: NeuralNetForceConfig = NeuralNetForceConfig()

    #: Configuration of the CNN MHD corrector module.
    cnn_mhd_corrector_config: CNNMHDconfig = CNNMHDconfig()


def _parse_compute_capability(cc):
    """``"8.0"`` -> ``(8, 0)``; ``None`` for anything that is not ``major.minor``."""
    try:
        major, minor = str(cc).strip().split(".")[:2]
        return int(major), int(minor)
    except (ValueError, TypeError):
        return None


def gpu_compute_capability_at_least_80() -> bool:
    """Return whether JAX runs on GPUs that all have compute capability >= 8.0.

    Compute capability 8.0 (Ampere) is the floor for the Triton kernels the
    Pallas backend compiles to, so this is the predicate that decides whether
    OPTIMAL_BACKEND resolves to PALLAS.

    The question is asked of JAX, not of the machine: what matters is the
    platform the kernels will be compiled for. ``JAX_PLATFORMS=cpu`` on a GPU
    node therefore answers False (the kernels would run on XLA:CPU, where
    Pallas only has an interpreter), which ``nvidia-smi`` could not tell.
    The capability is read from ``jax.devices()``; ``nvidia-smi`` is only a
    fallback for GPU devices that do not expose it. Any failure is treated as
    "not capable" so the safe NATIVE_JAX fallback is chosen.

    Returns:
        True if JAX's default backend is a GPU backend and every one of its
        devices reports compute capability >= 8.0, False otherwise.
    """
    try:
        devices = jax.devices()
    except RuntimeError:
        return False
    if not devices or any(d.platform != "gpu" for d in devices):
        return False
    caps = [_parse_compute_capability(getattr(d, "compute_capability", None))
            for d in devices]
    if all(cc is not None for cc in caps):
        return all(cc >= (8, 0) for cc in caps)
    return _nvidia_smi_compute_capability_at_least_80()


def _nvidia_smi_compute_capability_at_least_80() -> bool:
    """Fallback of :func:`gpu_compute_capability_at_least_80`: ask ``nvidia-smi``
    (every visible NVIDIA GPU; False if it cannot be queried)."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    compute_caps = []
    for line in output.strip().splitlines():
        major, minor = map(int, line.strip().split("."))
        compute_caps.append((major, minor))

    if not compute_caps:
        return False

    return all(compute_cap >= (8, 0) for compute_cap in compute_caps)


def finalize_config(config: SimulationConfig, state_shape) -> SimulationConfig:
    """Fill in derived configuration fields and validate the configuration.

    Resolves the values that depend on the actual state shape or on
    cross-field consistency: the positivity-protection defaults, the number
    of cells per axis, the grid spacing, the geometry- and solver-specific
    overrides, the master gravity switch, the boundary defaults, and the
    disk-snapshot requirements.

    Args:
        config: The user-supplied simulation configuration.
        state_shape: The shape of the (unpadded) primitive state array, used
            to derive ``num_cells`` per axis.

    Returns:
        The finalized simulation configuration.
    """

    # The positivity-preserving reconstruction evaluates its characteristic
    # basis at the admissible interface state, so it switches that on too.
    if config.weno_positivity_preserving and not config.weno_admissible_face_state:
        config = config._replace(weno_admissible_face_state=True)

    # Resolve the OPTIMAL_BACKEND request into a concrete backend before any
    # downstream code inspects ``config.backend_config.backend``. PALLAS needs an Ampere-class
    # (compute capability >= 8.0) GPU for its Triton kernels; anywhere else we
    # fall back to the portable NATIVE_JAX backend.
    if config.backend_config.backend == OPTIMAL_BACKEND:
        if gpu_compute_capability_at_least_80():
            print("OPTIMAL_BACKEND: using the PALLAS backend (GPU compute capability >= 8.0).")
            config = config._replace(backend_config=config.backend_config._replace(backend=PALLAS))
        else:
            print("OPTIMAL_BACKEND: using the NATIVE_JAX backend (JAX is not running on a "
                  "compute capability >= 8.0 GPU).")
            config = config._replace(backend_config=config.backend_config._replace(backend=NATIVE_JAX))

    # Cosmic rays exist only in the finite-volume solver. Under the
    # finite-difference solver the registry does not add the CR variable, so
    # the CR code paths that still run (DSA injection, energy totals) index
    # variable -1 -- the pressure in 1D, the last passive scalar in a Cas A
    # configuration -- and the run silently returns a CR-free (and corrupted)
    # solution. Refuse instead.
    if config.cosmic_ray_config.cosmic_rays and config.solver_mode == FINITE_DIFFERENCE:
        raise ValueError(
            "cosmic_ray_config.cosmic_rays is not implemented for the "
            "finite-difference solver (the default solver_mode): the CR "
            "configuration would be silently ignored and the CR modules would "
            "write into variable -1. Use solver_mode=FINITE_VOLUME for cosmic "
            "rays."
        )

    # Reverse-mode / AD options.
    if config.ad_remat not in AD_REMAT_MODES:
        raise ValueError(
            f"ad_remat must be one of {AD_REMAT_MODES}, got {config.ad_remat!r}."
        )
    if int(config.ad_remat_chunks) < 1:
        raise ValueError(f"ad_remat_chunks must be >= 1, got {config.ad_remat_chunks!r}.")
    if config.passive_scalar_substep_loop not in (SUBSTEPS_AUTO, SUBSTEPS_DYNAMIC, SUBSTEPS_MASKED):
        raise ValueError(
            "passive_scalar_substep_loop must be SUBSTEPS_AUTO, SUBSTEPS_DYNAMIC "
            f"or SUBSTEPS_MASKED, got {config.passive_scalar_substep_loop!r}."
        )
    if int(config.max_passive_scalar_substeps) < 1:
        raise ValueError("max_passive_scalar_substeps must be >= 1.")
    if (config.passive_scalar_substep_loop == SUBSTEPS_DYNAMIC
            and config.differentiation_mode == BACKWARDS
            and (config.num_passive_scalars > 0 or config.track_shock_history)):
        print(
            "NOTE: passive_scalar_substep_loop = SUBSTEPS_DYNAMIC uses a traced "
            "trip count (a while loop); reverse-mode AD through the passive "
            "scalars will fail. Use SUBSTEPS_AUTO or SUBSTEPS_MASKED."
        )
    if config.ad_smooth_shock_latch and not config.track_shock_history:
        print("NOTE: ad_smooth_shock_latch has no effect without track_shock_history.")

    # weno_z is implemented in the FORWARD Pallas kernels; their hand-written
    # adjoints still hard-code Jiang-Shu, so reverse-mode gradients would not
    # match the forward pass. weno_epsilon_relative is native-only.
    if config.weno_epsilon_relative > 0.0 and \
            config.backend_config.backend == PALLAS:
        raise ValueError(
            "weno_epsilon_relative is implemented for the NATIVE_JAX backend "
            "only; pass BackendConfig(backend=NATIVE_JAX)."
        )
    if config.weno_z and config.backend_config.backend == PALLAS:
        print(
            "NOTE: weno_z runs the Pallas FORWARD kernels; the hand-written "
            "Pallas adjoints are still Jiang-Shu, so reverse-mode gradients "
            "would be inconsistent with the forward pass. Forward-only runs "
            "(evolution, convergence, timing) are unaffected."
        )

    # Approximate-rsqrt fast path: the forward MHD WENO kernel is switched to the
    # refined approximate rsqrt, but the hand-written Pallas adjoints still use
    # IEEE sqrt.  Purely forward runs (convergence, runtime) are unaffected; warn
    # only so a reverse-mode/AD user knows the forward and backward differ by the
    # rsqrt's ~1 ULP.
    if config.backend_config.use_approximate_rsqrt:
        print(
            "NOTE: backend_config.use_approximate_rsqrt is ON — the MHD WENO "
            "forward kernel uses approximate rsqrt while its adjoints use IEEE "
            "sqrt, so reverse-mode gradients are ~1 ULP inconsistent with the "
            "forward. Fine for forward-only runs; turn off for exact AD."
        )

    # ``default_positivity_protection`` is a casual on/off switch for the STATE
    # floors only: the default ``False`` is a clean slate (no per-stage /
    # per-step flooring). When set, turn the floors on (HARD_FLOOR) unless the
    # user explicitly chose a mode. The read-only clamps (clamp_in_estimates)
    # are decoupled and left untouched (default on), as are the feature toggles
    # (deepvoid_blend, preserving_flux, conservative redistribution,
    # vacuum_rest, nan_safe).
    positivity_config = config.positivity_config
    if positivity_config.default_positivity_protection:
        config = config._replace(positivity_config=positivity_config._replace(
            per_stage_mode=(POSITIVITY_HARD_FLOOR
                            if positivity_config.per_stage_mode == POSITIVITY_NONE
                            else positivity_config.per_stage_mode),
            per_step_mode=(POSITIVITY_HARD_FLOOR
                           if positivity_config.per_step_mode == POSITIVITY_NONE
                           else positivity_config.per_step_mode),
        ))

    if jax.config.jax_enable_x64:
        config._replace(numerical_precision=DOUBLE_PRECISION)
    else:
        config._replace(numerical_precision=SINGLE_PRECISION)

    # set the number of cells
    if config.dimensionality == 1:
        num_cells_x = state_shape[-1]
        config = config._replace(num_cells=StaticIntVector(num_cells_x, -1, -1))
    if config.dimensionality == 2:
        num_cells_x, num_cells_y = state_shape[-2:]
        config = config._replace(num_cells=StaticIntVector(num_cells_x, num_cells_y, -1))
    elif config.dimensionality == 3:
        num_cells_x, num_cells_y, num_cells_z = state_shape[-3:]
        config = config._replace(num_cells=StaticIntVector(num_cells_x, num_cells_y, num_cells_z))

    if isinstance(config.box_size, float):
        config = config._replace(
            box_size=StaticFloatVector(
                config.box_size,
                config.box_size,
                config.box_size
            )
        )

    # For now we assume the grid spacing is the same in all dimensions, so the
    # scalar ``grid_spacing`` is taken from the x-axis and the other axes are
    # only checked for consistency below. This restriction can be lifted once
    # the solver accepts a per-axis grid-spacing vector.
    grid_spacing_vec = config.box_size / config.num_cells

    if config.dimensionality == 1:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
    elif config.dimensionality == 2:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
        if not math.isclose(grid_spacing_vec.x, grid_spacing_vec.y):
            raise ValueError(
                "For now, we assume the grid spacing is the same in all dimensions. "
                f"Got grid spacing {grid_spacing_vec}."
            )
    elif config.dimensionality == 3:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
        if not (math.isclose(grid_spacing_vec.x, grid_spacing_vec.y) and math.isclose(grid_spacing_vec.x, grid_spacing_vec.z)):
            raise ValueError(
                "For now, we assume the grid spacing is the same in all dimensions. "
                f"Got grid spacing {grid_spacing_vec}."
            )

    if config.geometry == SPHERICAL:
        print(
            "For spherical geometry, only HLL is currently supported. Also, only the unsplit mode has been tested."
        )
        # SPHERICAL is intrinsically 1D in this code; pick the x component
        # so grid_spacing stays a scalar (otherwise CFL divisions blow up
        # because StaticFloatVector can't be divided by a scalar wave speed).
        config = config._replace(grid_spacing=(config.box_size / config.num_cells).x)

        if config.riemann_solver != HLL:
            print("Setting HLL Riemann solver for spherical geometry.")
            config = config._replace(riemann_solver=HLL)

        if config.split != SPLIT:
            print("Setting unsplit mode for spherical geometry")
            config = config._replace(split=SPLIT)

        if config.limiter == VAN_ALBADA or config.limiter == VAN_ALBADA_PP:
            print("Setting minmod limiter for spherical geometry")
            config = config._replace(limiter=MINMOD)

        if config.time_integrator != MUSCL:
            print("Setting MUSCL time integrator for spherical geometry")
            config = config._replace(time_integrator=MUSCL)

    # master gravity switch: active if self-gravity and/or an external
    # potential is used. This gates the (shared) gravity source-term machinery.
    config = config._replace(gravity_config=config.gravity_config._replace(
        gravity=config.gravity_config.self_gravity
        or config.gravity_config.external_potential
    ))

    if config.gravity_config.gravity and (config.limiter != MINMOD):
        print(
            "Curiously, in self-gravitating systems, the VAN_ALBADA limiters seem to cause crashes."
        )
        print("Setting MINMOD limiter for gravity.")
        config = config._replace(limiter=MINMOD)

    # Finite-difference-specific checks.
    if config.solver_mode == FINITE_DIFFERENCE:

        # The WENO5 stencil reaches three cells beyond each interface, so the
        # default ghost-cell count (reconstruction_order + 1 = 2) is too few:
        # 1D runs then read wrapped / stale values at the edges, and smooth
        # periodic problems converged at first order. Same rule as 2D/3D.
        if config.dimensionality == 1:
            if config.boundary_settings == BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ):
                config = config._replace(boundary_handling=PERIODIC_ROLL, num_ghost_cells=0)
            elif config.boundary_handling == GHOST_CELLS:
                config = config._replace(num_ghost_cells=max(config.num_ghost_cells, 4))

        if config.dimensionality == 3 and config.boundary_settings == BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ):
            # Fully periodic boundaries are enforced more cheaply by rolling the
            # arrays (PERIODIC_ROLL) than by maintaining explicit ghost cells.
            print(
                "For 3D simulations with periodic boundaries, setting boundary handling to " \
                "PERIODIC_ROLL and num_ghost_cells to 0 for better performance."
            )
            config = config._replace(boundary_handling=PERIODIC_ROLL, num_ghost_cells=0)
        else:
            if config.dimensionality == 3:
                config = config._replace(boundary_handling=GHOST_CELLS, num_ghost_cells=4)

        if config.dimensionality == 2 and config.boundary_settings == BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ):
            # Fully periodic boundaries are enforced more cheaply by rolling the
            # arrays (PERIODIC_ROLL) than by maintaining explicit ghost cells.
            print(
                "For 2D simulations with periodic boundaries, setting boundary handling to " \
                "PERIODIC_ROLL and num_ghost_cells to 0 for better performance."
            )
            config = config._replace(boundary_handling=PERIODIC_ROLL, num_ghost_cells=0)
        else:
            if config.dimensionality == 2:
                config = config._replace(boundary_handling=GHOST_CELLS, num_ghost_cells=4)

        # The FD scheme has two supported time integrators: the SSPRK4
        # Spiteri-Ruuth 3-register scheme (default) and the Carpenter-Kennedy
        # 2N-storage LSRK4 ("RK4_LSRK") which trades CFL margin for one fewer
        # full-state buffer.  Anything else falls back to SSPRK4.
        if config.time_integrator not in (RK4_SSP, RK4_LSRK):
            print(
                "Setting time integrator to RK4_SSP for finite difference solver mode."
            )
            config = config._replace(time_integrator=RK4_SSP)

        if config.boundary_handling == PERIODIC_ROLL:
            config = config._replace(num_ghost_cells=0)

        if config.boundary_handling == GHOST_CELLS and (config.diffusion or config.thermal_conduction
                                                        or config.resistivity):
            config = config._replace(num_ghost_cells=max(config.num_ghost_cells, 6))

        if config.resistivity:
            if not config.mhd or config.dimensionality != 3:
                raise ValueError("resistivity requires 3D finite-difference MHD")
            if config.equation_of_state != ISOTHERMAL:
                raise ValueError(
                    "resistivity is implemented for the ISOTHERMAL EOS only: the "
                    "ohmic heating eta J^2 is not added to the energy equation"
                )

    # Pick sensible default boundary conditions when the user left them unset.
    if config.boundary_settings is None:
        if config.geometry == CARTESIAN:
            print("Automatically setting open boundaries for Cartesian geometry.")
            if config.dimensionality == 1:
                config = config._replace(
                    boundary_settings=BoundarySettings1D(
                        left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
                    )
                )
            else:
                config = config._replace(boundary_settings=BoundarySettings())
        elif config.geometry == SPHERICAL and config.dimensionality == 1:
            print(
                "Automatically setting reflective left and open right boundary for spherical geometry."
            )
            config = config._replace(
                boundary_settings=BoundarySettings1D(
                    left_boundary=REFLECTIVE_BOUNDARY, right_boundary=OPEN_BOUNDARY
                )
            )

    if config.wind_config.stellar_wind:
        print(
            "For stellar wind simulations, we need source term aware timesteps, turning on."
        )
        config = config._replace(source_term_aware_timestep=True)

    # Disk-snapshot (Orbax) mode requirements.
    if config.snapshot_storage_mode == TO_DISK:
        if not config.snapshot_storage_path:
            raise ValueError(
                "snapshot_storage_mode == TO_DISK requires a non-empty "
                "snapshot_storage_path (the directory the Orbax checkpoints "
                "are written to)."
            )
        if config.differentiation_mode != FORWARDS:
            raise ValueError(
                "snapshot_storage_mode == TO_DISK is forward-mode only; "
                "set differentiation_mode = FORWARDS."
            )

    return config


def riemann_solver_to_string(riemann_solver: int) -> str:
    """Return the human-readable name of a Riemann-solver constant."""
    if riemann_solver == HLL:
        return "HLL"
    elif riemann_solver == HLLC:
        return "HLLC"
    elif riemann_solver == HLLC_LM:
        return "HLLC_LM"
    elif riemann_solver == LAX_FRIEDRICHS:
        return "Lax-Friedrichs"
    elif riemann_solver == HYBRID_HLLC:
        return "Hybrid HLLC"
    elif riemann_solver == AM_HLLC:
        return "AM HLLC"


def limiter_to_string(limiter: int) -> str:
    """Return the human-readable name of a slope-limiter constant."""
    if limiter == MINMOD:
        return "Minmod"
    elif limiter == SUPERBEE:
        return "Superbee"
    elif limiter == OSHER:
        return "Osher"
    elif limiter == DOUBLE_MINMOD:
        return "Double Minmod"
    elif limiter == VAN_ALBADA:
        return "Van Albada"
    elif limiter == VAN_ALBADA_PP:
        return "Van Albada PP"


def solver_mode_to_string(solver_mode: int) -> str:
    """Return the short label (``"FV"`` / ``"FD"``) of a solver-mode constant."""
    if solver_mode == FINITE_VOLUME:
        return "FV"
    elif solver_mode == FINITE_DIFFERENCE:
        return "FD"


def config_to_string(config: SimulationConfig) -> str:
    """Return a compact one-line description of the solver configuration."""
    if config.solver_mode == FINITE_VOLUME:
        return f"FV, {riemann_solver_to_string(config.riemann_solver)}, {limiter_to_string(config.limiter)}, {config.num_cells.x} cells"
    elif config.solver_mode == FINITE_DIFFERENCE:
        return f"FD, {config.num_cells.x} cells"