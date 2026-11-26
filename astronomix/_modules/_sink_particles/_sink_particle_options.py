"""
Configuration container for sink particles.

Sink particles grow from seeds, one in every cell, whose activation rises
from 0 to 1 with smooth versions of the creation checks of Federrath et al.
(2010), ApJ 713, 269, Section 2.2, and they accrete the gas above the density
threshold within their accretion radius (Section 2.3), in proportion to their
activation. The widths of the smooth checks are set here; at their hard
limit the checks are those of the paper. The density threshold is derived
from the Jeans length resolution (Eq. 32 of the paper).
They move with a kick-drift-kick leapfrog, once per hydro step, and all their
gravity goes through the grid (Gong & Ostriker 2013, ApJS 204, 8). All options
here are static: changing them triggers recompilation.
"""

# typing
from typing import NamedTuple


class SinkParticleConfig(NamedTuple):
    """
    Static configuration of the sink particles: their formation, their
    accretion of gas and their motion.

    A sink moves with a kick-drift-kick leapfrog, once per hydro step, under
    the gravity of the gas, of the other sinks and of the external potential.
    All of it goes through the grid: the sink's mass is spread over the 3×3×3
    cells around it with triangular-shaped-cloud weights and added to the gas
    density of the Poisson solve, and its acceleration is read back from the
    same cells with the same weights, so a sink feels no force from itself
    and the forces between two masses are equal and opposite. Adaptive steps
    keep every sink from moving more than half a cell per step.

    Along a non-periodic axis a sink can leave the box. It keeps its slot, a
    warning is printed, and its gravity fades out between one cell inside and
    one cell outside the edge; beyond that it moves in a straight line. Where
    the Poisson solve treats the edge of the box differently from the gas
    boundaries (periodic gas with ``poisson_manual_open_boundaries``, mixed
    boundaries), a sink within 4 cells of the edge (2 with VL2) feels a force
    from itself, and the forces between masses are not exactly equal and
    opposite.
    """

    #: Switch sink particles on or off.
    sink_particles: bool = False

    #: Size of the sink particle arrays. JIT requires fixed array sizes, so
    #: this many slots are allocated up front; every seed whose activation
    #: has ever exceeded 0 holds one. When a new seed would need more slots,
    #: the run stops with an error. Sinks passed into a run, or
    #: restored from a checkpoint, with fewer slots get empty slots appended;
    #: with more slots, the run does not start.
    max_num_sinks: int = 256

    #: Length of the list of cells whose inactive seed can get a creation
    #: weight above 0: the interior cells that pass the density and
    #: converging-flow checks and are not blocked by a sink or by an earlier
    #: seed of their cell. Only these get the checks that need a control
    #: volume (potential minimum, Jeans instability, bound state). Cells
    #: beyond this number are skipped for that step, with a warning.
    max_num_candidates: int = 128

    #: Accretion radius r_acc in units of the grid spacing. It sets both the
    #: radius of the control volume and the density threshold. Federrath et
    #: al. (2010), Section 4.2.1, use r_acc = 2.5 grid cells. With ghost cells,
    #: its integer part must not exceed num_ghost_cells.
    accretion_radius_in_cells: float = 2.5

    #: Width ω of the smooth steps that replace the yes/no creation and
    #: accretion checks, in units of each check's own margin (e.g. ρ / ρ_res − 1
    #: for the density check). A check contributes 0 below its threshold and 1
    #: beyond threshold + ω. 0 gives the hard checks. Must be ≥ 0.
    check_width: float = 0.1

    #: Width ℓ, in cells, over which spatial weights pass from 1 to 0: the edge
    #: of the accretion radius, the passage from a cell to its neighbour, and
    #: the softening of a sink's potential in the accretion bound check. 0
    #: gives sharp edges. Must lie in [0, 1].
    spatial_width_in_cells: float = 0.5

    #: Sharpness β of the sharing of a cell's gas between the seeds that can
    #: accrete it: each seed's share is weighted by e^(−β η), with η the
    #: specific energy of the gas relative to the seed in units of c_s². As
    #: β → ∞ the most bound seed takes it all. Must be > 0; ``float("inf")``
    #: is allowed.
    sharing_sharpness: float = 10.0

    #: Mass ε, in units of ρ_res ΔV, added to a seed's mass in the
    #: denominator of its position update, so that a seed that has accreted
    #: little moves little. 0 moves a seed to the centre of mass of itself and
    #: the accreted gas. Must be ≥ 0.
    position_update_mass: float = 0.01
