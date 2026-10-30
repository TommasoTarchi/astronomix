"""
Configuration container for sink particle formation.

Sink particles are created following the checks of Federrath et al. (2010),
ApJ 713, 269, Section 2.2, with a safety density threshold above which the
checks that judge by gas motion are skipped. All options here are static:
changing them triggers recompilation. The density threshold is derived from
the Jeans length resolution (Eq. 32 of the paper).
"""

# typing
from typing import NamedTuple


class SinkParticleConfig(NamedTuple):
    """Static configuration of the sink particle formation."""

    #: Switch sink particle formation on or off.
    sink_particles: bool = False

    #: Size of the sink particle arrays. JIT requires fixed array sizes, so
    #: this many slots are allocated up front. When a new sink would need
    #: more slots, the run stops with an error. Sinks passed into a run, or
    #: restored from a checkpoint, with fewer slots get empty slots appended;
    #: with more slots, the run does not start.
    max_num_sinks: int = 64

    #: Maximum number of cells that can be examined per time step with the
    #: control-volume checks (Jeans instability, bound state, proximity).
    #: Cells beyond this number are ignored for that step (with a warning).
    max_num_candidates: int = 64

    #: Accretion radius r_acc in units of the grid spacing. It sets both the
    #: radius of the control volume and the density threshold. Federrath et
    #: al. (2010), Section 4.2.1, use r_acc = 2.5 grid cells.
    accretion_radius_in_cells: float = 2.5

    #: Jeans length N, in units of the grid spacing, that defines the safety
    #: density threshold ρ_safe = π c_s² / (G N² Δx²), the density at which
    #: the Jeans length is N cells. Above ρ_safe, the checks that judge by gas
    #: motion are skipped: the converging flow and bound state checks of the
    #: creation, and the bound and radial velocity checks of the accretion. On
    #: a uniform grid these checks can keep rejecting collapsing gas, e.g. a
    #: dense layer fed by fast inflow, until the Jeans length is far below a
    #: cell. The default puts ρ_safe at the Larson–Penston density at half a
    #: cell, the creation threshold of Gong & Ostriker (2013), ApJS 204, 8,
    #: about 22 ρ_res for r_acc = 2.5 cells. The safety threshold is not part
    #: of the method of Federrath et al. (2010).
    safety_jeans_length_in_cells: float = 1.06
