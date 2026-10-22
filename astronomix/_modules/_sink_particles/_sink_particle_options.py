"""
Configuration container for sink particle formation.

Sink particles are created following the checks of Federrath et al. (2010),
ApJ 713, 269, Section 2.2, and accrete gas following Section 2.3. All options
here are static (changing them triggers recompilation). The density threshold
is derived from the Jeans length resolution (Eq. 32 of the paper). The
defaults of the threshold factor, the Jeans factor, the check switches, the
accretion cutoff factor and the radial velocity tolerance reproduce the
criteria of the paper and of Federrath's FLASH code; other values loosen or
tighten them.
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

    #: Factor multiplying the density threshold of Eq. 32,
    #: ρ_thr = π c_s² / (4 G r_acc²). The same threshold is used for creation
    #: and for accretion, where accreted cells are left at ρ_thr. Must be
    #: positive.
    density_threshold_factor: float = 1.0

    #: Whether the creation step requires the flow to converge toward the cell
    #: along every axis (Section 2.2.3).
    converging_flow_check: bool = True

    #: Factor f in front of the thermal energy in the Jeans instability check
    #: of the creation step, |E_grav| > f E_th + E_mag (Section 2.2.5). Must
    #: not be negative.
    jeans_thermal_energy_factor: float = 2.0

    #: Whether the creation step requires the gas in the control volume to be
    #: bound, E_grav + E_th + E_kin + E_mag < 0 (Section 2.2.6).
    formation_bound_check: bool = True

    #: Whether accretion requires the gas to be bound to the sink,
    #: E_grav + E_kin < 0 (Section 2.3). The gas in the cell containing the
    #: sink is accreted without this check either way.
    accretion_bound_check: bool = True

    #: Factor multiplying r_acc to give the radius r_cut of the accretion
    #: sphere: a sink accretes from the cells whose centres lie within r_cut
    #: (Section 2.3), and the gravitational energy of the accretion bound check
    #: is measured from the edge of this sphere. The cell containing the sink
    #: is accreted from at any r_cut. The creation step (control volume and
    #: proximity) still uses r_acc. Must be positive.
    accretion_distance_cutoff_factor: float = 1.0

    #: Largest radial velocity, away from the sink and in units of the local
    #: sound speed, at which gas is still accreted (Section 2.3). Must not be
    #: negative.
    radial_velocity_tolerance_in_sound_speeds: float = 1e-5
