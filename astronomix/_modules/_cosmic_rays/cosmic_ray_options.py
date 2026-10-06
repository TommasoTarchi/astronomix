"""
Configuration and parameter containers for the cosmic-ray module.

``CosmicRayConfig`` holds the static (compile-time) switches that turn the
cosmic-ray physics on and off, while ``CosmicRayParams`` holds the runtime
numerical values controlling diffusive shock acceleration. The integer tags
``STRONGEST_SHOCK`` and ``OUTERMOST_SHOCK`` select which flagged shock receives
the injection.
"""

# typing
from typing import NamedTuple


# -------------------------------------------------------------
# ============== ↓ Shock-selection constants ↓ ================
# -------------------------------------------------------------

#: Inject at the shock with the largest pressure-smoothness sensor among the
#: cells that pass the Pfrommer et al. (2017) shock criteria (default).
STRONGEST_SHOCK = 0

#: Inject at the outermost flagged shock (largest cell index, i.e. largest
#: radius in spherical geometry): the forward shock of an ejecta-driven
#: remnant, whatever the relative sensor strength of the other shocks.
OUTERMOST_SHOCK = 1

# -------------------------------------------------------------
# ============== ↑ Shock-selection constants ↑ ================
# -------------------------------------------------------------


class CosmicRayConfig(NamedTuple):
    """Static switches of the cosmic-ray module."""

    #: main switch for cosmic rays
    cosmic_rays: bool = False

    #: turn on injection of CRs at shocks
    diffusive_shock_acceleration: bool = False

    #: which flagged shock receives the injection: ``STRONGEST_SHOCK``
    #: (default) or ``OUTERMOST_SHOCK`` (forward shock).
    shock_selection: int = STRONGEST_SHOCK

    #: for ``OUTERMOST_SHOCK``: the sensor maximum is searched only among the
    #: flagged cells within this many cells inside the outermost flagged cell
    #: (a captured shock is ~3-5 cells wide, so this brackets exactly one).
    outermost_shock_window_cells: int = 8


class CosmicRayParams(NamedTuple):
    """Runtime parameters of diffusive shock acceleration."""

    #: starting time of diffusive shock acceleration
    diffusive_shock_acceleration_start_time: float = 0.0

    #: efficiency of diffusive shock acceleration: the fraction zeta of the
    #: energy dissipated at the shock that the gas loses to accelerated CRs
    #: (Pfrommer et al. 2017, Eq. 36).
    diffusive_shock_acceleration_efficiency: float = 0.1

    #: fraction of the freshly accelerated CR energy that escapes the system
    #: at injection (upstream escape of the highest-energy particles). The gas
    #: loses ``zeta * dissipated``; the CR fluid gains
    #: ``(1 - escape_fraction) * zeta * dissipated``; the rest leaves the
    #: domain, i.e. total energy is deliberately not conserved when > 0.
    escape_fraction: float = 0.0

    #: safety cap: a single injection step may remove at most this fraction of
    #: a cell's thermal energy. Never active for a resolved shock (the per-step
    #: injection there is ~zeta * C_cfl / (zone width) of e_th, a few per cent);
    #: it only prevents a negative gas pressure when the shock zone is
    #: mis-identified (e.g. a shock that has not formed yet).
    max_thermal_fraction_per_step: float = 0.5
