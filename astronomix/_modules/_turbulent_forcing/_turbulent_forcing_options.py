"""
Configuration and parameter containers for turbulent forcing.

``TurbulentForcingConfig`` holds the static switches (forcing on/off, the choice
of Ornstein-Uhlenbeck versus white-in-time forcing, the forcing spectrum and its
normalisation), while ``TurbulentForcingParams`` holds the tunable physical
parameters.
"""

# typing
from typing import NamedTuple, Tuple


class TurbulentForcingConfig(NamedTuple):
    """
    Static switches of the turbulent forcing. Changing them requires
    recompilation; the traceable forcing parameters live in
    :class:`TurbulentForcingParams`.
    """

    #: Drive the velocity field with a solenoidal random acceleration, by
    #: default white in time at the fixed energy injection rate
    #: ``TurbulentForcingParams.energy_injection_rate`` (see ``ou_forcing``
    #: for the temporally correlated alternative).
    turbulent_forcing: bool = False

    #: Use Ornstein-Uhlenbeck (temporally correlated) forcing instead of the
    #: default white-in-time forcing. The OU field persists across steps and is
    #: evolved as f <- a f + sqrt(1 - a^2) xi (a = exp(-dt / correlation_time)),
    #: with xi a fresh unit-rms solenoidal field (by default with the smooth
    #: spectrum peaked at forcing_wavenumber; see ``banded_spectrum`` and
    #: ``forcing_modes``). By default it is applied as a constant-amplitude
    #: acceleration (velocity += F0 f dt), which is state-independent (clean
    #: adjoint) and -- unlike the white forcing -- lets rotation organise
    #: coherent structures (columns). ``ou_exact_injection`` instead makes the
    #: amplitude depend on the state, and ``ou_unit_rms_each_step`` rescales the
    #: applied field every step.
    ou_forcing: bool = False

    #: Coarse spectral-synthesis resolution for the OU forcing. ``0`` (the
    #: default) keeps the full-grid path: each fresh solenoidal draw is
    #: synthesised by an inverse FFT on the whole simulation grid. A positive
    #: value ``nc`` instead draws and OU-evolves the forcing in spectral space
    #: on a small ``nc^3`` grid and evaluates the band-limited field on the
    #: fine grid by exact per-axis inverse-DFT matrix products. Because the
    #: forcing spectrum k^6 exp(-8 k / kpk) is negligible beyond a few kpk,
    #: ``nc = 64`` already carries the full spectrum to fp32 precision for the
    #: standard k_f -- while making the forcing scale to sharded multi-node
    #: grids (the full-grid inverse FFT would otherwise be replicated per
    #: device by the SPMD partitioner) and cutting its cost at large N. The OU
    #: update is linear, so evolving the spectrum and synthesising afterwards
    #: is mathematically identical to evolving the real-space field. Only the
    #: smooth spectrum is available here (not ``banded_spectrum`` or
    #: ``forcing_modes``).
    synthesis_resolution: int = 0

    #: Normalise the OU field to inject exactly ``energy_injection_rate * dt``
    #: each step (the same quadratic the white forcing solves) instead of
    #: applying the constant amplitude ``forcing_amplitude``. This is what
    #: AthenaK's ``turb_driver`` does (``<turb_driving> dedt``); the constant-F0
    #: default is kept so existing F0 calibrations are untouched.
    ou_exact_injection: bool = False

    #: Use the AthenaK-style DISCRETE driving band (OU forcing only; the white
    #: forcing always uses its smooth spectrum). The band edges and exponent
    #: live in ``TurbulentForcingParams`` (``forcing_nlow``, ``forcing_nhigh``,
    #: ``forcing_expo``) so they stay traceable, while selecting the spectrum
    #: must be a static choice because the params are jit-traced.
    banded_spectrum: bool = False

    #: AthenaPK-style FEW-MODES driving (``few_modes_ft``, OU forcing only): a
    #: static tuple of integer mode triples ``((nx, ny, nz), ...)`` in
    #: mode-number units (n = k L / 2pi). When non-empty the OU spectrum has
    #: power ONLY on these modes, weighted by AthenaPK's parabolic envelope
    #: ``(n / n_pk)^2 (2 - (n / n_pk)^2)`` with ``n_pk`` from
    #: ``forcing_wavenumber``; the smooth and banded spectra are ignored. Static
    #: because the mode set fixes the trace.
    forcing_modes: Tuple[Tuple[int, int, int], ...] = ()

    #: Normalise the OU field to unit rms at *application* time every step
    #: (AthenaPK rescales its acceleration to ``accel_rms`` each cycle), so the
    #: applied acceleration rms is exactly ``forcing_amplitude`` rather than
    #: fluctuating around it. The persistent field itself is left untouched.
    ou_unit_rms_each_step: bool = False


class TurbulentForcingParams(NamedTuple):
    """
    Traceable parameters of the turbulent forcing. They can be changed without
    recompilation; the static switches live in :class:`TurbulentForcingConfig`.
    """

    #: Kinetic energy injected per unit time into the whole box by the
    #: white-in-time forcing (and by the OU forcing with
    #: ``ou_exact_injection``).
    energy_injection_rate: float = 2.0

    #: OU forcing correlation time tau_f (~ one eddy turnover). Only used when
    #: TurbulentForcingConfig.ou_forcing is True.
    correlation_time: float = 1.0

    #: OU forcing peak wavenumber k_f (in physical units, k = 2 pi n / L). The
    #: smooth solenoidal forcing spectrum k^6 exp(-8 k / kpk) is peaked at k_f
    #: by setting kpk = k_f / 0.75; with ``forcing_modes`` it sets the envelope
    #: peak ``n_pk = k_f L / 2pi`` instead, and the banded spectrum ignores it.
    #: Only used when ou_forcing is True.
    forcing_wavenumber: float = 4.0

    #: OU forcing amplitude F0 (acceleration scale); tunes the stationary
    #: u_rms. Only used when ou_forcing is True and ou_exact_injection is False.
    forcing_amplitude: float = 1.0

    #: AthenaK-style DISCRETE driving band in mode number n = k L / 2pi, used
    #: only with ``TurbulentForcingConfig.banded_spectrum`` (OU forcing): the
    #: OU spectrum is then confined to ``forcing_nlow <= n <= forcing_nhigh``
    #: with an isotropic ``k^-(forcing_expo+2)/2`` envelope (AthenaK's
    #: ``<turb_driving> nlow/nhigh/expo``), replacing the smooth peaked
    #: spectrum selected by ``forcing_wavenumber``. The band must satisfy
    #: ``forcing_nhigh >= forcing_nlow > 0``; with the default
    #: ``forcing_nhigh = 0`` the spectrum is empty and nothing is driven. The
    #: params are traced, so this cannot be validated when the configuration
    #: is finalised.
    forcing_nlow: int = 0

    #: Upper edge of the discrete driving band (see ``forcing_nlow``).
    forcing_nhigh: int = 0

    #: Spectral exponent of the discrete driving band (see ``forcing_nlow``).
    forcing_expo: float = 5.0 / 3.0
