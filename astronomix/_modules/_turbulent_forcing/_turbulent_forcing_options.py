"""
Configuration and parameter containers for turbulent forcing.

``TurbulentForcingConfig`` holds the static switches (forcing on/off, vacuum
protection, and the choice of Ornstein-Uhlenbeck versus white-in-time forcing),
while ``TurbulentForcingParams`` holds the tunable physical parameters.
"""

# typing
from typing import NamedTuple


class TurbulentForcingConfig(NamedTuple):
    turbulent_forcing: bool = False

    #: Use Ornstein-Uhlenbeck (temporally correlated) forcing instead of the
    #: default white-in-time forcing. The OU field persists across steps and is
    #: evolved as f <- a f + sqrt(1 - a^2) xi (a = exp(-dt / correlation_time)),
    #: with xi a fresh unit-rms solenoidal field peaked at forcing_wavenumber.
    #: It is applied as a constant-amplitude acceleration (velocity += F0 f dt),
    #: which is state-independent (clean adjoint) and -- unlike the white
    #: forcing -- lets rotation organise coherent structures (columns).
    ou_forcing: bool = False

    #: Normalise the OU field to inject exactly ``energy_injection_rate * dt``
    #: each step (the same quadratic the white forcing solves) instead of
    #: applying the constant amplitude ``forcing_amplitude``. This is what
    #: AthenaK's ``turb_driver`` does (``<turb_driving> dedt``); the constant-F0
    #: default is kept so existing F0 calibrations are untouched.
    ou_exact_injection: bool = False

    #: Use the AthenaK-style DISCRETE driving band (static switch; the band
    #: edges and exponent themselves live in ``TurbulentForcingParams`` so they
    #: stay traceable). Selecting the spectrum must be a static choice because
    #: the params are jit-traced.
    banded_spectrum: bool = False

    #: AthenaPK-style FEW-MODES driving (``few_modes_ft``): a static tuple of
    #: integer mode triples ``((nx, ny, nz), ...)`` in mode-number units
    #: (n = k L / 2pi). When non-empty the OU spectrum has power ONLY on these
    #: modes, weighted by AthenaPK's parabolic envelope
    #: ``(n / n_pk)^2 (2 - (n / n_pk)^2)`` with ``n_pk`` from
    #: ``forcing_wavenumber``; the smooth and banded spectra are ignored. Static
    #: because the mode set fixes the trace.
    forcing_modes: tuple = ()

    #: Normalise the OU field to unit rms at *application* time every step
    #: (AthenaPK rescales its acceleration to ``accel_rms`` each cycle), so the
    #: applied acceleration rms is exactly ``forcing_amplitude`` rather than
    #: fluctuating around it. The persistent field itself is left untouched.
    ou_unit_rms_each_step: bool = False


class TurbulentForcingParams(NamedTuple):
    energy_injection_rate: float = 2.0

    #: OU forcing correlation time tau_f (~ one eddy turnover). Only used when
    #: TurbulentForcingConfig.ou_forcing is True.
    correlation_time: float = 1.0

    #: OU forcing peak wavenumber k_f (in physical units, k = 2 pi n / L). The
    #: solenoidal forcing spectrum k^6 exp(-8 k / kpk) is peaked at k_f by
    #: setting kpk = k_f / 0.75. Only used when ou_forcing is True.
    forcing_wavenumber: float = 4.0

    #: OU forcing amplitude F0 (acceleration scale); tunes the stationary
    #: u_rms. Only used when ou_forcing is True and ou_exact_injection is False.
    forcing_amplitude: float = 1.0

    #: AthenaK-style DISCRETE driving band in mode number n = k L / 2pi. When
    #: ``forcing_nhigh > 0`` the OU spectrum is confined to
    #: ``forcing_nlow <= n <= forcing_nhigh`` with an isotropic
    #: ``k^-(forcing_expo+2)/2`` envelope (``<turb_driving> nlow/nhigh/expo``),
    #: replacing the smooth peaked spectrum selected by ``forcing_wavenumber``.
    forcing_nlow: int = 0
    forcing_nhigh: int = 0
    forcing_expo: float = 5.0 / 3.0
