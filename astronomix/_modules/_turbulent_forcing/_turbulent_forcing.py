"""
Turbulent forcing of the velocity field.

Provides two driving schemes plus a vacuum-protection helper. The default
white-in-time forcing draws a fresh solenoidal field each step and rescales it
so that a prescribed energy injection rate is met; the Ornstein-Uhlenbeck
variant carries a temporally correlated solenoidal field across steps and
applies it as a constant-amplitude acceleration. The construction of the
solenoidal forcing fields follows https://arxiv.org/pdf/2304.04360.
"""

# general
from functools import partial

# numerics
import numpy as np

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import PERIODIC_ROLL

# astronomix containers
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingParams,
)
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables


@partial(jax.jit, static_argnames=["config"])
def _create_forcing_field(
    key,
    config: SimulationConfig,
):
    """Draw a fresh solenoidal (divergence-free) random forcing field.

    Builds a random field in Fourier space with the power spectrum
    k^6 exp(-8 k / kpk), removes the compressible component to make it
    solenoidal, and transforms it back to real space.

    Args:
        key: The PRNG key.
        config: The simulation configuration.

    Returns:
        ``(key, wx_real, wy_real, wz_real)``: the advanced PRNG key and the three
        real-space components of the solenoidal forcing field.
    """

    xsize = config.box_size.x
    ysize = config.box_size.y
    zsize = config.box_size.z

    # The field lives on the physical grid; with ghost cells it is extended
    # periodically by ``_on_state_grid`` when it is applied.
    nx = config.num_cells.x
    ny = config.num_cells.y
    nz = config.num_cells.z

    # Wavenumbers along each axis via fftfreq.
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=xsize/nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=ysize/ny)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=zsize/nz)

    # Broadcast the 1D wavenumber arrays into 3D rather than materialising a
    # full meshgrid.
    kx_3d = kx.reshape(nx, 1, 1)
    ky_3d = ky.reshape(1, ny, 1)
    kz_3d = kz.reshape(1, 1, nz)

    k_squared = kx_3d**2 + ky_3d**2 + kz_3d**2
    kk = jnp.sqrt(k_squared)

    # Forcing power spectrum peaked at intermediate wavenumbers.
    kpk = 4.0 * jnp.pi / config.box_size.x
    Pk = kk**6 * jnp.exp(-8.0 * kk / kpk)

    key, sk1, sk2 = jax.random.split(key, 3)

    raw_noise = jax.random.normal(sk1, shape=(3, nx, ny, nz)) + \
                1j * jax.random.normal(sk2, shape=(3, nx, ny, nz))

    cwx = jnp.sqrt(Pk) * raw_noise[0]
    cwy = jnp.sqrt(Pk) * raw_noise[1]
    cwz = jnp.sqrt(Pk) * raw_noise[2]

    # Zero the DC mode so the forcing has no net momentum.
    cwx = cwx.at[0, 0, 0].set(0.0 + 0.0j)
    cwy = cwy.at[0, 0, 0].set(0.0 + 0.0j)
    cwz = cwz.at[0, 0, 0].set(0.0 + 0.0j)

    # Project out the compressible (curl-free) component to leave a solenoidal
    # field. The DC mode is guarded against the division by zero at k = 0.
    k_squared_safe = jnp.where(k_squared == 0.0, 1.0, k_squared)
    div_k = (kx_3d * cwx + ky_3d * cwy + kz_3d * cwz) / k_squared_safe
    div_k = div_k.at[0, 0, 0].set(0.0 + 0.0j)
    cwx = cwx - kx_3d * div_k
    cwy = cwy - ky_3d * div_k
    cwz = cwz - kz_3d * div_k

    # Transform back to real space.
    wx_real = jnp.real(jnp.fft.ifftn(cwx))
    wy_real = jnp.real(jnp.fft.ifftn(cwy))
    wz_real = jnp.real(jnp.fft.ifftn(cwz))

    return key, wx_real, wy_real, wz_real


# -------------------------------------------------------------
# ===== ↓ Ornstein-Uhlenbeck (temporally correlated) forcing ↓ =====
# -------------------------------------------------------------


@partial(jax.jit, static_argnames=["config"])
def _create_solenoidal_field(key, config, k_f, band=None):
    """A fresh solenoidal (divergence-free) random velocity field, unit rms.

    ``band=None`` uses the legacy smooth ``k^6 exp(-8k/kpk)`` spectrum peaked at
    ``k_f``. ``band=(nlow, nhigh, expo)`` instead reproduces AthenaK's
    ``turb_driver``: power confined to the discrete mode-number shell
    ``nlow <= n <= nhigh`` with an isotropic ``k^-(expo+2)/2`` envelope. A
    non-empty ``config.turbulent_forcing_config.forcing_modes`` overrides both
    and reproduces AthenaPK's ``few_modes_ft``: power only on the listed integer
    modes with the parabolic envelope ``(n/n_pk)^2 (2 - (n/n_pk)^2)`` peaked at
    ``n_pk = k_f L / 2pi``, including its conjugate-pairing of ``k_x = 0`` modes.
    """
    nx = config.num_cells.x
    ny = config.num_cells.y
    nz = config.num_cells.z

    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=config.box_size.x / nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=config.box_size.y / ny)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=config.box_size.z / nz)
    kx_3d = kx.reshape(nx, 1, 1)
    ky_3d = ky.reshape(1, ny, 1)
    kz_3d = kz.reshape(1, 1, nz)
    k_squared = kx_3d ** 2 + ky_3d ** 2 + kz_3d ** 2
    kk = jnp.sqrt(k_squared)

    modes = tuple(config.turbulent_forcing_config.forcing_modes)
    if modes:
        # AthenaPK ``few_modes_ft``: a fixed list of integer modes (mode number
        # n = k L / 2pi, so the array index of mode n is n mod N), each with the
        # parabolic amplitude (n/n_pk)^2 (2 - (n/n_pk)^2), clipped at zero. The
        # mode set is static, so the mask is built in numpy at trace time.
        # ``k_f`` is a traced parameter, so only the mode positions and their
        # |n| are static; the envelope itself is evaluated in jnp.
        n_pk = k_f * config.box_size.x / (2.0 * jnp.pi)
        nmag = np.zeros((nx, ny, nz))
        listed = np.zeros((nx, ny, nz), dtype=bool)
        for mx, my, mz in modes:
            nmag[mx % nx, my % ny, mz % nz] = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
            listed[mx % nx, my % ny, mz % nz] = True
        ratio2 = (jnp.asarray(nmag) / n_pk) ** 2
        amp = jnp.where(jnp.asarray(listed),
                        jnp.maximum(ratio2 * (2.0 - ratio2), 0.0), 0.0)
        Pk = amp ** 2
    elif band is None:
        # The spectrum k^6 exp(-8 k / kpk) peaks at k = 0.75 kpk, so set kpk =
        # k_f / 0.75 to place the peak at the requested forcing wavenumber k_f.
        kpk = k_f / 0.75
        Pk = kk ** 6 * jnp.exp(-8.0 * kk / kpk)
    else:
        # AthenaK ``turb_driver`` spectrum: power ONLY on the discrete mode-number
        # band nlow <= n <= nhigh (n = k L / 2pi), with the isotropic power-law
        # envelope |F(k)| ~ k^-(expo+2)/2. Sharp band edges (not a smooth
        # envelope with a high-k tail) are what AthenaK actually drives.
        nlow, nhigh, expo = band
        n_x = kx_3d * config.box_size.x / (2.0 * jnp.pi)
        n_y = ky_3d * config.box_size.y / (2.0 * jnp.pi)
        n_z = kz_3d * config.box_size.z / (2.0 * jnp.pi)
        n_sq = n_x ** 2 + n_y ** 2 + n_z ** 2
        in_band = (n_sq >= nlow ** 2 - 1e-6) & (n_sq <= nhigh ** 2 + 1e-6)
        kk_safe = jnp.where(kk > 0.0, kk, 1.0)
        amp = kk_safe ** (-(expo + 2.0) / 2.0)
        Pk = jnp.where(in_band, amp ** 2, 0.0)

    key, sk1, sk2 = jax.random.split(key, 3)
    raw = jax.random.normal(sk1, shape=(3, nx, ny, nz)) + \
        1j * jax.random.normal(sk2, shape=(3, nx, ny, nz))
    cwx = jnp.sqrt(Pk) * raw[0]
    cwy = jnp.sqrt(Pk) * raw[1]
    cwz = jnp.sqrt(Pk) * raw[2]
    cwx = cwx.at[0, 0, 0].set(0.0 + 0.0j)
    cwy = cwy.at[0, 0, 0].set(0.0 + 0.0j)
    cwz = cwz.at[0, 0, 0].set(0.0 + 0.0j)

    if modes:
        # AthenaPK's "enforce symmetry" rule: a k_x = 0 mode whose (k_y, k_z) is
        # the negative of an EARLIER listed mode gets that mode's conjugate
        # amplitude, so the pair adds coherently in the real part rather than
        # as two independent draws. Same construction here, since the field is
        # Re(ifft), which pairs cw(k) with conj(cw(-k)).
        for j, (mx, my, mz) in enumerate(modes):
            if mx != 0:
                continue
            for mx2, my2, mz2 in modes[:j]:
                if mx2 == 0 and my2 == -my and mz2 == -mz:
                    src = (0, my2 % ny, mz2 % nz)
                    dst = (0, my % ny, mz % nz)
                    cwx = cwx.at[dst].set(jnp.conj(cwx[src]))
                    cwy = cwy.at[dst].set(jnp.conj(cwy[src]))
                    cwz = cwz.at[dst].set(jnp.conj(cwz[src]))

    # Project out the compressible (curl-free) component to leave a solenoidal
    # field.
    k_squared_safe = jnp.where(k_squared == 0.0, 1.0, k_squared)
    div_k = (kx_3d * cwx + ky_3d * cwy + kz_3d * cwz) / k_squared_safe
    div_k = div_k.at[0, 0, 0].set(0.0 + 0.0j)
    cwx = cwx - kx_3d * div_k
    cwy = cwy - ky_3d * div_k
    cwz = cwz - kz_3d * div_k

    wx = jnp.real(jnp.fft.ifftn(cwx))
    wy = jnp.real(jnp.fft.ifftn(cwy))
    wz = jnp.real(jnp.fft.ifftn(cwz))

    # Normalise to unit rms (the small epsilon guards an all-zero field).
    norm = jnp.sqrt(jnp.mean(wx ** 2 + wy ** 2 + wz ** 2) + 1e-30)
    field = jnp.stack([wx, wy, wz]) / norm
    return key, field


def _uses_ghost_cells(config: SimulationConfig) -> bool:
    """Whether the state carries a ghost-cell halo around the physical grid."""
    return config.boundary_handling != PERIODIC_ROLL and config.num_ghost_cells > 0


def _on_state_grid(field, config: SimulationConfig):
    """
    Extend a forcing component from the physical grid to the state's grid.

    With ghost cells the field is continued periodically into the halo, which
    is exactly what periodic boundaries put there (for other boundaries the
    boundary handler overwrites the halo before the next update anyway).
    """
    if not _uses_ghost_cells(config):
        return field
    return jnp.pad(field, config.num_ghost_cells, mode="wrap")


def _add_velocity_kick(primitive_state, amplitude, field, config, registered_variables):
    """
    Add ``amplitude * field`` (three components on the physical grid) to the
    velocity, in the precision of the state.
    """
    velocity_index = registered_variables.velocity_index
    for index, component in zip((velocity_index.x, velocity_index.y, velocity_index.z), field):
        kick = amplitude * _on_state_grid(component, config)
        primitive_state = primitive_state.at[index].add(kick.astype(primitive_state.dtype))
    return primitive_state


def _physical_cells(field, config: SimulationConfig):
    """The physical (non-ghost) cells of a single field on the state's grid."""
    if not _uses_ghost_cells(config):
        return field
    ghosts = config.num_ghost_cells
    return field[ghosts:-ghosts, ghosts:-ghosts, ghosts:-ghosts]


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _exact_injection_amplitude(primitive_state, wx, wy, wz, dt, Edot,
                              config, registered_variables):
    """Amplitude ``a`` with ``v -> v + a w`` injecting exactly ``Edot * dt``.

    Solves ``a^2 * sum(rho|w|^2)/2 + a * sum(rho v.w) - Edot dt / dV = 0`` for
    the positive root — the normalisation AthenaK's ``turb_driver`` applies via
    its ``dedt`` parameter, shared here by the white and OU paths.
    """
    # Only the physical cells count towards the injected energy.
    rho = _physical_cells(primitive_state[registered_variables.density_index], config)
    u = _physical_cells(primitive_state[registered_variables.velocity_index.x], config)
    v = _physical_cells(primitive_state[registered_variables.velocity_index.y], config)
    w = _physical_cells(primitive_state[registered_variables.velocity_index.z], config)
    dV = config.grid_spacing ** 3
    tempa = 0.5 * jnp.sum(rho * (wx ** 2 + wy ** 2 + wz ** 2))
    tempb = jnp.sum(rho * u * wx + rho * v * wy + rho * w * wz)
    tempc = -Edot * dt / dV
    disc = tempb ** 2 - 4.0 * tempa * tempc
    return jax.lax.cond(
        (disc >= 0) & (jnp.abs(tempa) > 1e-10),
        lambda: (-tempb + jnp.sqrt(disc)) / (2.0 * tempa),
        lambda: 0.0,
    )


@partial(jax.jit, static_argnames=["config"])
def _create_solenoidal_spectrum(key, config, k_f):
    """A fresh solenoidal random forcing *spectrum* on the coarse synthesis
    grid (``nc = config.turbulent_forcing_config.synthesis_resolution``),
    Hermitian-symmetrised so its inverse DFT is real, and normalised to unit
    real-space rms via Parseval's theorem.

    Mathematically identical to the field produced by
    :func:`_create_solenoidal_field` restricted to the coarse band limit --
    the construction (power spectrum, k = 0 removal, solenoidal projection,
    unit-rms normalisation) is the same, but only ``nc^3`` arrays are ever
    touched, so the draw stays cheap and fully replicated across devices.
    """
    nc = config.turbulent_forcing_config.synthesis_resolution

    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nc, d=config.box_size.x / nc)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(nc, d=config.box_size.y / nc)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nc, d=config.box_size.z / nc)
    kx_3d = kx.reshape(nc, 1, 1)
    ky_3d = ky.reshape(1, nc, 1)
    kz_3d = kz.reshape(1, 1, nc)
    k_squared = kx_3d ** 2 + ky_3d ** 2 + kz_3d ** 2
    kk = jnp.sqrt(k_squared)

    # The spectrum k^6 exp(-8 k / kpk) peaks at k = 0.75 kpk, so set kpk =
    # k_f / 0.75 to place the peak at the requested forcing wavenumber k_f.
    kpk = k_f / 0.75
    Pk = kk ** 6 * jnp.exp(-8.0 * kk / kpk)

    key, sk1, sk2 = jax.random.split(key, 3)
    raw = jax.random.normal(sk1, shape=(3, nc, nc, nc)) + \
        1j * jax.random.normal(sk2, shape=(3, nc, nc, nc))
    cwx = jnp.sqrt(Pk) * raw[0]
    cwy = jnp.sqrt(Pk) * raw[1]
    cwz = jnp.sqrt(Pk) * raw[2]
    cwx = cwx.at[0, 0, 0].set(0.0 + 0.0j)
    cwy = cwy.at[0, 0, 0].set(0.0 + 0.0j)
    cwz = cwz.at[0, 0, 0].set(0.0 + 0.0j)

    # Project out the compressible (curl-free) component to leave a solenoidal
    # field.
    k_squared_safe = jnp.where(k_squared == 0.0, 1.0, k_squared)
    div_k = (kx_3d * cwx + ky_3d * cwy + kz_3d * cwz) / k_squared_safe
    div_k = div_k.at[0, 0, 0].set(0.0 + 0.0j)
    cwx = cwx - kx_3d * div_k
    cwy = cwy - ky_3d * div_k
    cwz = cwz - kz_3d * div_k
    spectrum = jnp.stack([cwx, cwy, cwz])

    # Hermitian-symmetrise, h(k) = (c(k) + conj(c(-k))) / 2, so that the
    # inverse DFT is exactly real.  This equals taking jnp.real(ifftn(c)), the
    # operation the full-grid path performs.  The (-k) index map in fft order
    # is a flip followed by a one-slot roll along each spatial axis.
    def _reflect(c):
        for axis in (1, 2, 3):
            c = jnp.roll(jnp.flip(c, axis=axis), shift=1, axis=axis)
        return c

    spectrum = 0.5 * (spectrum + jnp.conj(_reflect(spectrum)))

    # Normalise to unit real-space rms.  By Parseval (with the 1/nc^3 inverse
    # DFT convention), mean_x |w|^2 summed over components = sum_k |h_k|^2 /
    # nc^6; the small epsilon guards an all-zero draw.
    norm = jnp.sqrt(jnp.sum(jnp.abs(spectrum) ** 2) / float(nc) ** 6 + 1e-30)
    return key, spectrum / norm


@partial(jax.jit, static_argnames=["config"])
def _synthesize_forcing_field(spectrum, config):
    """Evaluate the coarse solenoidal forcing spectrum on the simulation grid.

    The field is band-limited by construction, so evaluating its Fourier
    series on the fine grid is *exact* -- no interpolation error.  It is done
    as three per-axis inverse-DFT matrix products (einsums), which shard
    cleanly under GSPMD: the large output axes follow the primitive state's
    sharding, so each device only ever materialises its own shard.
    """
    nc = config.turbulent_forcing_config.synthesis_resolution

    # Like the full-grid draw, the field lives on the physical grid and is
    # continued into a ghost-cell halo by ``_on_state_grid`` when applied.
    nx = config.num_cells.x
    ny = config.num_cells.y
    nz = config.num_cells.z

    # Integer mode numbers in fft order, shared by all axes of the coarse grid.
    modes = jnp.fft.fftfreq(nc) * nc

    def _dft_matrix(n_fine):
        # Fine-grid sample positions as box fractions i / n_fine, matching the
        # implicit sampling of jnp.fft.ifftn on the coarse grid.
        positions = jnp.arange(n_fine) / n_fine
        return jnp.exp(2j * jnp.pi * positions[:, None] * modes[None, :])

    E_x = _dft_matrix(nx)
    E_y = _dft_matrix(ny)
    E_z = _dft_matrix(nz)

    def _synthesize_component(coeffs):
        w = jnp.einsum("xa,abc->xbc", E_x, coeffs)
        w = jnp.einsum("yb,xbc->xyc", E_y, w)
        w = jnp.einsum("zc,xyc->xyz", E_z, w)
        # 1/nc^3: the inverse-DFT normalisation of the coarse grid.
        return jnp.real(w) / float(nc) ** 3

    return jnp.stack([
        _synthesize_component(spectrum[0]),
        _synthesize_component(spectrum[1]),
        _synthesize_component(spectrum[2]),
    ])


def _ou_driving_band(config, turbulent_forcing_params):
    """
    The discrete AthenaK driving band ``(nlow, nhigh, expo)`` if the banded
    spectrum is selected, otherwise ``None`` (the smooth peaked spectrum).
    """
    if not config.turbulent_forcing_config.banded_spectrum:
        return None
    return (
        turbulent_forcing_params.forcing_nlow,
        turbulent_forcing_params.forcing_nhigh,
        turbulent_forcing_params.forcing_expo,
    )


def _draw_solenoidal_increment(key, config, turbulent_forcing_params):
    """
    Draw a fresh unit-rms solenoidal forcing realisation, either as a field on
    the simulation grid or, with coarse spectral synthesis, as the coarse
    spectrum (only the smooth peaked spectrum is available there).
    """
    band = _ou_driving_band(config, turbulent_forcing_params)
    if config.turbulent_forcing_config.synthesis_resolution > 0:
        if band is not None or config.turbulent_forcing_config.forcing_modes:
            raise ValueError(
                "Coarse spectral synthesis (synthesis_resolution > 0) only "
                "supports the smooth forcing spectrum, not banded_spectrum "
                "or forcing_modes."
            )
        return _create_solenoidal_spectrum(
            key,
            config,
            turbulent_forcing_params.forcing_wavenumber,
        )
    return _create_solenoidal_field(
        key,
        config,
        turbulent_forcing_params.forcing_wavenumber,
        band=band,
    )


@partial(jax.jit, static_argnames=["config"])
def _init_ou_forcing_state(key, config, turbulent_forcing_params):
    """Initial OU forcing state ``(key, f0)`` with f0 a stationary draw.

    With coarse spectral synthesis enabled, f0 is the coarse *spectrum* (the
    OU update is linear, so evolving the spectrum and synthesising the field
    each step is mathematically identical to evolving the real-space field);
    otherwise it is the real-space field on the simulation grid.
    """
    return _draw_solenoidal_increment(key, config, turbulent_forcing_params)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _apply_ou_forcing(
    forcing_state,
    primitive_state,
    dt,
    turbulent_forcing_params,
    config,
    registered_variables,
):
    """Apply Ornstein-Uhlenbeck forcing.

    The persistent forcing field ``f`` (carried in ``forcing_state``) is evolved
    with the exact OU discretisation ``f <- a f + sqrt(1 - a^2) xi``,
    ``a = exp(-dt / tau_f)``, keeping it at unit rms, then applied as a
    constant-amplitude acceleration ``velocity += F0 f dt`` (state-independent,
    so the adjoint is clean and the realisation is reproducible for a fixed
    timestep sequence), or with the amplitude that injects exactly
    ``energy_injection_rate * dt`` if ``ou_exact_injection`` is set.
    """
    key, f = forcing_state
    tau_f = turbulent_forcing_params.correlation_time
    a = jnp.exp(-dt / tau_f)
    key, xi = _draw_solenoidal_increment(key, config, turbulent_forcing_params)
    f = a * f + jnp.sqrt(jnp.maximum(1.0 - a ** 2, 0.0)) * xi

    # With coarse spectral synthesis the persistent state is the spectrum; the
    # real-space acceleration is synthesised only for this step's application.
    if config.turbulent_forcing_config.synthesis_resolution > 0:
        field = _synthesize_forcing_field(f, config)
    else:
        field = f

    if config.turbulent_forcing_config.ou_unit_rms_each_step:
        # AthenaPK rescales the real-space acceleration to ``accel_rms`` every
        # cycle; the persistent spectral field is not renormalised, only the
        # applied copy. Same here: ``g`` is what is applied, ``f`` what persists.
        rms = jnp.sqrt(jnp.mean(field[0] ** 2 + field[1] ** 2 + field[2] ** 2) + 1e-30)
        g = field / rms
    else:
        g = field

    if config.turbulent_forcing_config.ou_exact_injection:
        # AthenaK ``dedt`` normalisation: scale the (unit-rms) OU field so the
        # box gains exactly Edot*dt of kinetic energy this step.
        amp = _exact_injection_amplitude(
            primitive_state, g[0], g[1], g[2], dt,
            turbulent_forcing_params.energy_injection_rate,
            config, registered_variables,
        )
    else:
        amp = turbulent_forcing_params.forcing_amplitude * dt
    primitive_state = _add_velocity_kick(primitive_state, amp, g, config, registered_variables)

    return (key, f), primitive_state


# -------------------------------------------------------------
# ===== ↑ Ornstein-Uhlenbeck (temporally correlated) forcing ↑ =====
# -------------------------------------------------------------


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _apply_forcing(
    key,
    primitive_state,
    dt,
    turbulent_forcing_params: TurbulentForcingParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Apply white-in-time turbulent forcing at a fixed energy injection rate.

    Draws a fresh solenoidal field and solves a quadratic for the forcing
    amplitude that injects exactly the configured energy per step, then adds the
    scaled field to the velocity.

    Args:
        key: The PRNG key.
        primitive_state: The primitive state array.
        dt: The time step.
        turbulent_forcing_params: The turbulent-forcing parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        ``(key, primitive_state)``: the advanced PRNG key and the forced state.
    """

    key, wx_real, wy_real, wz_real = _create_forcing_field(key, config)

    # Normalise the drawn field to unit rms before the amplitude solve. The
    # raw spectrum k^6 exp(-8k/kpk) is dimensional (kpk = 4 pi / L), so the
    # field amplitude scales as L^-3 — in float32 a large box (e.g. 64 pc)
    # underflows the quadratic's coefficients and the forcing silently turns
    # off. The energy-injection quadratic rescales the amplitude exactly, so
    # this is statistically a no-op at any box size.
    w_rms = jnp.sqrt(jnp.mean(wx_real**2 + wy_real**2 + wz_real**2) / 3.0)
    w_rms = jnp.maximum(w_rms, 1e-30)
    wx_real = wx_real / w_rms
    wy_real = wy_real / w_rms
    wz_real = wz_real / w_rms

    Edot = turbulent_forcing_params.energy_injection_rate
    dtforc = dt
    dV = config.grid_spacing**3

    # Density and velocity components of the physical cells (the injected
    # energy is measured there; ghost cells only mirror them).
    rho = _physical_cells(primitive_state[registered_variables.density_index], config)
    u = _physical_cells(primitive_state[registered_variables.velocity_index.x], config)
    v = _physical_cells(primitive_state[registered_variables.velocity_index.y], config)
    w = _physical_cells(primitive_state[registered_variables.velocity_index.z], config)

    # Solve the quadratic a * amp^2 + b * amp + c = 0 for the forcing amplitude
    # that injects the prescribed energy Edot * dt over the box.
    tempa = 0.5 * jnp.sum(rho * (wx_real**2 + wy_real**2 + wz_real**2))
    tempb = jnp.sum(rho * u * wx_real + rho * v * wy_real + rho * w * wz_real)
    tempc = -Edot * dtforc / dV

    discriminant = tempb**2 - 4.0 * tempa * tempc

    # Guard against a negative discriminant or a vanishing quadratic
    # coefficient: in those degenerate cases apply no forcing this step.
    amp = jax.lax.cond(
        (discriminant >= 0) & (jnp.abs(tempa) > 1e-10),
        lambda: (-tempb + jnp.sqrt(discriminant)) / (2.0 * tempa),
        lambda: 0.0
    )

    # Add the scaled forcing field directly to the velocity components.
    primitive_state = _add_velocity_kick(
        primitive_state, amp, (wx_real, wy_real, wz_real), config, registered_variables
    )

    return key, primitive_state
