"""
Turbulent forcing of the velocity field.

Provides two driving schemes. The default white-in-time forcing draws a fresh
solenoidal field each step and rescales it so that a prescribed energy
injection rate is met; the Ornstein-Uhlenbeck variant carries a temporally
correlated solenoidal field across steps and applies it as a constant-amplitude
acceleration (or with the amplitude of an exact energy injection). Besides the
smooth peaked spectrum, the OU forcing reproduces AthenaK's discrete driving
band and AthenaPK's few-modes driver, and it can be synthesised from a coarse
spectral grid for large sharded runs. The fields live on the physical grid and
are continued periodically into a ghost-cell halo. The construction of the
solenoidal forcing fields follows https://arxiv.org/pdf/2304.04360.
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix.option_classes.simulation_config import PERIODIC_ROLL

# astronomix containers
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingParams,
)
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables


# -------------------------------------------------------------
# ============ ↓ Applying a forcing field to the state ↓ ============
# -------------------------------------------------------------


def _uses_ghost_cells(config: SimulationConfig) -> bool:
    """Whether the state carries a ghost-cell halo around the physical grid."""
    return config.boundary_handling != PERIODIC_ROLL and config.num_ghost_cells > 0


def _extend_periodically_to_state_grid(scalar_field, config: SimulationConfig):
    """
    Extend a forcing component from the physical grid to the state's grid.

    With ghost cells the field is continued periodically into the halo, which
    is exactly what periodic boundaries put there (for other boundaries the
    boundary handler overwrites the halo before the next update anyway).

    Args:
        scalar_field: One forcing component on the physical grid.
        config: The simulation configuration.

    Returns:
        The component on the state's grid (unchanged without ghost cells).
    """
    if not _uses_ghost_cells(config):
        return scalar_field
    return jnp.pad(scalar_field, config.num_ghost_cells, mode="wrap")


def _add_velocity_kick(
    primitive_state,
    amplitude,
    forcing_components,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Add ``amplitude * forcing_components`` to the velocity, in the precision
    of the state.

    Args:
        primitive_state: The primitive state on the state's grid.
        amplitude: The scalar kick amplitude (the velocity change per unit of
            the forcing field).
        forcing_components: The three forcing components on the physical grid.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The primitive state with the kicked velocity.
    """
    velocity_index = registered_variables.velocity_index
    velocity_indices = (velocity_index.x, velocity_index.y, velocity_index.z)
    for index, component in zip(velocity_indices, forcing_components):
        kick = amplitude * _extend_periodically_to_state_grid(component, config)
        primitive_state = primitive_state.at[index].add(kick.astype(primitive_state.dtype))
    return primitive_state


def _physical_cells(scalar_field, config: SimulationConfig):
    """
    The physical (non-ghost) cells of a single 3D field on the state's grid
    (the forcing is 3D only, so the halo is stripped along all three axes).
    """
    if not _uses_ghost_cells(config):
        return scalar_field
    ghosts = config.num_ghost_cells
    return scalar_field[ghosts:-ghosts, ghosts:-ghosts, ghosts:-ghosts]


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _exact_injection_amplitude(
    primitive_state,
    forcing_x,
    forcing_y,
    forcing_z,
    dt,
    energy_injection_rate,
    config,
    registered_variables,
):
    """
    The amplitude ``A`` for which the kick ``v -> v + A w`` injects exactly
    ``energy_injection_rate * dt`` of kinetic energy into the box.

    Solves ``A^2 sum(rho |w|^2) / 2 + A sum(rho v.w) - Edot dt / dV = 0`` for
    the positive root, the normalisation AthenaK's ``turb_driver`` applies via
    its ``dedt`` parameter. Shared by the white forcing and the OU forcing
    with ``ou_exact_injection``.

    Args:
        primitive_state: The primitive state on the state's grid.
        forcing_x: The x component of the forcing field on the physical grid.
        forcing_y: The y component of the forcing field on the physical grid.
        forcing_z: The z component of the forcing field on the physical grid.
        dt: The time step.
        energy_injection_rate: The kinetic energy to inject per unit time.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The amplitude, or zero when the quadratic has no real root or its
        leading coefficient vanishes (no forcing is applied this step).
    """
    # Only the physical cells count towards the injected energy; the ghost
    # cells only mirror them.
    density = _physical_cells(primitive_state[registered_variables.density_index], config)
    velocity_x = _physical_cells(primitive_state[registered_variables.velocity_index.x], config)
    velocity_y = _physical_cells(primitive_state[registered_variables.velocity_index.y], config)
    velocity_z = _physical_cells(primitive_state[registered_variables.velocity_index.z], config)
    cell_volume = config.grid_spacing ** 3

    forcing_squared = forcing_x ** 2 + forcing_y ** 2 + forcing_z ** 2
    quadratic_coefficient = 0.5 * jnp.sum(density * forcing_squared)
    linear_coefficient = jnp.sum(
        density * velocity_x * forcing_x
        + density * velocity_y * forcing_y
        + density * velocity_z * forcing_z
    )
    constant_term = -energy_injection_rate * dt / cell_volume
    discriminant = linear_coefficient ** 2 - 4.0 * quadratic_coefficient * constant_term

    # Guard against a negative discriminant or a vanishing quadratic
    # coefficient: in those degenerate cases apply no forcing this step.
    return jax.lax.cond(
        (discriminant >= 0) & (jnp.abs(quadratic_coefficient) > 1e-10),
        lambda: (-linear_coefficient + jnp.sqrt(discriminant)) / (2.0 * quadratic_coefficient),
        lambda: 0.0,
    )


# -------------------------------------------------------------
# ============ ↑ Applying a forcing field to the state ↑ ============
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ White-in-time forcing ↓ ===================
# -------------------------------------------------------------


@partial(jax.jit, static_argnames=["config"])
def _create_forcing_field(
    key,
    config: SimulationConfig,
):
    """
    Draw a fresh solenoidal (divergence-free) random forcing field.

    Builds a random field in Fourier space with the power spectrum
    k^6 exp(-8 k / kpk), kpk = 4 pi / L, removes the compressible component to
    make it solenoidal, and transforms it back to real space.

    Args:
        key: The PRNG key.
        config: The simulation configuration.

    Returns:
        ``(key, wx_real, wy_real, wz_real)``: the advanced PRNG key and the three
        real-space components of the solenoidal forcing field on the physical
        grid (no ghost halo), each of shape
        ``(num_cells.x, num_cells.y, num_cells.z)``.
    """

    box_size_x = config.box_size.x
    box_size_y = config.box_size.y
    box_size_z = config.box_size.z

    # The field lives on the physical grid; with ghost cells it is extended
    # periodically by ``_extend_periodically_to_state_grid`` when it is applied.
    nx = config.num_cells.x
    ny = config.num_cells.y
    nz = config.num_cells.z

    # Wavenumbers along each axis via fftfreq.
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=box_size_x / nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=box_size_y / ny)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=box_size_z / nz)

    # Broadcast the 1D wavenumber arrays into 3D rather than materialising a
    # full meshgrid.
    kx_3d = kx.reshape(nx, 1, 1)
    ky_3d = ky.reshape(1, ny, 1)
    kz_3d = kz.reshape(1, 1, nz)

    k_squared = kx_3d**2 + ky_3d**2 + kz_3d**2
    wavenumber_magnitude = jnp.sqrt(k_squared)

    # Forcing power spectrum peaked at intermediate wavenumbers.
    kpk = 4.0 * jnp.pi / config.box_size.x
    power_spectrum = wavenumber_magnitude**6 * jnp.exp(-8.0 * wavenumber_magnitude / kpk)

    key, real_part_key, imaginary_part_key = jax.random.split(key, 3)

    complex_noise = (
        jax.random.normal(real_part_key, shape=(3, nx, ny, nz))
        + 1j * jax.random.normal(imaginary_part_key, shape=(3, nx, ny, nz))
    )

    spectrum_x = jnp.sqrt(power_spectrum) * complex_noise[0]
    spectrum_y = jnp.sqrt(power_spectrum) * complex_noise[1]
    spectrum_z = jnp.sqrt(power_spectrum) * complex_noise[2]

    # Zero the DC mode so the forcing has no net momentum.
    spectrum_x = spectrum_x.at[0, 0, 0].set(0.0 + 0.0j)
    spectrum_y = spectrum_y.at[0, 0, 0].set(0.0 + 0.0j)
    spectrum_z = spectrum_z.at[0, 0, 0].set(0.0 + 0.0j)

    # Project out the compressible (curl-free) component, (k . c) k / k^2, to
    # leave a solenoidal field. The DC mode is guarded against the division
    # by zero at k = 0.
    k_squared_safe = jnp.where(k_squared == 0.0, 1.0, k_squared)
    k_dot_spectrum = kx_3d * spectrum_x + ky_3d * spectrum_y + kz_3d * spectrum_z
    longitudinal_coefficient = k_dot_spectrum / k_squared_safe
    longitudinal_coefficient = longitudinal_coefficient.at[0, 0, 0].set(0.0 + 0.0j)
    spectrum_x = spectrum_x - kx_3d * longitudinal_coefficient
    spectrum_y = spectrum_y - ky_3d * longitudinal_coefficient
    spectrum_z = spectrum_z - kz_3d * longitudinal_coefficient

    # Transform back to real space.
    wx_real = jnp.real(jnp.fft.ifftn(spectrum_x))
    wy_real = jnp.real(jnp.fft.ifftn(spectrum_y))
    wz_real = jnp.real(jnp.fft.ifftn(spectrum_z))

    return key, wx_real, wy_real, wz_real


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _apply_forcing(
    key,
    primitive_state,
    dt,
    turbulent_forcing_params: TurbulentForcingParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Apply white-in-time turbulent forcing at a fixed energy injection rate.

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

    # Normalise the drawn field to unit rms per component before the amplitude
    # solve. The raw spectrum k^6 exp(-8k/kpk) is dimensional (kpk = 4 pi / L),
    # so the field amplitude scales as L^-3 — in float32 a large box (e.g.
    # 64 pc) underflows the quadratic's coefficients and the forcing silently
    # turns off. The energy-injection quadratic rescales the amplitude exactly,
    # so this is statistically a no-op at any box size.
    w_rms = jnp.sqrt(jnp.mean(wx_real**2 + wy_real**2 + wz_real**2) / 3.0)
    w_rms = jnp.maximum(w_rms, 1e-30)
    wx_real = wx_real / w_rms
    wy_real = wy_real / w_rms
    wz_real = wz_real / w_rms

    amplitude = _exact_injection_amplitude(
        primitive_state,
        wx_real,
        wy_real,
        wz_real,
        dt,
        turbulent_forcing_params.energy_injection_rate,
        config,
        registered_variables,
    )

    # Add the scaled forcing field directly to the velocity components.
    primitive_state = _add_velocity_kick(
        primitive_state,
        amplitude,
        (wx_real, wy_real, wz_real),
        config,
        registered_variables,
    )

    return key, primitive_state


# -------------------------------------------------------------
# =============== ↑ White-in-time forcing ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===== ↓ Ornstein-Uhlenbeck (temporally correlated) forcing ↓ =====
# -------------------------------------------------------------


@partial(jax.jit, static_argnames=["config"])
def _create_solenoidal_field(key, config, k_f, band=None):
    """
    A fresh solenoidal (divergence-free) random velocity field of unit rms.

    ``band=None`` uses the smooth ``k^6 exp(-8k/kpk)`` spectrum peaked at
    ``k_f``. ``band=(lowest mode, highest mode, exponent)`` (AthenaK's
    ``nlow``, ``nhigh``, ``expo``) instead reproduces AthenaK's
    ``turb_driver``: power confined to the discrete mode-number shell
    ``nlow <= n <= nhigh`` with an isotropic ``k^-(expo+2)/2`` envelope. A
    non-empty ``config.turbulent_forcing_config.forcing_modes`` overrides both
    and reproduces AthenaPK's ``few_modes_ft``: power only on the listed integer
    modes with the parabolic envelope ``(n/n_pk)^2 (2 - (n/n_pk)^2)`` peaked at
    ``n_pk = k_f L / 2pi``, including its conjugate-pairing of ``k_x = 0`` modes.

    Args:
        key: The PRNG key.
        config: The simulation configuration.
        k_f: The forcing wavenumber (peak of the smooth spectrum, envelope peak
            of the few-modes driver; ignored by the banded spectrum).
        band: ``None`` or the discrete driving band
            ``(lowest_mode_number, highest_mode_number, spectral_exponent)``.

    Returns:
        ``(key, field)``: the advanced PRNG key and the unit-rms field of shape
        ``(3, num_cells.x, num_cells.y, num_cells.z)`` on the physical grid.
    """
    nx = config.num_cells.x
    ny = config.num_cells.y
    nz = config.num_cells.z

    # --------------- ↓ Wavenumber grid ↓ ----------------
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=config.box_size.x / nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=config.box_size.y / ny)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=config.box_size.z / nz)
    kx_3d = kx.reshape(nx, 1, 1)
    ky_3d = ky.reshape(1, ny, 1)
    kz_3d = kz.reshape(1, 1, nz)
    k_squared = kx_3d ** 2 + ky_3d ** 2 + kz_3d ** 2
    wavenumber_magnitude = jnp.sqrt(k_squared)
    # --------------- ↑ Wavenumber grid ↑ ----------------

    # --------------- ↓ Forcing power spectrum ↓ ----------------
    modes = tuple(config.turbulent_forcing_config.forcing_modes)
    if modes:
        # AthenaPK ``few_modes_ft``: a fixed list of integer modes (mode number
        # n = k L / 2pi, so the array index of mode n is n mod N), each with the
        # parabolic amplitude (n/n_pk)^2 (2 - (n/n_pk)^2), clipped at zero. The
        # mode set is static, so the mask is built in numpy at trace time.
        # ``k_f`` is a traced parameter, so only the mode positions and their
        # |n| are static; the envelope itself is evaluated in jnp.
        n_pk = k_f * config.box_size.x / (2.0 * jnp.pi)
        mode_number_magnitude = np.zeros((nx, ny, nz))
        is_listed_mode = np.zeros((nx, ny, nz), dtype=bool)
        for mode_x, mode_y, mode_z in modes:
            grid_index = (mode_x % nx, mode_y % ny, mode_z % nz)
            mode_number_magnitude[grid_index] = np.sqrt(mode_x ** 2 + mode_y ** 2 + mode_z ** 2)
            is_listed_mode[grid_index] = True
        squared_mode_ratio = (jnp.asarray(mode_number_magnitude) / n_pk) ** 2
        mode_amplitude = jnp.where(
            jnp.asarray(is_listed_mode),
            jnp.maximum(squared_mode_ratio * (2.0 - squared_mode_ratio), 0.0),
            0.0,
        )
        power_spectrum = mode_amplitude ** 2
    elif band is None:
        # The spectrum k^6 exp(-8 k / kpk) peaks at k = 0.75 kpk, so set kpk =
        # k_f / 0.75 to place the peak at the requested forcing wavenumber k_f.
        kpk = k_f / 0.75
        power_spectrum = wavenumber_magnitude ** 6 * jnp.exp(-8.0 * wavenumber_magnitude / kpk)
    else:
        # AthenaK ``turb_driver`` spectrum: power ONLY on the discrete mode-number
        # band nlow <= n <= nhigh (n = k L / 2pi), with the isotropic power-law
        # envelope |F(k)| ~ k^-(expo+2)/2. Sharp band edges (not a smooth
        # envelope with a high-k tail) are what AthenaK actually drives.
        lowest_mode_number, highest_mode_number, spectral_exponent = band
        mode_number_x = kx_3d * config.box_size.x / (2.0 * jnp.pi)
        mode_number_y = ky_3d * config.box_size.y / (2.0 * jnp.pi)
        mode_number_z = kz_3d * config.box_size.z / (2.0 * jnp.pi)
        mode_number_squared = mode_number_x ** 2 + mode_number_y ** 2 + mode_number_z ** 2
        # The squared mode numbers are integers up to the float round-off of
        # (k L / 2pi)^2; the tolerance keeps the shells on the band edges in.
        in_band = (mode_number_squared >= lowest_mode_number ** 2 - 1e-6) & (
            mode_number_squared <= highest_mode_number ** 2 + 1e-6
        )
        safe_wavenumber = jnp.where(wavenumber_magnitude > 0.0, wavenumber_magnitude, 1.0)
        mode_amplitude = safe_wavenumber ** (-(spectral_exponent + 2.0) / 2.0)
        power_spectrum = jnp.where(in_band, mode_amplitude ** 2, 0.0)
    # --------------- ↑ Forcing power spectrum ↑ ----------------

    # --------------- ↓ Random draw ↓ ----------------
    key, real_part_key, imaginary_part_key = jax.random.split(key, 3)
    complex_noise = (
        jax.random.normal(real_part_key, shape=(3, nx, ny, nz))
        + 1j * jax.random.normal(imaginary_part_key, shape=(3, nx, ny, nz))
    )
    spectrum_x = jnp.sqrt(power_spectrum) * complex_noise[0]
    spectrum_y = jnp.sqrt(power_spectrum) * complex_noise[1]
    spectrum_z = jnp.sqrt(power_spectrum) * complex_noise[2]
    spectrum_x = spectrum_x.at[0, 0, 0].set(0.0 + 0.0j)
    spectrum_y = spectrum_y.at[0, 0, 0].set(0.0 + 0.0j)
    spectrum_z = spectrum_z.at[0, 0, 0].set(0.0 + 0.0j)

    if modes:
        # AthenaPK's "enforce symmetry" rule: a k_x = 0 mode whose (k_y, k_z) is
        # the negative of an EARLIER listed mode gets that mode's conjugate
        # amplitude, so the pair adds coherently in the real part rather than
        # as two independent draws. Same construction here, since the field is
        # Re(ifft), which pairs cw(k) with conj(cw(-k)).
        for mode_position, (mode_x, mode_y, mode_z) in enumerate(modes):
            if mode_x != 0:
                continue
            for earlier_x, earlier_y, earlier_z in modes[:mode_position]:
                if earlier_x == 0 and earlier_y == -mode_y and earlier_z == -mode_z:
                    source_index = (0, earlier_y % ny, earlier_z % nz)
                    target_index = (0, mode_y % ny, mode_z % nz)
                    spectrum_x = spectrum_x.at[target_index].set(jnp.conj(spectrum_x[source_index]))
                    spectrum_y = spectrum_y.at[target_index].set(jnp.conj(spectrum_y[source_index]))
                    spectrum_z = spectrum_z.at[target_index].set(jnp.conj(spectrum_z[source_index]))
    # --------------- ↑ Random draw ↑ ----------------

    # --------------- ↓ Solenoidal projection and normalisation ↓ ----------------
    # Project out the compressible (curl-free) component, (k . c) k / k^2, to
    # leave a solenoidal field.
    k_squared_safe = jnp.where(k_squared == 0.0, 1.0, k_squared)
    k_dot_spectrum = kx_3d * spectrum_x + ky_3d * spectrum_y + kz_3d * spectrum_z
    longitudinal_coefficient = k_dot_spectrum / k_squared_safe
    longitudinal_coefficient = longitudinal_coefficient.at[0, 0, 0].set(0.0 + 0.0j)
    spectrum_x = spectrum_x - kx_3d * longitudinal_coefficient
    spectrum_y = spectrum_y - ky_3d * longitudinal_coefficient
    spectrum_z = spectrum_z - kz_3d * longitudinal_coefficient

    field_x = jnp.real(jnp.fft.ifftn(spectrum_x))
    field_y = jnp.real(jnp.fft.ifftn(spectrum_y))
    field_z = jnp.real(jnp.fft.ifftn(spectrum_z))

    # Normalise to unit rms of |w| (the small epsilon guards an all-zero field).
    norm = jnp.sqrt(jnp.mean(field_x ** 2 + field_y ** 2 + field_z ** 2) + 1e-30)
    field = jnp.stack([field_x, field_y, field_z]) / norm
    # --------------- ↑ Solenoidal projection and normalisation ↑ ----------------
    return key, field


@partial(jax.jit, static_argnames=["config"])
def _create_solenoidal_spectrum(key, config, k_f):
    """
    A fresh solenoidal random forcing *spectrum* on the coarse synthesis grid
    (``synthesis_resolution^3``), Hermitian-symmetrised so its inverse DFT is
    real, and normalised to unit real-space rms via Parseval's theorem.

    Mathematically identical to the field produced by
    :func:`_create_solenoidal_field` restricted to the coarse band limit --
    the construction (power spectrum, k = 0 removal, solenoidal projection,
    unit-rms normalisation) is the same, but only ``nc^3`` arrays are ever
    touched, so the draw stays cheap and fully replicated across devices.

    Args:
        key: The PRNG key.
        config: The simulation configuration.
        k_f: The forcing wavenumber (peak of the smooth spectrum).

    Returns:
        ``(key, spectrum)``: the advanced PRNG key and the complex spectrum of
        shape ``(3, nc, nc, nc)`` in fft order.
    """
    coarse_resolution = config.turbulent_forcing_config.synthesis_resolution

    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(coarse_resolution, d=config.box_size.x / coarse_resolution)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(coarse_resolution, d=config.box_size.y / coarse_resolution)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(coarse_resolution, d=config.box_size.z / coarse_resolution)
    kx_3d = kx.reshape(coarse_resolution, 1, 1)
    ky_3d = ky.reshape(1, coarse_resolution, 1)
    kz_3d = kz.reshape(1, 1, coarse_resolution)
    k_squared = kx_3d ** 2 + ky_3d ** 2 + kz_3d ** 2
    wavenumber_magnitude = jnp.sqrt(k_squared)

    # The spectrum k^6 exp(-8 k / kpk) peaks at k = 0.75 kpk, so set kpk =
    # k_f / 0.75 to place the peak at the requested forcing wavenumber k_f.
    kpk = k_f / 0.75
    power_spectrum = wavenumber_magnitude ** 6 * jnp.exp(-8.0 * wavenumber_magnitude / kpk)

    coarse_shape = (3, coarse_resolution, coarse_resolution, coarse_resolution)
    key, real_part_key, imaginary_part_key = jax.random.split(key, 3)
    complex_noise = (
        jax.random.normal(real_part_key, shape=coarse_shape)
        + 1j * jax.random.normal(imaginary_part_key, shape=coarse_shape)
    )
    spectrum_x = jnp.sqrt(power_spectrum) * complex_noise[0]
    spectrum_y = jnp.sqrt(power_spectrum) * complex_noise[1]
    spectrum_z = jnp.sqrt(power_spectrum) * complex_noise[2]
    spectrum_x = spectrum_x.at[0, 0, 0].set(0.0 + 0.0j)
    spectrum_y = spectrum_y.at[0, 0, 0].set(0.0 + 0.0j)
    spectrum_z = spectrum_z.at[0, 0, 0].set(0.0 + 0.0j)

    # Project out the compressible (curl-free) component, (k . c) k / k^2, to
    # leave a solenoidal field.
    k_squared_safe = jnp.where(k_squared == 0.0, 1.0, k_squared)
    k_dot_spectrum = kx_3d * spectrum_x + ky_3d * spectrum_y + kz_3d * spectrum_z
    longitudinal_coefficient = k_dot_spectrum / k_squared_safe
    longitudinal_coefficient = longitudinal_coefficient.at[0, 0, 0].set(0.0 + 0.0j)
    spectrum_x = spectrum_x - kx_3d * longitudinal_coefficient
    spectrum_y = spectrum_y - ky_3d * longitudinal_coefficient
    spectrum_z = spectrum_z - kz_3d * longitudinal_coefficient
    spectrum = jnp.stack([spectrum_x, spectrum_y, spectrum_z])

    # Hermitian-symmetrise, h(k) = (c(k) + conj(c(-k))) / 2, so that the
    # inverse DFT is exactly real. This equals taking jnp.real(ifftn(c)), the
    # operation the full-grid path performs. The (-k) index map in fft order
    # is a flip followed by a one-slot roll along each spatial axis.
    def negated_wavenumbers(coefficients):
        for axis in (1, 2, 3):
            coefficients = jnp.roll(jnp.flip(coefficients, axis=axis), shift=1, axis=axis)
        return coefficients

    spectrum = 0.5 * (spectrum + jnp.conj(negated_wavenumbers(spectrum)))

    # Normalise to unit real-space rms. By Parseval (with the 1/nc^3 inverse
    # DFT convention), mean_x |w|^2 summed over components = sum_k |h_k|^2 /
    # nc^6; the small epsilon guards an all-zero draw.
    norm = jnp.sqrt(jnp.sum(jnp.abs(spectrum) ** 2) / float(coarse_resolution) ** 6 + 1e-30)
    return key, spectrum / norm


@partial(jax.jit, static_argnames=["config"])
def _synthesize_forcing_field(spectrum, config):
    """
    Evaluate the coarse solenoidal forcing spectrum on the simulation grid.

    The field is band-limited by construction, so evaluating its Fourier
    series on the fine grid is *exact* -- no interpolation error. It is done
    as three per-axis inverse-DFT matrix products (einsums), which shard
    cleanly under GSPMD: the large output axes follow the primitive state's
    sharding, so each device only ever materialises its own shard.

    Args:
        spectrum: The coarse spectrum from :func:`_create_solenoidal_spectrum`.
        config: The simulation configuration.

    Returns:
        The real field of shape ``(3, num_cells.x, num_cells.y, num_cells.z)``
        on the physical grid.
    """
    coarse_resolution = config.turbulent_forcing_config.synthesis_resolution

    # Like the full-grid draw, the field lives on the physical grid and is
    # continued into a ghost-cell halo by ``_extend_periodically_to_state_grid``
    # when applied.
    nx = config.num_cells.x
    ny = config.num_cells.y
    nz = config.num_cells.z

    # Integer mode numbers in fft order, shared by all axes of the coarse grid.
    mode_numbers = jnp.fft.fftfreq(coarse_resolution) * coarse_resolution

    def inverse_dft_matrix(num_fine_cells):
        # Fine-grid sample positions as box fractions i / num_fine_cells,
        # matching the implicit sampling of jnp.fft.ifftn on the coarse grid.
        positions = jnp.arange(num_fine_cells) / num_fine_cells
        return jnp.exp(2j * jnp.pi * positions[:, None] * mode_numbers[None, :])

    inverse_dft_x = inverse_dft_matrix(nx)
    inverse_dft_y = inverse_dft_matrix(ny)
    inverse_dft_z = inverse_dft_matrix(nz)

    def synthesize_component(coefficients):
        component = jnp.einsum("xa,abc->xbc", inverse_dft_x, coefficients)
        component = jnp.einsum("yb,xbc->xyc", inverse_dft_y, component)
        component = jnp.einsum("zc,xyc->xyz", inverse_dft_z, component)
        # 1/nc^3: the inverse-DFT normalisation of the coarse grid.
        return jnp.real(component) / float(coarse_resolution) ** 3

    return jnp.stack([
        synthesize_component(spectrum[0]),
        synthesize_component(spectrum[1]),
        synthesize_component(spectrum[2]),
    ])


def _ou_driving_band(config, turbulent_forcing_params):
    """
    The discrete AthenaK driving band ``(forcing_nlow, forcing_nhigh,
    forcing_expo)`` if the banded spectrum is selected, otherwise ``None``
    (the smooth peaked spectrum).
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

    Args:
        key: The PRNG key.
        config: The simulation configuration.
        turbulent_forcing_params: The turbulent-forcing parameters.

    Returns:
        ``(key, draw)``: the advanced PRNG key and the field (or, with
        ``synthesis_resolution > 0``, the coarse spectrum).
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
    """
    Initial OU forcing state ``(key, f0)`` with f0 a stationary draw.

    With coarse spectral synthesis enabled, f0 is the coarse *spectrum* (the
    OU update is linear, so evolving the spectrum and synthesising the field
    each step is mathematically identical to evolving the real-space field);
    otherwise it is the real-space field on the simulation grid.

    Args:
        key: The PRNG key.
        config: The simulation configuration.
        turbulent_forcing_params: The turbulent-forcing parameters.

    Returns:
        ``(key, f0)``: the advanced PRNG key and the initial persistent field.
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
    """
    Apply Ornstein-Uhlenbeck forcing.

    The persistent forcing field ``f`` (carried in ``forcing_state``) is evolved
    with the exact OU discretisation ``f <- a f + sqrt(1 - a^2) xi``,
    ``a = exp(-dt / tau_f)``, which keeps it at unit rms. The applied field is
    ``f`` itself, or ``f`` rescaled to unit rms with ``ou_unit_rms_each_step``
    (the persistent field is not renormalised). It is added to the velocity

    - by default as a constant-amplitude acceleration ``velocity += F0 f dt``.
      This variant alone is state-independent, so its adjoint is clean and the
      realisation is reproducible for a fixed timestep sequence;
    - with ``ou_exact_injection`` with the amplitude that injects exactly
      ``energy_injection_rate * dt`` (a function of the density and velocity).

    Args:
        forcing_state: ``(key, f)``, the PRNG key and the persistent field (the
            coarse spectrum with ``synthesis_resolution > 0``).
        primitive_state: The primitive state array.
        dt: The time step.
        turbulent_forcing_params: The turbulent-forcing parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        ``((key, f), primitive_state)``: the advanced forcing state and the
        forced primitive state.
    """
    key, persistent_field = forcing_state
    correlation_time = turbulent_forcing_params.correlation_time
    decay_factor = jnp.exp(-dt / correlation_time)
    key, fresh_increment = _draw_solenoidal_increment(key, config, turbulent_forcing_params)
    persistent_field = (
        decay_factor * persistent_field
        + jnp.sqrt(jnp.maximum(1.0 - decay_factor ** 2, 0.0)) * fresh_increment
    )

    # With coarse spectral synthesis the persistent state is the spectrum; the
    # real-space acceleration is synthesised only for this step's application.
    if config.turbulent_forcing_config.synthesis_resolution > 0:
        field = _synthesize_forcing_field(persistent_field, config)
    else:
        field = persistent_field

    if config.turbulent_forcing_config.ou_unit_rms_each_step:
        # AthenaPK rescales the real-space acceleration to ``accel_rms`` every
        # cycle; the persistent spectral field is not renormalised, only the
        # applied copy.
        field_rms = jnp.sqrt(jnp.mean(field[0] ** 2 + field[1] ** 2 + field[2] ** 2) + 1e-30)
        applied_field = field / field_rms
    else:
        applied_field = field

    if config.turbulent_forcing_config.ou_exact_injection:
        # AthenaK ``dedt`` normalisation: scale the (unit-rms) OU field so the
        # box gains exactly Edot*dt of kinetic energy this step.
        amplitude = _exact_injection_amplitude(
            primitive_state,
            applied_field[0],
            applied_field[1],
            applied_field[2],
            dt,
            turbulent_forcing_params.energy_injection_rate,
            config,
            registered_variables,
        )
    else:
        amplitude = turbulent_forcing_params.forcing_amplitude * dt
    primitive_state = _add_velocity_kick(
        primitive_state,
        amplitude,
        applied_field,
        config,
        registered_variables,
    )

    return (key, persistent_field), primitive_state


# -------------------------------------------------------------
# ===== ↑ Ornstein-Uhlenbeck (temporally correlated) forcing ↑ =====
# -------------------------------------------------------------
