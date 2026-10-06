"""
AthenaPK's Riemann solvers for the VL2 finite-volume scheme.

Transcriptions of the interface solvers of AthenaPK (and hence Athena++): HLLD
(Miyoshi & Kusano 2005), HLLE and local Lax-Friedrichs for GLM-MHD, and HLLC,
HLLE and local Lax-Friedrichs for adiabatic hydrodynamics. The GLM-MHD solvers
first solve the decoupled ``(B_normal, psi)`` subsystem exactly (Mignone &
Tzeferacos 2010, eq. 24) and use the resulting interface normal field in the
MHD solver. The expressions follow the C++ closely (including its
floating-point grouping and its special cases), so that the scheme agrees with
AthenaPK to round-off.

Every function is purely elementwise on its operands — it never indexes,
reduces or reshapes — so the same code is evaluated on whole arrays by the
native JAX path and on register tiles inside the Pallas kernels.

All states are given in the frame of the interface: ``velocity_normal`` and
``field_normal`` are the components along the flux direction and the two
transverse components follow in AthenaPK's cyclic order (for an x-interface:
y then z; for a y-interface: z then x; for a z-interface: x then y).
"""

# typing
from typing import NamedTuple, Any

# jax
import jax.numpy as jnp


# -------------------------------------------------------------
# ======================= ↓ Containers ↓ ======================
# -------------------------------------------------------------


class MHDFaceState(NamedTuple):
    """Primitive GLM-MHD state on one side of an interface (interface frame)."""

    density: Any
    velocity_normal: Any
    velocity_transverse_1: Any
    velocity_transverse_2: Any
    pressure: Any
    field_normal: Any
    field_transverse_1: Any
    field_transverse_2: Any
    psi: Any


class MHDFlux(NamedTuple):
    """Interface flux of the GLM-MHD conserved variables (interface frame)."""

    mass: Any
    momentum_normal: Any
    momentum_transverse_1: Any
    momentum_transverse_2: Any
    energy: Any
    field_normal: Any
    field_transverse_1: Any
    field_transverse_2: Any
    psi: Any


class HydroFaceState(NamedTuple):
    """Primitive hydrodynamic state on one side of an interface (interface frame)."""

    density: Any
    velocity_normal: Any
    velocity_transverse_1: Any
    velocity_transverse_2: Any
    pressure: Any


class HydroFlux(NamedTuple):
    """Interface flux of the hydrodynamic conserved variables (interface frame)."""

    mass: Any
    momentum_normal: Any
    momentum_transverse_1: Any
    momentum_transverse_2: Any
    energy: Any


# -------------------------------------------------------------
# ======================= ↑ Containers ↑ ======================
# -------------------------------------------------------------

#: AthenaPK's ``SMALL_NUMBER`` of the HLLD degeneracy check.
_HLLD_SMALL_NUMBER = 1.0e-8

#: Parthenon's ``TINY_NUMBER``, used by the hydro HLLE / HLLC wave speeds.
_TINY_NUMBER = 1.0e-20


def _square(value):
    """AthenaPK's ``SQR(x)``: ``x * x``."""
    return value * value


def fast_magnetosonic_speed(gamma, density, pressure, field_normal, field_transverse_1, field_transverse_2):
    """
    Fast magnetosonic speed along the ``field_normal`` direction, exactly as
    ``AdiabaticGLMMHDEOS::FastMagnetosonicSpeed``.

    Args:
        gamma: The adiabatic index.
        density: The density.
        pressure: The gas pressure.
        field_normal: The field component along the propagation direction.
        field_transverse_1: The first transverse field component.
        field_transverse_2: The second transverse field component.

    Returns:
        The fast magnetosonic speed.
    """
    gamma_pressure = gamma * pressure
    transverse_field_squared = (
        field_transverse_1 * field_transverse_1 + field_transverse_2 * field_transverse_2
    )
    sum_of_squares = field_normal * field_normal + transverse_field_squared + gamma_pressure
    difference_of_squares = field_normal * field_normal + transverse_field_squared - gamma_pressure
    discriminant_root = jnp.sqrt(
        difference_of_squares * difference_of_squares
        + 4.0 * gamma_pressure * transverse_field_squared
    )
    return jnp.sqrt(0.5 * (sum_of_squares + discriminant_root) / density)


def adiabatic_sound_speed(gamma, density, pressure):
    """``AdiabaticHydroEOS::SoundSpeed``: ``sqrt(gamma * p / rho)``."""
    return jnp.sqrt(gamma * pressure / density)


def _glm_interface_field_and_psi(left: MHDFaceState, right: MHDFaceState, cleaning_speed):
    """
    Solve the decoupled ``(B_normal, psi)`` GLM subsystem exactly
    (Mignone & Tzeferacos 2010, eq. 24).

    Args:
        left: The state left of the interface.
        right: The state right of the interface.
        cleaning_speed: The hyperbolic divergence-cleaning speed ``c_h``.

    Returns:
        The interface normal field and the interface psi.
    """
    interface_field_normal = 0.5 * (left.field_normal + right.field_normal) - 0.5 / cleaning_speed * (
        right.psi - left.psi
    )
    interface_psi = 0.5 * (left.psi + right.psi) - 0.5 * cleaning_speed * (
        right.field_normal - left.field_normal
    )
    return interface_field_normal, interface_psi


# -------------------------------------------------------------
# ===================== ↓ GLM-MHD: HLLD ↓ =====================
# -------------------------------------------------------------


class _MHDConserved(NamedTuple):
    """The seven conserved quantities HLLD tracks across its fan (Athena's Cons1D)."""

    density: Any
    momentum_normal: Any
    momentum_transverse_1: Any
    momentum_transverse_2: Any
    energy: Any
    field_transverse_1: Any
    field_transverse_2: Any


def _scaled_jump(speed, upper: _MHDConserved, lower: _MHDConserved) -> _MHDConserved:
    """Return ``speed * (upper - lower)`` component-wise (Athena's step 6)."""
    return _MHDConserved(*(speed * (upper_value - lower_value) for upper_value, lower_value in zip(upper, lower)))


def hlld_flux(left: MHDFaceState, right: MHDFaceState, gamma, cleaning_speed) -> MHDFlux:
    """
    The HLLD flux of Miyoshi & Kusano (2005) in AthenaPK's GLM form
    (``glmmhd_hlld.hpp``, including the Athena++ degeneracy check).

    Args:
        left: The primitive state left of the interface.
        right: The primitive state right of the interface.
        gamma: The adiabatic index.
        cleaning_speed: The hyperbolic divergence-cleaning speed ``c_h``.

    Returns:
        The GLM-MHD interface flux.
    """

    # --------------- ↓ GLM subsystem and left/right states ↓ ----------------

    gamma_minus_one = gamma - 1.0
    inverse_gamma_minus_one = 1.0 / gamma_minus_one

    field_normal, psi_interface = _glm_interface_field_and_psi(left, right, cleaning_speed)
    field_normal_squared = field_normal * field_normal

    # The transverse components are grouped first for floating-point
    # associativity symmetry between the left and right states.
    magnetic_pressure_left = 0.5 * (
        field_normal_squared
        + (_square(left.field_transverse_1) + _square(left.field_transverse_2))
    )
    magnetic_pressure_right = 0.5 * (
        field_normal_squared
        + (_square(right.field_transverse_1) + _square(right.field_transverse_2))
    )
    kinetic_energy_left = 0.5 * left.density * (
        _square(left.velocity_normal)
        + (_square(left.velocity_transverse_1) + _square(left.velocity_transverse_2))
    )
    kinetic_energy_right = 0.5 * right.density * (
        _square(right.velocity_normal)
        + (_square(right.velocity_transverse_1) + _square(right.velocity_transverse_2))
    )

    conserved_left = _MHDConserved(
        density=left.density,
        momentum_normal=left.velocity_normal * left.density,
        momentum_transverse_1=left.velocity_transverse_1 * left.density,
        momentum_transverse_2=left.velocity_transverse_2 * left.density,
        energy=left.pressure * inverse_gamma_minus_one + kinetic_energy_left + magnetic_pressure_left,
        field_transverse_1=left.field_transverse_1,
        field_transverse_2=left.field_transverse_2,
    )
    conserved_right = _MHDConserved(
        density=right.density,
        momentum_normal=right.velocity_normal * right.density,
        momentum_transverse_1=right.velocity_transverse_1 * right.density,
        momentum_transverse_2=right.velocity_transverse_2 * right.density,
        energy=right.pressure * inverse_gamma_minus_one + kinetic_energy_right + magnetic_pressure_right,
        field_transverse_1=right.field_transverse_1,
        field_transverse_2=right.field_transverse_2,
    )

    # --------------- ↑ GLM subsystem and left/right states ↑ ----------------

    # --------------- ↓ Outer wave speeds and left/right fluxes ↓ ----------------

    # The fast speeds use each side's own reconstructed normal field, as in
    # AthenaPK (Miyoshi & Kusano eq. 67).
    fast_speed_left = fast_magnetosonic_speed(
        gamma,
        left.density,
        left.pressure,
        left.field_normal,
        left.field_transverse_1,
        left.field_transverse_2,
    )
    fast_speed_right = fast_magnetosonic_speed(
        gamma,
        right.density,
        right.pressure,
        right.field_normal,
        right.field_transverse_1,
        right.field_transverse_2,
    )

    speed_left_outer = jnp.minimum(
        left.velocity_normal - fast_speed_left,
        right.velocity_normal - fast_speed_right,
    )
    speed_right_outer = jnp.maximum(
        left.velocity_normal + fast_speed_left,
        right.velocity_normal + fast_speed_right,
    )

    total_pressure_left = left.pressure + magnetic_pressure_left
    total_pressure_right = right.pressure + magnetic_pressure_right

    def side_flux(state: MHDFaceState, conserved: _MHDConserved, total_pressure) -> _MHDConserved:
        return _MHDConserved(
            density=conserved.momentum_normal,
            momentum_normal=conserved.momentum_normal * state.velocity_normal
            + total_pressure
            - field_normal_squared,
            momentum_transverse_1=conserved.momentum_transverse_1 * state.velocity_normal
            - field_normal * conserved.field_transverse_1,
            momentum_transverse_2=conserved.momentum_transverse_2 * state.velocity_normal
            - field_normal * conserved.field_transverse_2,
            energy=state.velocity_normal
            * (conserved.energy + total_pressure - field_normal_squared)
            - field_normal
            * (
                state.velocity_transverse_1 * conserved.field_transverse_1
                + state.velocity_transverse_2 * conserved.field_transverse_2
            ),
            field_transverse_1=conserved.field_transverse_1 * state.velocity_normal
            - field_normal * state.velocity_transverse_1,
            field_transverse_2=conserved.field_transverse_2 * state.velocity_normal
            - field_normal * state.velocity_transverse_2,
        )

    flux_left = side_flux(left, conserved_left, total_pressure_left)
    flux_right = side_flux(right, conserved_right, total_pressure_right)

    # --------------- ↑ Outer wave speeds and left/right fluxes ↑ ----------------

    # --------------- ↓ Contact and Alfvén speeds ↓ ----------------

    speed_minus_velocity_left = speed_left_outer - left.velocity_normal
    speed_minus_velocity_right = speed_right_outer - right.velocity_normal

    # S_M, Miyoshi & Kusano eq. (38); the pressure terms are grouped for
    # floating-point associativity symmetry.
    contact_speed = (
        speed_minus_velocity_right * conserved_right.momentum_normal
        - speed_minus_velocity_left * conserved_left.momentum_normal
        + (total_pressure_left - total_pressure_right)
    ) / (
        speed_minus_velocity_right * conserved_right.density
        - speed_minus_velocity_left * conserved_left.density
    )

    speed_minus_contact_left = speed_left_outer - contact_speed
    speed_minus_contact_right = speed_right_outer - contact_speed
    inverse_speed_minus_contact_left = 1.0 / speed_minus_contact_left
    inverse_speed_minus_contact_right = 1.0 / speed_minus_contact_right

    # Eq. (43): the star-state densities.
    star_density_left = conserved_left.density * speed_minus_velocity_left * inverse_speed_minus_contact_left
    star_density_right = (
        conserved_right.density * speed_minus_velocity_right * inverse_speed_minus_contact_right
    )
    inverse_star_density_left = 1.0 / star_density_left
    inverse_star_density_right = 1.0 / star_density_right
    sqrt_star_density_left = jnp.sqrt(star_density_left)
    sqrt_star_density_right = jnp.sqrt(star_density_right)

    # Eq. (51): the rotational (Alfvén) speeds.
    speed_left_alfven = contact_speed - jnp.abs(field_normal) / sqrt_star_density_left
    speed_right_alfven = contact_speed + jnp.abs(field_normal) / sqrt_star_density_right

    # --------------- ↑ Contact and Alfvén speeds ↑ ----------------

    # --------------- ↓ Star (*) states ↓ ----------------

    # Eq. (41): the total pressure of the star states, averaged over the two
    # sides (which agree analytically).
    star_total_pressure_left = total_pressure_left + conserved_left.density * speed_minus_velocity_left * (
        contact_speed - left.velocity_normal
    )
    star_total_pressure_right = total_pressure_right + conserved_right.density * speed_minus_velocity_right * (
        contact_speed - right.velocity_normal
    )
    star_total_pressure = 0.5 * (star_total_pressure_right + star_total_pressure_left)

    def star_state(
        state: MHDFaceState,
        conserved: _MHDConserved,
        total_pressure,
        star_density,
        inverse_star_density,
        speed_minus_velocity,
        speed_minus_contact,
        inverse_speed_minus_contact,
    ):
        star_momentum_normal = star_density * contact_speed

        # Eqs. (44)-(47), with the Athena++ guard against the degenerate case
        # in which the denominator vanishes (the star state then keeps the
        # transverse velocity and field of the outer state).
        denominator = conserved.density * speed_minus_velocity * speed_minus_contact - field_normal_squared
        degenerate = jnp.abs(denominator) < _HLLD_SMALL_NUMBER * star_total_pressure
        safe_denominator = jnp.where(degenerate, 1.0, denominator)

        velocity_factor = field_normal * (speed_minus_velocity - speed_minus_contact) / safe_denominator
        field_factor = (
            conserved.density * _square(speed_minus_velocity) - field_normal_squared
        ) / safe_denominator

        star_momentum_transverse_1 = jnp.where(
            degenerate,
            star_density * state.velocity_transverse_1,
            star_density * (state.velocity_transverse_1 - conserved.field_transverse_1 * velocity_factor),
        )
        star_momentum_transverse_2 = jnp.where(
            degenerate,
            star_density * state.velocity_transverse_2,
            star_density * (state.velocity_transverse_2 - conserved.field_transverse_2 * velocity_factor),
        )
        star_field_transverse_1 = jnp.where(
            degenerate,
            conserved.field_transverse_1,
            conserved.field_transverse_1 * field_factor,
        )
        star_field_transverse_2 = jnp.where(
            degenerate,
            conserved.field_transverse_2,
            conserved.field_transverse_2 * field_factor,
        )

        # v* . B*, with the transverse terms grouped for associativity symmetry.
        star_velocity_dot_field = (
            star_momentum_normal * field_normal
            + (
                star_momentum_transverse_1 * star_field_transverse_1
                + star_momentum_transverse_2 * star_field_transverse_2
            )
        ) * inverse_star_density

        # Eq. (48): the star energy.
        star_energy = (
            speed_minus_velocity * conserved.energy
            - total_pressure * state.velocity_normal
            + star_total_pressure * contact_speed
            + field_normal
            * (
                state.velocity_normal * field_normal
                + (
                    state.velocity_transverse_1 * conserved.field_transverse_1
                    + state.velocity_transverse_2 * conserved.field_transverse_2
                )
                - star_velocity_dot_field
            )
        ) * inverse_speed_minus_contact

        star = _MHDConserved(
            density=star_density,
            momentum_normal=star_momentum_normal,
            momentum_transverse_1=star_momentum_transverse_1,
            momentum_transverse_2=star_momentum_transverse_2,
            energy=star_energy,
            field_transverse_1=star_field_transverse_1,
            field_transverse_2=star_field_transverse_2,
        )
        return star, star_velocity_dot_field

    star_left, star_velocity_dot_field_left = star_state(
        left,
        conserved_left,
        total_pressure_left,
        star_density_left,
        inverse_star_density_left,
        speed_minus_velocity_left,
        speed_minus_contact_left,
        inverse_speed_minus_contact_left,
    )
    star_right, star_velocity_dot_field_right = star_state(
        right,
        conserved_right,
        total_pressure_right,
        star_density_right,
        inverse_star_density_right,
        speed_minus_velocity_right,
        speed_minus_contact_right,
        inverse_speed_minus_contact_right,
    )

    # --------------- ↑ Star (*) states ↑ ----------------

    # --------------- ↓ Double-star (**) states ↓ ----------------

    # If B_normal is near zero these coincide with the star states.
    inverse_sum_sqrt_density = 1.0 / (sqrt_star_density_left + sqrt_star_density_right)
    # The sign is built from a typed operand: a select between two bare
    # literals would enter the Pallas/Triton lowering as the default float type.
    unit = jnp.ones_like(field_normal)
    field_normal_sign = jnp.where(field_normal > 0.0, unit, -unit)

    # Eqs. (59) and (60): the common transverse velocity.
    double_star_velocity_transverse_1 = inverse_sum_sqrt_density * (
        sqrt_star_density_left * (star_left.momentum_transverse_1 * inverse_star_density_left)
        + sqrt_star_density_right * (star_right.momentum_transverse_1 * inverse_star_density_right)
        + field_normal_sign * (star_right.field_transverse_1 - star_left.field_transverse_1)
    )
    double_star_velocity_transverse_2 = inverse_sum_sqrt_density * (
        sqrt_star_density_left * (star_left.momentum_transverse_2 * inverse_star_density_left)
        + sqrt_star_density_right * (star_right.momentum_transverse_2 * inverse_star_density_right)
        + field_normal_sign * (star_right.field_transverse_2 - star_left.field_transverse_2)
    )

    # Eqs. (61) and (62): the common transverse field.
    double_star_field_transverse_1 = inverse_sum_sqrt_density * (
        sqrt_star_density_left * star_right.field_transverse_1
        + sqrt_star_density_right * star_left.field_transverse_1
        + field_normal_sign
        * sqrt_star_density_left
        * sqrt_star_density_right
        * (
            (star_right.momentum_transverse_1 * inverse_star_density_right)
            - (star_left.momentum_transverse_1 * inverse_star_density_left)
        )
    )
    double_star_field_transverse_2 = inverse_sum_sqrt_density * (
        sqrt_star_density_left * star_right.field_transverse_2
        + sqrt_star_density_right * star_left.field_transverse_2
        + field_normal_sign
        * sqrt_star_density_left
        * sqrt_star_density_right
        * (
            (star_right.momentum_transverse_2 * inverse_star_density_right)
            - (star_left.momentum_transverse_2 * inverse_star_density_left)
        )
    )

    double_star_left_momentum_transverse_1 = star_left.density * double_star_velocity_transverse_1
    double_star_left_momentum_transverse_2 = star_left.density * double_star_velocity_transverse_2

    # Eq. (63): the double-star energies.
    double_star_velocity_dot_field = contact_speed * field_normal + (
        double_star_left_momentum_transverse_1 * double_star_field_transverse_1
        + double_star_left_momentum_transverse_2 * double_star_field_transverse_2
    ) / star_left.density

    double_star_left = _MHDConserved(
        density=star_left.density,
        momentum_normal=star_left.momentum_normal,
        momentum_transverse_1=double_star_left_momentum_transverse_1,
        momentum_transverse_2=double_star_left_momentum_transverse_2,
        energy=star_left.energy
        - sqrt_star_density_left * field_normal_sign * (star_velocity_dot_field_left - double_star_velocity_dot_field),
        field_transverse_1=double_star_field_transverse_1,
        field_transverse_2=double_star_field_transverse_2,
    )
    double_star_right = _MHDConserved(
        density=star_right.density,
        momentum_normal=star_right.momentum_normal,
        momentum_transverse_1=star_right.density * double_star_velocity_transverse_1,
        momentum_transverse_2=star_right.density * double_star_velocity_transverse_2,
        energy=star_right.energy
        + sqrt_star_density_right * field_normal_sign * (star_velocity_dot_field_right - double_star_velocity_dot_field),
        field_transverse_1=double_star_field_transverse_1,
        field_transverse_2=double_star_field_transverse_2,
    )

    # --------------- ↑ Double-star (**) states ↑ ----------------

    # --------------- ↓ Flux selection ↓ ----------------

    # The jumps across the individual waves, S_k * (U_k - U_{k-1}).
    jump_left_alfven = _scaled_jump(speed_left_alfven, double_star_left, star_left)
    jump_left_outer = _scaled_jump(speed_left_outer, star_left, conserved_left)
    jump_right_alfven = _scaled_jump(speed_right_alfven, double_star_right, star_right)
    jump_right_outer = _scaled_jump(speed_right_outer, star_right, conserved_right)

    flux_left_star = _MHDConserved(*(f + j for f, j in zip(flux_left, jump_left_outer)))
    flux_right_star = _MHDConserved(*(f + j for f, j in zip(flux_right, jump_right_outer)))
    flux_left_double_star = _MHDConserved(*(f + j for f, j in zip(flux_left_star, jump_left_alfven)))
    flux_right_double_star = _MHDConserved(*(f + j for f, j in zip(flux_right_star, jump_right_alfven)))

    def select(component: int):
        selected = jnp.where(contact_speed >= 0.0, flux_left_double_star[component], flux_right_double_star[component])
        selected = jnp.where(speed_right_alfven <= 0.0, flux_right_star[component], selected)
        selected = jnp.where(speed_left_alfven >= 0.0, flux_left_star[component], selected)
        selected = jnp.where(speed_right_outer <= 0.0, flux_right[component], selected)
        selected = jnp.where(speed_left_outer >= 0.0, flux_left[component], selected)
        return selected

    selected_flux = _MHDConserved(*(select(component) for component in range(7)))

    # --------------- ↑ Flux selection ↑ ----------------

    return MHDFlux(
        mass=selected_flux.density,
        momentum_normal=selected_flux.momentum_normal,
        momentum_transverse_1=selected_flux.momentum_transverse_1,
        momentum_transverse_2=selected_flux.momentum_transverse_2,
        energy=selected_flux.energy,
        field_normal=psi_interface,
        field_transverse_1=selected_flux.field_transverse_1,
        field_transverse_2=selected_flux.field_transverse_2,
        psi=cleaning_speed * cleaning_speed * field_normal,
    )


# -------------------------------------------------------------
# ===================== ↑ GLM-MHD: HLLD ↑ =====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================== ↓ GLM-MHD: HLLE, LLF ↓ ===================
# -------------------------------------------------------------


def hlle_mhd_flux(left: MHDFaceState, right: MHDFaceState, gamma, cleaning_speed) -> MHDFlux:
    """
    The HLLE flux for GLM-MHD with Roe-averaged (Einfeldt) wave speeds,
    as AthenaPK's ``glmmhd_hlle.hpp``.

    Args:
        left: The primitive state left of the interface.
        right: The primitive state right of the interface.
        gamma: The adiabatic index.
        cleaning_speed: The hyperbolic divergence-cleaning speed ``c_h``.

    Returns:
        The GLM-MHD interface flux.
    """
    gamma_minus_one = gamma - 1.0

    field_normal, psi_interface = _glm_interface_field_and_psi(left, right, cleaning_speed)

    # --------------- ↓ Roe-averaged state ↓ ----------------

    sqrt_density_left = jnp.sqrt(left.density)
    sqrt_density_right = jnp.sqrt(right.density)
    inverse_sum_sqrt_density = 1.0 / (sqrt_density_left + sqrt_density_right)

    roe_density = sqrt_density_left * sqrt_density_right
    roe_velocity_normal = (
        sqrt_density_left * left.velocity_normal + sqrt_density_right * right.velocity_normal
    ) * inverse_sum_sqrt_density
    roe_velocity_transverse_1 = (
        sqrt_density_left * left.velocity_transverse_1 + sqrt_density_right * right.velocity_transverse_1
    ) * inverse_sum_sqrt_density
    roe_velocity_transverse_2 = (
        sqrt_density_left * left.velocity_transverse_2 + sqrt_density_right * right.velocity_transverse_2
    ) * inverse_sum_sqrt_density
    # The Roe average of the field weights the sides the other way round.
    roe_field_transverse_1 = (
        sqrt_density_right * left.field_transverse_1 + sqrt_density_left * right.field_transverse_1
    ) * inverse_sum_sqrt_density
    roe_field_transverse_2 = (
        sqrt_density_right * left.field_transverse_2 + sqrt_density_left * right.field_transverse_2
    ) * inverse_sum_sqrt_density
    x_term = (
        0.5
        * (
            _square(left.field_transverse_1 - right.field_transverse_1)
            + _square(left.field_transverse_2 - right.field_transverse_2)
        )
        / (_square(sqrt_density_left + sqrt_density_right))
    )
    y_term = 0.5 * (left.density + right.density) / roe_density

    # Roe (1981): average the enthalpy H = (E + P) / rho rather than E or P.
    magnetic_pressure_left = 0.5 * (
        field_normal * field_normal + _square(left.field_transverse_1) + _square(left.field_transverse_2)
    )
    magnetic_pressure_right = 0.5 * (
        field_normal * field_normal + _square(right.field_transverse_1) + _square(right.field_transverse_2)
    )
    energy_left = (
        left.pressure / gamma_minus_one
        + 0.5
        * left.density
        * (
            _square(left.velocity_normal)
            + _square(left.velocity_transverse_1)
            + _square(left.velocity_transverse_2)
        )
        + magnetic_pressure_left
    )
    energy_right = (
        right.pressure / gamma_minus_one
        + 0.5
        * right.density
        * (
            _square(right.velocity_normal)
            + _square(right.velocity_transverse_1)
            + _square(right.velocity_transverse_2)
        )
        + magnetic_pressure_right
    )
    roe_enthalpy = (
        (energy_left + left.pressure + magnetic_pressure_left) / sqrt_density_left
        + (energy_right + right.pressure + magnetic_pressure_right) / sqrt_density_right
    ) * inverse_sum_sqrt_density

    # --------------- ↑ Roe-averaged state ↑ ----------------

    # --------------- ↓ Wave speeds ↓ ----------------

    fast_speed_left = fast_magnetosonic_speed(
        gamma,
        left.density,
        left.pressure,
        left.field_normal,
        left.field_transverse_1,
        left.field_transverse_2,
    )
    fast_speed_right = fast_magnetosonic_speed(
        gamma,
        right.density,
        right.pressure,
        right.field_normal,
        right.field_transverse_1,
        right.field_transverse_2,
    )

    # The Roe-averaged fast speed (Stone et al. 2008, eq. B18).
    roe_transverse_field_squared = _square(roe_field_transverse_1) + _square(roe_field_transverse_2)
    normal_alfven_speed_squared = field_normal * field_normal / roe_density
    transverse_field_star_squared = (gamma_minus_one - (gamma_minus_one - 1.0) * y_term) * roe_transverse_field_squared
    enthalpy_minus_magnetic = roe_enthalpy - (normal_alfven_speed_squared + roe_transverse_field_squared / roe_density)
    roe_velocity_squared = (
        _square(roe_velocity_normal) + _square(roe_velocity_transverse_1) + _square(roe_velocity_transverse_2)
    )
    sound_speed_squared_tilde = jnp.maximum(
        (gamma_minus_one * (enthalpy_minus_magnetic - 0.5 * roe_velocity_squared) - (gamma_minus_one - 1.0) * x_term),
        0.0,
    )
    transverse_alfven_speed_squared = transverse_field_star_squared / roe_density
    speed_sum = normal_alfven_speed_squared + transverse_alfven_speed_squared + sound_speed_squared_tilde
    speed_difference = normal_alfven_speed_squared + transverse_alfven_speed_squared - sound_speed_squared_tilde
    discriminant_root = jnp.sqrt(
        speed_difference * speed_difference + 4.0 * sound_speed_squared_tilde * transverse_alfven_speed_squared
    )
    roe_fast_speed = jnp.sqrt(0.5 * (speed_sum + discriminant_root))

    speed_left = jnp.minimum((roe_velocity_normal - roe_fast_speed), (left.velocity_normal - fast_speed_left))
    speed_right = jnp.maximum((roe_velocity_normal + roe_fast_speed), (right.velocity_normal + fast_speed_right))

    speed_plus = jnp.where(speed_right > 0.0, speed_right, 0.0)
    speed_minus = jnp.where(speed_left < 0.0, speed_left, 0.0)

    # --------------- ↑ Wave speeds ↑ ----------------

    # --------------- ↓ HLLE flux ↓ ----------------

    # The left/right fluxes along the lines speed_minus / speed_plus:
    # F_L - S_L U_L and F_R - S_R U_R.
    relative_velocity_left = left.velocity_normal - speed_minus
    relative_velocity_right = right.velocity_normal - speed_plus

    def side_flux(state: MHDFaceState, relative_velocity, magnetic_pressure, energy):
        momentum_normal_flux = (
            state.density * state.velocity_normal * relative_velocity
            + magnetic_pressure
            - _square(field_normal)
        )
        energy_flux = energy * relative_velocity + state.velocity_normal * (
            state.pressure + magnetic_pressure - field_normal * field_normal
        )
        energy_flux = energy_flux - field_normal * (
            state.field_transverse_1 * state.velocity_transverse_1
            + state.field_transverse_2 * state.velocity_transverse_2
        )
        return (
            state.density * relative_velocity,
            momentum_normal_flux + state.pressure,
            state.density * state.velocity_transverse_1 * relative_velocity
            - field_normal * state.field_transverse_1,
            state.density * state.velocity_transverse_2 * relative_velocity
            - field_normal * state.field_transverse_2,
            energy_flux,
            state.field_transverse_1 * relative_velocity - field_normal * state.velocity_transverse_1,
            state.field_transverse_2 * relative_velocity - field_normal * state.velocity_transverse_2,
        )

    flux_left = side_flux(left, relative_velocity_left, magnetic_pressure_left, energy_left)
    flux_right = side_flux(right, relative_velocity_right, magnetic_pressure_right, energy_right)

    speeds_differ = speed_plus != speed_minus
    safe_speed_difference = jnp.where(speeds_differ, speed_plus - speed_minus, 1.0)
    weight = jnp.where(speeds_differ, 0.5 * (speed_plus + speed_minus) / safe_speed_difference, 0.0)

    hlle = tuple(
        0.5 * (value_left + value_right) + (value_left - value_right) * weight
        for value_left, value_right in zip(flux_left, flux_right)
    )

    # --------------- ↑ HLLE flux ↑ ----------------

    return MHDFlux(
        mass=hlle[0],
        momentum_normal=hlle[1],
        momentum_transverse_1=hlle[2],
        momentum_transverse_2=hlle[3],
        energy=hlle[4],
        field_normal=psi_interface,
        field_transverse_1=hlle[5],
        field_transverse_2=hlle[6],
        psi=cleaning_speed * cleaning_speed * field_normal,
    )


def llf_mhd_flux(left: MHDFaceState, right: MHDFaceState, gamma, cleaning_speed) -> MHDFlux:
    """
    The local Lax-Friedrichs (Rusanov) flux for GLM-MHD, as AthenaPK's
    ``glmmhd_dc_llf.hpp`` (used by the first-order flux correction).

    Args:
        left: The primitive state left of the interface.
        right: The primitive state right of the interface.
        gamma: The adiabatic index.
        cleaning_speed: The hyperbolic divergence-cleaning speed ``c_h``.

    Returns:
        The GLM-MHD interface flux.
    """
    inverse_gamma_minus_one = 1.0 / (gamma - 1.0)

    field_normal, psi_interface = _glm_interface_field_and_psi(left, right, cleaning_speed)

    # --------------- ↓ Sum of the left and right fluxes ↓ ----------------

    mass_flux_left = left.density * left.velocity_normal
    mass_flux_right = right.density * right.velocity_normal
    magnetic_term_left = 0.5 * (
        _square(left.field_transverse_1) + _square(left.field_transverse_2) - _square(field_normal)
    )
    magnetic_term_right = 0.5 * (
        _square(right.field_transverse_1) + _square(right.field_transverse_2) - _square(field_normal)
    )

    sum_mass = mass_flux_left + mass_flux_right
    sum_momentum_normal = (
        mass_flux_left * left.velocity_normal
        + mass_flux_right * right.velocity_normal
        + magnetic_term_left
        + magnetic_term_right
    )
    sum_momentum_transverse_1 = (
        mass_flux_left * left.velocity_transverse_1
        + mass_flux_right * right.velocity_transverse_1
        - field_normal * (left.field_transverse_1 + right.field_transverse_1)
    )
    sum_momentum_transverse_2 = (
        mass_flux_left * left.velocity_transverse_2
        + mass_flux_right * right.velocity_transverse_2
        - field_normal * (left.field_transverse_2 + right.field_transverse_2)
    )
    sum_field_transverse_1 = (
        left.field_transverse_1 * left.velocity_normal
        + right.field_transverse_1 * right.velocity_normal
        - field_normal * (left.velocity_transverse_1 + right.velocity_transverse_1)
    )
    sum_field_transverse_2 = (
        left.field_transverse_2 * left.velocity_normal
        + right.field_transverse_2 * right.velocity_normal
        - field_normal * (left.velocity_transverse_2 + right.velocity_transverse_2)
    )

    energy_left = (
        left.pressure * inverse_gamma_minus_one
        + 0.5
        * left.density
        * (
            _square(left.velocity_normal)
            + _square(left.velocity_transverse_1)
            + _square(left.velocity_transverse_2)
        )
        + magnetic_term_left
        + _square(field_normal)
    )
    energy_right = (
        right.pressure * inverse_gamma_minus_one
        + 0.5
        * right.density
        * (
            _square(right.velocity_normal)
            + _square(right.velocity_transverse_1)
            + _square(right.velocity_transverse_2)
        )
        + magnetic_term_right
        + _square(field_normal)
    )
    sum_momentum_normal = sum_momentum_normal + (left.pressure + right.pressure)
    sum_energy = (energy_left + left.pressure + magnetic_term_left) * left.velocity_normal + (
        energy_right + right.pressure + magnetic_term_right
    ) * right.velocity_normal
    sum_energy = sum_energy - field_normal * (
        left.field_transverse_1 * left.velocity_transverse_1
        + left.field_transverse_2 * left.velocity_transverse_2
    )
    sum_energy = sum_energy - field_normal * (
        right.field_transverse_1 * right.velocity_transverse_1
        + right.field_transverse_2 * right.velocity_transverse_2
    )

    # --------------- ↑ Sum of the left and right fluxes ↑ ----------------

    # --------------- ↓ Dissipation ↓ ----------------

    fast_speed_left = fast_magnetosonic_speed(
        gamma,
        left.density,
        left.pressure,
        left.field_normal,
        left.field_transverse_1,
        left.field_transverse_2,
    )
    fast_speed_right = fast_magnetosonic_speed(
        gamma,
        right.density,
        right.pressure,
        right.field_normal,
        right.field_transverse_1,
        right.field_transverse_2,
    )
    maximum_speed = jnp.maximum(
        (jnp.abs(left.velocity_normal) + fast_speed_left),
        (jnp.abs(right.velocity_normal) + fast_speed_right),
    )

    jump_mass = maximum_speed * (right.density - left.density)
    jump_momentum_normal = maximum_speed * (
        right.density * right.velocity_normal - left.density * left.velocity_normal
    )
    jump_momentum_transverse_1 = maximum_speed * (
        right.density * right.velocity_transverse_1 - left.density * left.velocity_transverse_1
    )
    jump_momentum_transverse_2 = maximum_speed * (
        right.density * right.velocity_transverse_2 - left.density * left.velocity_transverse_2
    )
    jump_energy = maximum_speed * (energy_right - energy_left)
    jump_field_transverse_1 = maximum_speed * (right.field_transverse_1 - left.field_transverse_1)
    jump_field_transverse_2 = maximum_speed * (right.field_transverse_2 - left.field_transverse_2)

    # --------------- ↑ Dissipation ↑ ----------------

    return MHDFlux(
        mass=0.5 * (sum_mass - jump_mass),
        momentum_normal=0.5 * (sum_momentum_normal - jump_momentum_normal),
        momentum_transverse_1=0.5 * (sum_momentum_transverse_1 - jump_momentum_transverse_1),
        momentum_transverse_2=0.5 * (sum_momentum_transverse_2 - jump_momentum_transverse_2),
        energy=0.5 * (sum_energy - jump_energy),
        field_normal=psi_interface,
        field_transverse_1=0.5 * (sum_field_transverse_1 - jump_field_transverse_1),
        field_transverse_2=0.5 * (sum_field_transverse_2 - jump_field_transverse_2),
        psi=cleaning_speed * cleaning_speed * field_normal,
    )


# -------------------------------------------------------------
# ================== ↑ GLM-MHD: HLLE, LLF ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================ ↓ Hydro: HLLC, HLLE, LLF ↓ =================
# -------------------------------------------------------------


def _hydro_energy(state: HydroFaceState, inverse_gamma_minus_one):
    """Total energy density ``p/(gamma-1) + rho v^2 / 2`` in AthenaPK's order."""
    return state.pressure * inverse_gamma_minus_one + 0.5 * state.density * (
        _square(state.velocity_normal)
        + _square(state.velocity_transverse_1)
        + _square(state.velocity_transverse_2)
    )


def hllc_hydro_flux(left: HydroFaceState, right: HydroFaceState, gamma) -> HydroFlux:
    """
    The HLLC flux for adiabatic hydrodynamics with PVRS wave-speed estimates
    (Toro 10.5.2), as AthenaPK's ``hydro_hllc.hpp``.

    Args:
        left: The primitive state left of the interface.
        right: The primitive state right of the interface.
        gamma: The adiabatic index.

    Returns:
        The hydrodynamic interface flux.
    """
    inverse_gamma_minus_one = 1.0 / (gamma - 1.0)

    # --------------- ↓ Wave speeds ↓ ----------------

    sound_speed_left = adiabatic_sound_speed(gamma, left.density, left.pressure)
    sound_speed_right = adiabatic_sound_speed(gamma, right.density, right.pressure)
    energy_left = _hydro_energy(left, inverse_gamma_minus_one)
    energy_right = _hydro_energy(right, inverse_gamma_minus_one)

    # The middle-state pressure estimate of the primitive-variable solver.
    average_density = 0.5 * (left.density + right.density)
    average_sound_speed = 0.5 * (sound_speed_left + sound_speed_right)
    middle_pressure = 0.5 * (
        left.pressure
        + right.pressure
        + (left.velocity_normal - right.velocity_normal) * average_density * average_sound_speed
    )

    # Shock-strength corrections of the outer wave speeds.
    shock_factor_left = jnp.where(
        middle_pressure <= left.pressure,
        1.0,
        jnp.sqrt(1.0 + (gamma + 1) / (2 * gamma) * (middle_pressure / left.pressure - 1.0)),
    )
    shock_factor_right = jnp.where(
        middle_pressure <= right.pressure,
        1.0,
        jnp.sqrt(1.0 + (gamma + 1) / (2 * gamma) * (middle_pressure / right.pressure - 1.0)),
    )

    speed_left = left.velocity_normal - sound_speed_left * shock_factor_left
    speed_right = right.velocity_normal + sound_speed_right * shock_factor_right

    speed_plus = jnp.where(speed_right > 0.0, speed_right, _TINY_NUMBER)
    speed_minus = jnp.where(speed_left < 0.0, speed_left, -_TINY_NUMBER)

    # --------------- ↑ Wave speeds ↑ ----------------

    # --------------- ↓ Contact wave ↓ ----------------

    relative_velocity_left = left.velocity_normal - speed_left
    relative_velocity_right = right.velocity_normal - speed_right

    momentum_term_left = left.pressure + relative_velocity_left * left.density * left.velocity_normal
    momentum_term_right = right.pressure + relative_velocity_right * right.density * right.velocity_normal

    mass_term_left = left.density * relative_velocity_left
    mass_term_right = -(right.density * relative_velocity_right)

    contact_speed = (momentum_term_left - momentum_term_right) / (mass_term_left + mass_term_right)
    contact_pressure = (mass_term_left * momentum_term_right + mass_term_right * momentum_term_left) / (
        mass_term_left + mass_term_right
    )
    contact_pressure = jnp.where(contact_pressure > 0.0, contact_pressure, 0.0)

    # --------------- ↑ Contact wave ↑ ----------------

    # --------------- ↓ HLLC flux ↓ ----------------

    relative_velocity_left = left.velocity_normal - speed_minus
    relative_velocity_right = right.velocity_normal - speed_plus

    def side_flux(state: HydroFaceState, relative_velocity, energy):
        return (
            state.density * relative_velocity,
            state.density * state.velocity_normal * relative_velocity + state.pressure,
            state.density * state.velocity_transverse_1 * relative_velocity,
            state.density * state.velocity_transverse_2 * relative_velocity,
            energy * relative_velocity + state.pressure * state.velocity_normal,
        )

    flux_left = side_flux(left, relative_velocity_left, energy_left)
    flux_right = side_flux(right, relative_velocity_right, energy_right)

    # The weights of the left, right and contact contributions.
    contact_moves_right = contact_speed >= 0.0
    safe_left_denominator = jnp.where(contact_moves_right, contact_speed - speed_minus, 1.0)
    safe_right_denominator = jnp.where(contact_moves_right, 1.0, speed_plus - contact_speed)
    weight_left = jnp.where(contact_moves_right, contact_speed / safe_left_denominator, 0.0)
    weight_right = jnp.where(contact_moves_right, 0.0, -contact_speed / safe_right_denominator)
    weight_contact = jnp.where(
        contact_moves_right,
        -speed_minus / safe_left_denominator,
        speed_plus / safe_right_denominator,
    )

    # --------------- ↑ HLLC flux ↑ ----------------

    return HydroFlux(
        mass=weight_left * flux_left[0] + weight_right * flux_right[0],
        momentum_normal=weight_left * flux_left[1] + weight_right * flux_right[1] + weight_contact * contact_pressure,
        momentum_transverse_1=weight_left * flux_left[2] + weight_right * flux_right[2],
        momentum_transverse_2=weight_left * flux_left[3] + weight_right * flux_right[3],
        energy=weight_left * flux_left[4]
        + weight_right * flux_right[4]
        + weight_contact * contact_pressure * contact_speed,
    )


def hlle_hydro_flux(left: HydroFaceState, right: HydroFaceState, gamma) -> HydroFlux:
    """
    The HLLE flux for adiabatic hydrodynamics with Roe-averaged (Einfeldt) wave
    speeds, as AthenaPK's ``hydro_hlle.hpp``.

    NOTE: AthenaPK clamps the left speed to ``+TINY_NUMBER`` (not ``-TINY``)
    when it is non-negative; this is kept to stay faithful to AthenaPK.

    Args:
        left: The primitive state left of the interface.
        right: The primitive state right of the interface.
        gamma: The adiabatic index.

    Returns:
        The hydrodynamic interface flux.
    """
    gamma_minus_one = gamma - 1.0
    inverse_gamma_minus_one = 1.0 / gamma_minus_one

    sqrt_density_left = jnp.sqrt(left.density)
    sqrt_density_right = jnp.sqrt(right.density)
    inverse_sum_sqrt_density = 1.0 / (sqrt_density_left + sqrt_density_right)

    roe_velocity_normal = (
        sqrt_density_left * left.velocity_normal + sqrt_density_right * right.velocity_normal
    ) * inverse_sum_sqrt_density
    roe_velocity_transverse_1 = (
        sqrt_density_left * left.velocity_transverse_1 + sqrt_density_right * right.velocity_transverse_1
    ) * inverse_sum_sqrt_density
    roe_velocity_transverse_2 = (
        sqrt_density_left * left.velocity_transverse_2 + sqrt_density_right * right.velocity_transverse_2
    ) * inverse_sum_sqrt_density

    energy_left = _hydro_energy(left, inverse_gamma_minus_one)
    energy_right = _hydro_energy(right, inverse_gamma_minus_one)
    roe_enthalpy = (
        (energy_left + left.pressure) / sqrt_density_left
        + (energy_right + right.pressure) / sqrt_density_right
    ) * inverse_sum_sqrt_density

    sound_speed_left = adiabatic_sound_speed(gamma, left.density, left.pressure)
    sound_speed_right = adiabatic_sound_speed(gamma, right.density, right.pressure)
    enthalpy_term = roe_enthalpy - 0.5 * (
        _square(roe_velocity_normal) + _square(roe_velocity_transverse_1) + _square(roe_velocity_transverse_2)
    )
    roe_sound_speed = jnp.where(
        enthalpy_term < 0.0,
        0.0,
        jnp.sqrt(gamma_minus_one * jnp.where(enthalpy_term < 0.0, 0.0, enthalpy_term)),
    )

    speed_left = jnp.minimum((roe_velocity_normal - roe_sound_speed), (left.velocity_normal - sound_speed_left))
    speed_right = jnp.maximum((roe_velocity_normal + roe_sound_speed), (right.velocity_normal + sound_speed_right))

    speed_plus = jnp.where(speed_right > 0.0, speed_right, _TINY_NUMBER)
    speed_minus = jnp.where(speed_left < 0.0, speed_left, _TINY_NUMBER)

    relative_velocity_left = left.velocity_normal - speed_minus
    relative_velocity_right = right.velocity_normal - speed_plus

    def side_flux(state: HydroFaceState, relative_velocity, energy):
        return (
            state.density * relative_velocity,
            state.density * state.velocity_normal * relative_velocity + state.pressure,
            state.density * state.velocity_transverse_1 * relative_velocity,
            state.density * state.velocity_transverse_2 * relative_velocity,
            energy * relative_velocity + state.pressure * state.velocity_normal,
        )

    flux_left = side_flux(left, relative_velocity_left, energy_left)
    flux_right = side_flux(right, relative_velocity_right, energy_right)

    speeds_differ = speed_plus != speed_minus
    safe_speed_difference = jnp.where(speeds_differ, speed_plus - speed_minus, 1.0)
    weight = jnp.where(speeds_differ, 0.5 * (speed_plus + speed_minus) / safe_speed_difference, 0.0)

    return HydroFlux(
        *(
            0.5 * (value_left + value_right) + (value_left - value_right) * weight
            for value_left, value_right in zip(flux_left, flux_right)
        )
    )


def llf_hydro_flux(left: HydroFaceState, right: HydroFaceState, gamma) -> HydroFlux:
    """
    The local Lax-Friedrichs (Rusanov) flux for adiabatic hydrodynamics, as
    AthenaPK's ``hydro_dc_llf.hpp``.

    Args:
        left: The primitive state left of the interface.
        right: The primitive state right of the interface.
        gamma: The adiabatic index.

    Returns:
        The hydrodynamic interface flux.
    """
    inverse_gamma_minus_one = 1.0 / (gamma - 1.0)

    mass_flux_left = left.density * left.velocity_normal
    mass_flux_right = right.density * right.velocity_normal

    sum_mass = mass_flux_left + mass_flux_right
    sum_momentum_normal = mass_flux_left * left.velocity_normal + mass_flux_right * right.velocity_normal
    sum_momentum_transverse_1 = (
        mass_flux_left * left.velocity_transverse_1 + mass_flux_right * right.velocity_transverse_1
    )
    sum_momentum_transverse_2 = (
        mass_flux_left * left.velocity_transverse_2 + mass_flux_right * right.velocity_transverse_2
    )

    energy_left = _hydro_energy(left, inverse_gamma_minus_one)
    energy_right = _hydro_energy(right, inverse_gamma_minus_one)
    sum_momentum_normal = sum_momentum_normal + (left.pressure + right.pressure)
    sum_energy = (energy_left + left.pressure) * left.velocity_normal + (
        energy_right + right.pressure
    ) * right.velocity_normal

    sound_speed_left = adiabatic_sound_speed(gamma, left.density, left.pressure)
    sound_speed_right = adiabatic_sound_speed(gamma, right.density, right.pressure)
    maximum_speed = jnp.maximum(
        (jnp.abs(left.velocity_normal) + sound_speed_left),
        (jnp.abs(right.velocity_normal) + sound_speed_right),
    )

    return HydroFlux(
        mass=0.5 * (sum_mass - maximum_speed * (right.density - left.density)),
        momentum_normal=0.5
        * (
            sum_momentum_normal
            - maximum_speed * (right.density * right.velocity_normal - left.density * left.velocity_normal)
        ),
        momentum_transverse_1=0.5
        * (
            sum_momentum_transverse_1
            - maximum_speed
            * (right.density * right.velocity_transverse_1 - left.density * left.velocity_transverse_1)
        ),
        momentum_transverse_2=0.5
        * (
            sum_momentum_transverse_2
            - maximum_speed
            * (right.density * right.velocity_transverse_2 - left.density * left.velocity_transverse_2)
        ),
        energy=0.5 * (sum_energy - maximum_speed * (energy_right - energy_left)),
    )


# -------------------------------------------------------------
# ================ ↑ Hydro: HLLC, HLLE, LLF ↑ =================
# -------------------------------------------------------------
