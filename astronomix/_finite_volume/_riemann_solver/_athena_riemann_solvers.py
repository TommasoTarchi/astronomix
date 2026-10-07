"""
AthenaPK's Riemann solvers for the VL2 finite-volume scheme.

Transcriptions of the interface solvers of AthenaPK (and hence Athena++): HLLD
(Miyoshi & Kusano 2005), HLLE and local Lax-Friedrichs for GLM-MHD, and HLLC,
HLLE and local Lax-Friedrichs for adiabatic hydrodynamics. The GLM-MHD solvers
first solve the decoupled ``(B_normal, psi)`` subsystem exactly (Mignone &
Tzeferacos 2010, eq. 24) and use the resulting interface normal field in the
MHD solver. The expressions follow AthenaPK's formulation, including its
special cases (the HLLD degeneracy check, the wave-speed clamps), and the
scheme agrees with AthenaPK to round-off
(``pytests/mhd/vl2_athenapk_regression.py``). These solvers serve the VL2
scheme only; the classic finite-volume schemes use ``hll.py``,
``_lax_friedrichs.py`` and the dispatch in ``_riemann_solver.py``.

Every function is purely elementwise on its operands — it never indexes,
reduces or reshapes — so the same code is evaluated on whole arrays by the
native JAX path and on register tiles inside the Pallas kernels.

All states are given in the frame of the interface: ``velocity_normal`` and
``field_normal`` are the components along the flux direction and the two
transverse components follow in AthenaPK's cyclic order (for an x-interface:
y then z; for a y-interface: z then x; for a z-interface: x then y).
"""

# typing
from typing import (
    Any,
    NamedTuple,
)

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
    """
    Interface flux of the GLM-MHD conserved variables (interface frame).

    The fields are ordered like :class:`MHDFaceState`; the integrator scatters
    them back to state order by position.
    """

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
    """
    Interface flux of the hydrodynamic conserved variables (interface frame).

    The fields are ordered like :class:`HydroFaceState`; the integrator
    scatters them back to state order by position.
    """

    mass: Any
    momentum_normal: Any
    momentum_transverse_1: Any
    momentum_transverse_2: Any
    energy: Any


# -------------------------------------------------------------
# ======================= ↑ Containers ↑ ======================
# -------------------------------------------------------------

#: Relative threshold of the HLLD degeneracy check (AthenaPK's ``SMALL_NUMBER``).
_HLLD_SMALL_NUMBER = 1.0e-8

#: Parthenon's ``TINY_NUMBER``: the outer wave speeds of the hydro HLLE / HLLC
#: solvers are clamped to at least this magnitude.
_TINY_NUMBER = 1.0e-20


def _square(value):
    """Return ``value * value``."""
    return value * value


def fast_magnetosonic_speed(
    gamma,
    density,
    pressure,
    field_normal,
    field_transverse_1,
    field_transverse_2,
):
    """
    Fast magnetosonic speed along the ``field_normal`` direction (AthenaPK:
    ``AdiabaticGLMMHDEOS::FastMagnetosonicSpeed``).

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
    """Return the adiabatic sound speed ``sqrt(gamma * p / rho)``."""
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
    interface_field_normal = 0.5 * (left.field_normal + right.field_normal) - (
        0.5 / cleaning_speed * (right.psi - left.psi)
    )
    interface_psi = 0.5 * (left.psi + right.psi) - (
        0.5 * cleaning_speed * (right.field_normal - left.field_normal)
    )
    return interface_field_normal, interface_psi


# -------------------------------------------------------------
# ===================== ↓ GLM-MHD: HLLD ↓ =====================
# -------------------------------------------------------------


class _MHDConserved(NamedTuple):
    """
    The seven conserved quantities HLLD carries across its wave fan (the
    normal field and psi are handled by the GLM subsystem).
    """

    density: Any
    momentum_normal: Any
    momentum_transverse_1: Any
    momentum_transverse_2: Any
    energy: Any
    field_transverse_1: Any
    field_transverse_2: Any


def _conserved_sum(first: _MHDConserved, second: _MHDConserved) -> _MHDConserved:
    """Return the component-wise sum ``first + second``."""
    return _MHDConserved(
        *(first_value + second_value for first_value, second_value in zip(first, second))
    )


def _scaled_jump(speed, upper: _MHDConserved, lower: _MHDConserved) -> _MHDConserved:
    """
    Return the wave jump ``S_k * (U_k - U_{k-1})`` that turns the flux of one
    region of the wave fan into the flux of the next (``F*_L = F_L + S_L
    (U*_L - U_L)``).
    """
    return _MHDConserved(
        *(speed * (upper_value - lower_value) for upper_value, lower_value in zip(upper, lower))
    )


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

    # The MHD fan uses the normal field of the exact GLM interface solution;
    # the reconstructed ``left.field_normal`` / ``right.field_normal`` only
    # enter the fast speeds below.
    interface_field_normal, interface_psi = _glm_interface_field_and_psi(
        left,
        right,
        cleaning_speed,
    )
    interface_field_normal_squared = interface_field_normal * interface_field_normal

    # The transverse components are grouped first for floating-point
    # associativity symmetry between the left and right states.
    magnetic_pressure_left = 0.5 * (
        interface_field_normal_squared
        + (_square(left.field_transverse_1) + _square(left.field_transverse_2))
    )
    magnetic_pressure_right = 0.5 * (
        interface_field_normal_squared
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
        energy=(
            left.pressure * inverse_gamma_minus_one
            + kinetic_energy_left
            + magnetic_pressure_left
        ),
        field_transverse_1=left.field_transverse_1,
        field_transverse_2=left.field_transverse_2,
    )
    conserved_right = _MHDConserved(
        density=right.density,
        momentum_normal=right.velocity_normal * right.density,
        momentum_transverse_1=right.velocity_transverse_1 * right.density,
        momentum_transverse_2=right.velocity_transverse_2 * right.density,
        energy=(
            right.pressure * inverse_gamma_minus_one
            + kinetic_energy_right
            + magnetic_pressure_right
        ),
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
        """The physical flux ``F(U)`` of one outer state."""
        return _MHDConserved(
            density=conserved.momentum_normal,
            momentum_normal=(
                conserved.momentum_normal * state.velocity_normal
                + total_pressure
                - interface_field_normal_squared
            ),
            momentum_transverse_1=(
                conserved.momentum_transverse_1 * state.velocity_normal
                - interface_field_normal * conserved.field_transverse_1
            ),
            momentum_transverse_2=(
                conserved.momentum_transverse_2 * state.velocity_normal
                - interface_field_normal * conserved.field_transverse_2
            ),
            energy=(
                state.velocity_normal
                * (conserved.energy + total_pressure - interface_field_normal_squared)
                - interface_field_normal
                * (
                    state.velocity_transverse_1 * conserved.field_transverse_1
                    + state.velocity_transverse_2 * conserved.field_transverse_2
                )
            ),
            field_transverse_1=(
                conserved.field_transverse_1 * state.velocity_normal
                - interface_field_normal * state.velocity_transverse_1
            ),
            field_transverse_2=(
                conserved.field_transverse_2 * state.velocity_normal
                - interface_field_normal * state.velocity_transverse_2
            ),
        )

    flux_left = side_flux(left, conserved_left, total_pressure_left)
    flux_right = side_flux(right, conserved_right, total_pressure_right)

    # --------------- ↑ Outer wave speeds and left/right fluxes ↑ ----------------

    # --------------- ↓ Contact and Alfvén speeds ↓ ----------------

    speed_minus_velocity_left = speed_left_outer - left.velocity_normal
    speed_minus_velocity_right = speed_right_outer - right.velocity_normal

    # The contact speed S_M (Miyoshi & Kusano 2005, eq. 38); the pressure terms
    # are grouped for floating-point associativity symmetry.
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

    # The star-state densities follow from mass conservation across the outer
    # waves (eq. 43).
    star_density_left = (
        conserved_left.density * speed_minus_velocity_left * inverse_speed_minus_contact_left
    )
    star_density_right = (
        conserved_right.density * speed_minus_velocity_right * inverse_speed_minus_contact_right
    )
    inverse_star_density_left = 1.0 / star_density_left
    inverse_star_density_right = 1.0 / star_density_right
    sqrt_star_density_left = jnp.sqrt(star_density_left)
    sqrt_star_density_right = jnp.sqrt(star_density_right)

    # The rotational (Alfvén) waves move at S_M -+ |B_n| / sqrt(rho*) (eq. 51).
    speed_left_alfven = contact_speed - jnp.abs(interface_field_normal) / sqrt_star_density_left
    speed_right_alfven = contact_speed + jnp.abs(interface_field_normal) / sqrt_star_density_right

    # --------------- ↑ Contact and Alfvén speeds ↑ ----------------

    # --------------- ↓ Star (*) states ↓ ----------------

    # The total pressure is constant across the star region (eq. 41); its two
    # one-sided estimates agree analytically and are averaged.
    star_total_pressure_left = total_pressure_left + (
        conserved_left.density
        * speed_minus_velocity_left
        * (contact_speed - left.velocity_normal)
    )
    star_total_pressure_right = total_pressure_right + (
        conserved_right.density
        * speed_minus_velocity_right
        * (contact_speed - right.velocity_normal)
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
        """
        The star state between an outer fast wave and the adjacent Alfvén wave
        (Miyoshi & Kusano 2005, eqs. 43-48).

        Args:
            state: The primitive outer state.
            conserved: The conserved outer state.
            total_pressure: The total (gas + magnetic) pressure of the outer state.
            star_density: The star-state density.
            inverse_star_density: ``1 / star_density``.
            speed_minus_velocity: ``S_outer - v_normal`` of the outer state.
            speed_minus_contact: ``S_outer - S_M``.
            inverse_speed_minus_contact: ``1 / (S_outer - S_M)``.

        Returns:
            The conserved star state and its ``v* . B*``.
        """
        star_momentum_normal = star_density * contact_speed

        # Eqs. (44)-(47), with the Athena++ guard against the degenerate case
        # in which the denominator vanishes (the star state then keeps the
        # transverse velocity and field of the outer state).
        denominator = (
            conserved.density * speed_minus_velocity * speed_minus_contact
            - interface_field_normal_squared
        )
        degenerate = jnp.abs(denominator) < _HLLD_SMALL_NUMBER * star_total_pressure
        safe_denominator = jnp.where(degenerate, 1.0, denominator)

        velocity_factor = (
            interface_field_normal * (speed_minus_velocity - speed_minus_contact) / safe_denominator
        )
        field_factor = (
            conserved.density * _square(speed_minus_velocity) - interface_field_normal_squared
        ) / safe_denominator

        star_momentum_transverse_1 = jnp.where(
            degenerate,
            star_density * state.velocity_transverse_1,
            star_density
            * (state.velocity_transverse_1 - conserved.field_transverse_1 * velocity_factor),
        )
        star_momentum_transverse_2 = jnp.where(
            degenerate,
            star_density * state.velocity_transverse_2,
            star_density
            * (state.velocity_transverse_2 - conserved.field_transverse_2 * velocity_factor),
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
            star_momentum_normal * interface_field_normal
            + (
                star_momentum_transverse_1 * star_field_transverse_1
                + star_momentum_transverse_2 * star_field_transverse_2
            )
        ) * inverse_star_density

        # The star energy follows from energy conservation across the outer
        # wave (eq. 48).
        star_energy = (
            speed_minus_velocity * conserved.energy
            - total_pressure * state.velocity_normal
            + star_total_pressure * contact_speed
            + interface_field_normal
            * (
                state.velocity_normal * interface_field_normal
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
    ones = jnp.ones_like(interface_field_normal)
    field_normal_sign = jnp.where(interface_field_normal > 0.0, ones, -ones)

    # The two double-star states share their transverse velocity (eqs. 59, 60)
    # and transverse field (eqs. 61, 62).
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

    # The double-star energies follow from energy conservation across the
    # Alfvén waves (eq. 63).
    double_star_velocity_dot_field = contact_speed * interface_field_normal + (
        double_star_left_momentum_transverse_1 * double_star_field_transverse_1
        + double_star_left_momentum_transverse_2 * double_star_field_transverse_2
    ) / star_left.density

    double_star_left = _MHDConserved(
        density=star_left.density,
        momentum_normal=star_left.momentum_normal,
        momentum_transverse_1=double_star_left_momentum_transverse_1,
        momentum_transverse_2=double_star_left_momentum_transverse_2,
        energy=star_left.energy
        - sqrt_star_density_left
        * field_normal_sign
        * (star_velocity_dot_field_left - double_star_velocity_dot_field),
        field_transverse_1=double_star_field_transverse_1,
        field_transverse_2=double_star_field_transverse_2,
    )
    double_star_right = _MHDConserved(
        density=star_right.density,
        momentum_normal=star_right.momentum_normal,
        momentum_transverse_1=star_right.density * double_star_velocity_transverse_1,
        momentum_transverse_2=star_right.density * double_star_velocity_transverse_2,
        energy=star_right.energy
        + sqrt_star_density_right
        * field_normal_sign
        * (star_velocity_dot_field_right - double_star_velocity_dot_field),
        field_transverse_1=double_star_field_transverse_1,
        field_transverse_2=double_star_field_transverse_2,
    )

    # --------------- ↑ Double-star (**) states ↑ ----------------

    # --------------- ↓ Flux selection ↓ ----------------

    # The flux of each region of the wave fan is the flux of its outer
    # neighbour plus the jump S_k * (U_k - U_{k-1}) across the separating wave.
    jump_left_alfven = _scaled_jump(speed_left_alfven, double_star_left, star_left)
    jump_left_outer = _scaled_jump(speed_left_outer, star_left, conserved_left)
    jump_right_alfven = _scaled_jump(speed_right_alfven, double_star_right, star_right)
    jump_right_outer = _scaled_jump(speed_right_outer, star_right, conserved_right)

    flux_left_star = _conserved_sum(flux_left, jump_left_outer)
    flux_right_star = _conserved_sum(flux_right, jump_right_outer)
    flux_left_double_star = _conserved_sum(flux_left_star, jump_left_alfven)
    flux_right_double_star = _conserved_sum(flux_right_star, jump_right_alfven)

    def select_upwind_flux(
        left_flux,
        left_star_flux,
        left_double_star_flux,
        right_double_star_flux,
        right_star_flux,
        right_flux,
    ):
        """
        Pick (for one component) the flux of the wave-fan region that contains
        the interface. Later selections take precedence, so the inner waves are
        tested first and the outer waves last.
        """
        selected = jnp.where(contact_speed >= 0.0, left_double_star_flux, right_double_star_flux)
        selected = jnp.where(speed_right_alfven <= 0.0, right_star_flux, selected)
        selected = jnp.where(speed_left_alfven >= 0.0, left_star_flux, selected)
        selected = jnp.where(speed_right_outer <= 0.0, right_flux, selected)
        selected = jnp.where(speed_left_outer >= 0.0, left_flux, selected)
        return selected

    selected_flux = _MHDConserved(
        *(
            select_upwind_flux(*region_fluxes)
            for region_fluxes in zip(
                flux_left,
                flux_left_star,
                flux_left_double_star,
                flux_right_double_star,
                flux_right_star,
                flux_right,
            )
        )
    )

    # --------------- ↑ Flux selection ↑ ----------------

    # The decoupled GLM subsystem contributes F(B_normal) = psi* and
    # F(psi) = c_h^2 B_normal* (Dedner et al. 2002); the MHD solver does not
    # touch these components.
    return MHDFlux(
        mass=selected_flux.density,
        momentum_normal=selected_flux.momentum_normal,
        momentum_transverse_1=selected_flux.momentum_transverse_1,
        momentum_transverse_2=selected_flux.momentum_transverse_2,
        energy=selected_flux.energy,
        field_normal=interface_psi,
        field_transverse_1=selected_flux.field_transverse_1,
        field_transverse_2=selected_flux.field_transverse_2,
        psi=cleaning_speed * cleaning_speed * interface_field_normal,
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
    gamma_minus_two = gamma_minus_one - 1.0

    # The fluxes use the normal field of the exact GLM interface solution; the
    # reconstructed ``left.field_normal`` / ``right.field_normal`` only enter
    # the fast speeds below.
    interface_field_normal, interface_psi = _glm_interface_field_and_psi(
        left,
        right,
        cleaning_speed,
    )

    # --------------- ↓ Roe-averaged state ↓ ----------------

    sqrt_density_left = jnp.sqrt(left.density)
    sqrt_density_right = jnp.sqrt(right.density)
    inverse_sum_sqrt_density = 1.0 / (sqrt_density_left + sqrt_density_right)

    roe_density = sqrt_density_left * sqrt_density_right
    roe_velocity_normal = (
        sqrt_density_left * left.velocity_normal + sqrt_density_right * right.velocity_normal
    ) * inverse_sum_sqrt_density
    roe_velocity_transverse_1 = (
        sqrt_density_left * left.velocity_transverse_1
        + sqrt_density_right * right.velocity_transverse_1
    ) * inverse_sum_sqrt_density
    roe_velocity_transverse_2 = (
        sqrt_density_left * left.velocity_transverse_2
        + sqrt_density_right * right.velocity_transverse_2
    ) * inverse_sum_sqrt_density
    # The Roe average of the field weights the sides the other way round.
    roe_field_transverse_1 = (
        sqrt_density_right * left.field_transverse_1
        + sqrt_density_left * right.field_transverse_1
    ) * inverse_sum_sqrt_density
    roe_field_transverse_2 = (
        sqrt_density_right * left.field_transverse_2
        + sqrt_density_left * right.field_transverse_2
    ) * inverse_sum_sqrt_density

    # The terms X (transverse field jump) and Y (density ratio) of the
    # Roe-averaged MHD eigensystem (Stone et al. 2008, appendix B).
    transverse_field_jump_term = (
        0.5
        * (
            _square(left.field_transverse_1 - right.field_transverse_1)
            + _square(left.field_transverse_2 - right.field_transverse_2)
        )
        / _square(sqrt_density_left + sqrt_density_right)
    )
    density_ratio_term = 0.5 * (left.density + right.density) / roe_density

    # Roe (1981): average the enthalpy H = (E + P) / rho rather than E or P.
    magnetic_pressure_left = 0.5 * (
        _square(interface_field_normal)
        + _square(left.field_transverse_1)
        + _square(left.field_transverse_2)
    )
    magnetic_pressure_right = 0.5 * (
        _square(interface_field_normal)
        + _square(right.field_transverse_1)
        + _square(right.field_transverse_2)
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
    normal_alfven_speed_squared = _square(interface_field_normal) / roe_density
    transverse_field_star_squared = (
        gamma_minus_one - gamma_minus_two * density_ratio_term
    ) * roe_transverse_field_squared
    enthalpy_minus_magnetic = roe_enthalpy - (
        normal_alfven_speed_squared + roe_transverse_field_squared / roe_density
    )
    roe_velocity_squared = (
        _square(roe_velocity_normal)
        + _square(roe_velocity_transverse_1)
        + _square(roe_velocity_transverse_2)
    )
    sound_speed_squared_tilde = jnp.maximum(
        gamma_minus_one * (enthalpy_minus_magnetic - 0.5 * roe_velocity_squared)
        - gamma_minus_two * transverse_field_jump_term,
        0.0,
    )
    transverse_alfven_speed_squared = transverse_field_star_squared / roe_density
    squared_speed_sum = (
        normal_alfven_speed_squared + transverse_alfven_speed_squared + sound_speed_squared_tilde
    )
    squared_speed_difference = (
        normal_alfven_speed_squared + transverse_alfven_speed_squared - sound_speed_squared_tilde
    )
    discriminant_root = jnp.sqrt(
        squared_speed_difference * squared_speed_difference
        + 4.0 * sound_speed_squared_tilde * transverse_alfven_speed_squared
    )
    roe_fast_speed = jnp.sqrt(0.5 * (squared_speed_sum + discriminant_root))

    speed_left = jnp.minimum(
        roe_velocity_normal - roe_fast_speed,
        left.velocity_normal - fast_speed_left,
    )
    speed_right = jnp.maximum(
        roe_velocity_normal + roe_fast_speed,
        right.velocity_normal + fast_speed_right,
    )

    speed_plus = jnp.where(speed_right > 0.0, speed_right, 0.0)
    speed_minus = jnp.where(speed_left < 0.0, speed_left, 0.0)

    # --------------- ↑ Wave speeds ↑ ----------------

    # --------------- ↓ HLLE flux ↓ ----------------

    # The left/right fluxes along the lines speed_minus / speed_plus:
    # F_L - S_L U_L and F_R - S_R U_R.
    relative_velocity_left = left.velocity_normal - speed_minus
    relative_velocity_right = right.velocity_normal - speed_plus

    def side_flux(state: MHDFaceState, relative_velocity, magnetic_pressure, energy):
        """``F - S U`` of one outer state, with ``relative_velocity = v_normal - S``."""
        return _MHDConserved(
            density=state.density * relative_velocity,
            momentum_normal=(
                state.density * state.velocity_normal * relative_velocity
                + magnetic_pressure
                - _square(interface_field_normal)
                + state.pressure
            ),
            momentum_transverse_1=(
                state.density * state.velocity_transverse_1 * relative_velocity
                - interface_field_normal * state.field_transverse_1
            ),
            momentum_transverse_2=(
                state.density * state.velocity_transverse_2 * relative_velocity
                - interface_field_normal * state.field_transverse_2
            ),
            energy=(
                energy * relative_velocity
                + state.velocity_normal
                * (state.pressure + magnetic_pressure - _square(interface_field_normal))
                - interface_field_normal
                * (
                    state.field_transverse_1 * state.velocity_transverse_1
                    + state.field_transverse_2 * state.velocity_transverse_2
                )
            ),
            field_transverse_1=(
                state.field_transverse_1 * relative_velocity
                - interface_field_normal * state.velocity_transverse_1
            ),
            field_transverse_2=(
                state.field_transverse_2 * relative_velocity
                - interface_field_normal * state.velocity_transverse_2
            ),
        )

    flux_left = side_flux(left, relative_velocity_left, magnetic_pressure_left, energy_left)
    flux_right = side_flux(right, relative_velocity_right, magnetic_pressure_right, energy_right)

    speeds_differ = speed_plus != speed_minus
    safe_speed_difference = jnp.where(speeds_differ, speed_plus - speed_minus, 1.0)
    weight = jnp.where(
        speeds_differ,
        0.5 * (speed_plus + speed_minus) / safe_speed_difference,
        0.0,
    )

    hlle_flux = _MHDConserved(
        *(
            0.5 * (value_left + value_right) + (value_left - value_right) * weight
            for value_left, value_right in zip(flux_left, flux_right)
        )
    )

    # --------------- ↑ HLLE flux ↑ ----------------

    # The decoupled GLM subsystem contributes F(B_normal) = psi* and
    # F(psi) = c_h^2 B_normal* (Dedner et al. 2002); the MHD solver does not
    # touch these components.
    return MHDFlux(
        mass=hlle_flux.density,
        momentum_normal=hlle_flux.momentum_normal,
        momentum_transverse_1=hlle_flux.momentum_transverse_1,
        momentum_transverse_2=hlle_flux.momentum_transverse_2,
        energy=hlle_flux.energy,
        field_normal=interface_psi,
        field_transverse_1=hlle_flux.field_transverse_1,
        field_transverse_2=hlle_flux.field_transverse_2,
        psi=cleaning_speed * cleaning_speed * interface_field_normal,
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

    # The fluxes use the normal field of the exact GLM interface solution; the
    # reconstructed ``left.field_normal`` / ``right.field_normal`` only enter
    # the fast speeds below.
    interface_field_normal, interface_psi = _glm_interface_field_and_psi(
        left,
        right,
        cleaning_speed,
    )

    # --------------- ↓ Sum of the left and right fluxes ↓ ----------------

    mass_flux_left = left.density * left.velocity_normal
    mass_flux_right = right.density * right.velocity_normal
    magnetic_term_left = 0.5 * (
        _square(left.field_transverse_1)
        + _square(left.field_transverse_2)
        - _square(interface_field_normal)
    )
    magnetic_term_right = 0.5 * (
        _square(right.field_transverse_1)
        + _square(right.field_transverse_2)
        - _square(interface_field_normal)
    )

    sum_mass = mass_flux_left + mass_flux_right
    sum_momentum_normal = (
        mass_flux_left * left.velocity_normal
        + mass_flux_right * right.velocity_normal
        + magnetic_term_left
        + magnetic_term_right
        + (left.pressure + right.pressure)
    )
    sum_momentum_transverse_1 = (
        mass_flux_left * left.velocity_transverse_1
        + mass_flux_right * right.velocity_transverse_1
        - interface_field_normal * (left.field_transverse_1 + right.field_transverse_1)
    )
    sum_momentum_transverse_2 = (
        mass_flux_left * left.velocity_transverse_2
        + mass_flux_right * right.velocity_transverse_2
        - interface_field_normal * (left.field_transverse_2 + right.field_transverse_2)
    )
    sum_field_transverse_1 = (
        left.field_transverse_1 * left.velocity_normal
        + right.field_transverse_1 * right.velocity_normal
        - interface_field_normal * (left.velocity_transverse_1 + right.velocity_transverse_1)
    )
    sum_field_transverse_2 = (
        left.field_transverse_2 * left.velocity_normal
        + right.field_transverse_2 * right.velocity_normal
        - interface_field_normal * (left.velocity_transverse_2 + right.velocity_transverse_2)
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
        + _square(interface_field_normal)
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
        + _square(interface_field_normal)
    )
    sum_energy = (
        (energy_left + left.pressure + magnetic_term_left) * left.velocity_normal
        + (energy_right + right.pressure + magnetic_term_right) * right.velocity_normal
        - interface_field_normal
        * (
            left.field_transverse_1 * left.velocity_transverse_1
            + left.field_transverse_2 * left.velocity_transverse_2
        )
        - interface_field_normal
        * (
            right.field_transverse_1 * right.velocity_transverse_1
            + right.field_transverse_2 * right.velocity_transverse_2
        )
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
        jnp.abs(left.velocity_normal) + fast_speed_left,
        jnp.abs(right.velocity_normal) + fast_speed_right,
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

    # The decoupled GLM subsystem contributes F(B_normal) = psi* and
    # F(psi) = c_h^2 B_normal* (Dedner et al. 2002); the MHD solver does not
    # touch these components.
    return MHDFlux(
        mass=0.5 * (sum_mass - jump_mass),
        momentum_normal=0.5 * (sum_momentum_normal - jump_momentum_normal),
        momentum_transverse_1=0.5 * (sum_momentum_transverse_1 - jump_momentum_transverse_1),
        momentum_transverse_2=0.5 * (sum_momentum_transverse_2 - jump_momentum_transverse_2),
        energy=0.5 * (sum_energy - jump_energy),
        field_normal=interface_psi,
        field_transverse_1=0.5 * (sum_field_transverse_1 - jump_field_transverse_1),
        field_transverse_2=0.5 * (sum_field_transverse_2 - jump_field_transverse_2),
        psi=cleaning_speed * cleaning_speed * interface_field_normal,
    )


# -------------------------------------------------------------
# ================== ↑ GLM-MHD: HLLE, LLF ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================ ↓ Hydro: HLLC, HLLE, LLF ↓ =================
# -------------------------------------------------------------


def _hydro_energy(state: HydroFaceState, inverse_gamma_minus_one):
    """Return the total energy density ``p / (gamma - 1) + rho v^2 / 2``."""
    return state.pressure * inverse_gamma_minus_one + 0.5 * state.density * (
        _square(state.velocity_normal)
        + _square(state.velocity_transverse_1)
        + _square(state.velocity_transverse_2)
    )


def _hydro_flux_minus_speed_times_state(
    state: HydroFaceState,
    relative_velocity,
    energy,
) -> HydroFlux:
    """
    Return ``F - S U`` of one outer state, the flux through a surface moving
    with the wave speed ``S``.

    Args:
        state: The primitive outer state.
        relative_velocity: ``v_normal - S``.
        energy: The total energy density of the outer state.

    Returns:
        ``F - S U`` (interface frame).
    """
    return HydroFlux(
        mass=state.density * relative_velocity,
        momentum_normal=state.density * state.velocity_normal * relative_velocity + state.pressure,
        momentum_transverse_1=state.density * state.velocity_transverse_1 * relative_velocity,
        momentum_transverse_2=state.density * state.velocity_transverse_2 * relative_velocity,
        energy=energy * relative_velocity + state.pressure * state.velocity_normal,
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

    # A shock (middle pressure above the outer one) moves faster than sound;
    # these factors correct the outer wave speeds for its strength.
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

    # The contact speed and pressure use the unclamped outer wave speeds.
    velocity_minus_speed_left = left.velocity_normal - speed_left
    velocity_minus_speed_right = right.velocity_normal - speed_right

    momentum_term_left = (
        left.pressure + velocity_minus_speed_left * left.density * left.velocity_normal
    )
    momentum_term_right = (
        right.pressure + velocity_minus_speed_right * right.density * right.velocity_normal
    )

    mass_term_left = left.density * velocity_minus_speed_left
    mass_term_right = -(right.density * velocity_minus_speed_right)

    contact_speed = (momentum_term_left - momentum_term_right) / (mass_term_left + mass_term_right)
    contact_pressure = (
        mass_term_left * momentum_term_right + mass_term_right * momentum_term_left
    ) / (mass_term_left + mass_term_right)
    contact_pressure = jnp.where(contact_pressure > 0.0, contact_pressure, 0.0)

    # --------------- ↑ Contact wave ↑ ----------------

    # --------------- ↓ HLLC flux ↓ ----------------

    # The outer fluxes use the clamped wave speeds.
    relative_velocity_left = left.velocity_normal - speed_minus
    relative_velocity_right = right.velocity_normal - speed_plus

    flux_left = _hydro_flux_minus_speed_times_state(left, relative_velocity_left, energy_left)
    flux_right = _hydro_flux_minus_speed_times_state(right, relative_velocity_right, energy_right)

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
        mass=weight_left * flux_left.mass + weight_right * flux_right.mass,
        momentum_normal=(
            weight_left * flux_left.momentum_normal
            + weight_right * flux_right.momentum_normal
            + weight_contact * contact_pressure
        ),
        momentum_transverse_1=(
            weight_left * flux_left.momentum_transverse_1
            + weight_right * flux_right.momentum_transverse_1
        ),
        momentum_transverse_2=(
            weight_left * flux_left.momentum_transverse_2
            + weight_right * flux_right.momentum_transverse_2
        ),
        energy=(
            weight_left * flux_left.energy
            + weight_right * flux_right.energy
            + weight_contact * contact_pressure * contact_speed
        ),
    )


def hlle_hydro_flux(left: HydroFaceState, right: HydroFaceState, gamma) -> HydroFlux:
    """
    The HLLE flux for adiabatic hydrodynamics with Roe-averaged (Einfeldt) wave
    speeds, as AthenaPK's ``hydro_hlle.hpp``.

    NOTE: AthenaPK clamps a non-negative left speed to ``+TINY_NUMBER`` (HLLC
    uses ``-TINY_NUMBER``); the clamp only affects the result at the level of
    1e-20, and AthenaPK's sign is kept.

    Args:
        left: The primitive state left of the interface.
        right: The primitive state right of the interface.
        gamma: The adiabatic index.

    Returns:
        The hydrodynamic interface flux.
    """
    gamma_minus_one = gamma - 1.0
    inverse_gamma_minus_one = 1.0 / gamma_minus_one

    # --------------- ↓ Roe-averaged state ↓ ----------------

    sqrt_density_left = jnp.sqrt(left.density)
    sqrt_density_right = jnp.sqrt(right.density)
    inverse_sum_sqrt_density = 1.0 / (sqrt_density_left + sqrt_density_right)

    roe_velocity_normal = (
        sqrt_density_left * left.velocity_normal + sqrt_density_right * right.velocity_normal
    ) * inverse_sum_sqrt_density
    roe_velocity_transverse_1 = (
        sqrt_density_left * left.velocity_transverse_1
        + sqrt_density_right * right.velocity_transverse_1
    ) * inverse_sum_sqrt_density
    roe_velocity_transverse_2 = (
        sqrt_density_left * left.velocity_transverse_2
        + sqrt_density_right * right.velocity_transverse_2
    ) * inverse_sum_sqrt_density

    # Roe (1981): average the enthalpy H = (E + p) / rho rather than E or p.
    energy_left = _hydro_energy(left, inverse_gamma_minus_one)
    energy_right = _hydro_energy(right, inverse_gamma_minus_one)
    roe_enthalpy = (
        (energy_left + left.pressure) / sqrt_density_left
        + (energy_right + right.pressure) / sqrt_density_right
    ) * inverse_sum_sqrt_density

    # --------------- ↑ Roe-averaged state ↑ ----------------

    # --------------- ↓ Wave speeds ↓ ----------------

    sound_speed_left = adiabatic_sound_speed(gamma, left.density, left.pressure)
    sound_speed_right = adiabatic_sound_speed(gamma, right.density, right.pressure)
    enthalpy_term = roe_enthalpy - 0.5 * (
        _square(roe_velocity_normal)
        + _square(roe_velocity_transverse_1)
        + _square(roe_velocity_transverse_2)
    )
    # The double where keeps the Roe sound speed and its derivatives finite
    # where the enthalpy term is negative: the inner where keeps the square
    # root's argument non-negative, the outer one discards the infinite
    # derivative of the square root at zero.
    roe_sound_speed = jnp.where(
        enthalpy_term < 0.0,
        0.0,
        jnp.sqrt(gamma_minus_one * jnp.where(enthalpy_term < 0.0, 0.0, enthalpy_term)),
    )

    speed_left = jnp.minimum(
        roe_velocity_normal - roe_sound_speed,
        left.velocity_normal - sound_speed_left,
    )
    speed_right = jnp.maximum(
        roe_velocity_normal + roe_sound_speed,
        right.velocity_normal + sound_speed_right,
    )

    speed_plus = jnp.where(speed_right > 0.0, speed_right, _TINY_NUMBER)
    speed_minus = jnp.where(speed_left < 0.0, speed_left, _TINY_NUMBER)

    # --------------- ↑ Wave speeds ↑ ----------------

    # --------------- ↓ HLLE flux ↓ ----------------

    relative_velocity_left = left.velocity_normal - speed_minus
    relative_velocity_right = right.velocity_normal - speed_plus

    flux_left = _hydro_flux_minus_speed_times_state(left, relative_velocity_left, energy_left)
    flux_right = _hydro_flux_minus_speed_times_state(right, relative_velocity_right, energy_right)

    speeds_differ = speed_plus != speed_minus
    safe_speed_difference = jnp.where(speeds_differ, speed_plus - speed_minus, 1.0)
    weight = jnp.where(
        speeds_differ,
        0.5 * (speed_plus + speed_minus) / safe_speed_difference,
        0.0,
    )

    # --------------- ↑ HLLE flux ↑ ----------------

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

    # --------------- ↓ Sum of the left and right fluxes ↓ ----------------

    mass_flux_left = left.density * left.velocity_normal
    mass_flux_right = right.density * right.velocity_normal

    sum_mass = mass_flux_left + mass_flux_right
    sum_momentum_normal = (
        mass_flux_left * left.velocity_normal
        + mass_flux_right * right.velocity_normal
        + (left.pressure + right.pressure)
    )
    sum_momentum_transverse_1 = (
        mass_flux_left * left.velocity_transverse_1
        + mass_flux_right * right.velocity_transverse_1
    )
    sum_momentum_transverse_2 = (
        mass_flux_left * left.velocity_transverse_2
        + mass_flux_right * right.velocity_transverse_2
    )

    energy_left = _hydro_energy(left, inverse_gamma_minus_one)
    energy_right = _hydro_energy(right, inverse_gamma_minus_one)
    sum_energy = (energy_left + left.pressure) * left.velocity_normal + (
        energy_right + right.pressure
    ) * right.velocity_normal

    # --------------- ↑ Sum of the left and right fluxes ↑ ----------------

    # --------------- ↓ Dissipation ↓ ----------------

    sound_speed_left = adiabatic_sound_speed(gamma, left.density, left.pressure)
    sound_speed_right = adiabatic_sound_speed(gamma, right.density, right.pressure)
    maximum_speed = jnp.maximum(
        jnp.abs(left.velocity_normal) + sound_speed_left,
        jnp.abs(right.velocity_normal) + sound_speed_right,
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

    # --------------- ↑ Dissipation ↑ ----------------

    return HydroFlux(
        mass=0.5 * (sum_mass - jump_mass),
        momentum_normal=0.5 * (sum_momentum_normal - jump_momentum_normal),
        momentum_transverse_1=0.5 * (sum_momentum_transverse_1 - jump_momentum_transverse_1),
        momentum_transverse_2=0.5 * (sum_momentum_transverse_2 - jump_momentum_transverse_2),
        energy=0.5 * (sum_energy - jump_energy),
    )


# -------------------------------------------------------------
# ================ ↑ Hydro: HLLC, HLLE, LLF ↑ =================
# -------------------------------------------------------------
