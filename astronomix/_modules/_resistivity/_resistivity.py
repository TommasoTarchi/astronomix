"""
Explicit ohmic resistivity for the finite-difference constrained-transport MHD.

Faraday's law with a resistive electric field ``E = eta J``, ``J = curl B``,

    dB/dt = -curl(eta J) = eta laplacian(B)      (for div B = 0, constant eta),

is applied to the INTERFACE magnetic fields as the discrete curl of an
edge-centred electromotive force, exactly as the ideal CT update applies the
curl of its edge EMF. The update is therefore a curl and keeps the discrete
divergence of the interface field at zero to floating-point precision, whereas
adding ``eta laplacian(B)`` to the cell-centred field would not.

Discretisation, chosen to mirror the ideal CT path and its operators:

* the cell-centred field is the 6th-order face-to-centre interpolation of the
  interface field (``interp_face_to_center``, as ``update_cell_center_fields``);
* ``J`` is its curl by 6th-order central differences (the same stencil the
  viscous stress uses);
* ``E = eta J`` is carried to the cell EDGES by two 4th-order centre-to-face
  interpolations (``interp_center_to_face``, as the ideal EMF pieces are), so
  ``E_z`` lives on z-edges ``(i+1/2, j+1/2, k)`` and so on;
* the curl of the edge EMF is taken with ``finite_difference_int6``, which maps
  interface-indexed values to the derivative at the cell index -- the same
  operator the CT curl uses -- landing each component on its own face.

Only the induction equation is touched. Ohmic heating ``eta J^2`` is NOT added
to the energy equation, so the module is restricted to the isothermal EOS
(``finalize_config`` enforces this); for an ideal gas the heating term would
have to be added to the conserved energy RHS.

The spectral footprint of this operator (needed when an imposed ``eta`` is to
be recovered from a spectral energy budget) is the product of the symbols of
the four operators above.
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig

# astronomix functions
from astronomix._spatial_operators._differencing import finite_difference_int6
from astronomix._spatial_operators._interpolate import (
    interp_center_to_face,
    interp_face_to_center,
)
from astronomix._stencil_operations._stencil_operations import _stencil_add

# Axes of a single field array. Unlike ``simulation_config.XAXIS`` (= 1), which
# counts the leading variable axis of the state array, these index the spatial
# axes of one field directly, as in the constrained-transport module.
XAXIS = 0
YAXIS = 1
ZAXIS = 2


def _central_first_derivative(field, axis, grid_spacing):
    """Sixth-order central first derivative along ``axis``."""
    return _stencil_add(
        field,
        indices=(3, 2, 1, -1, -2, -3),
        factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
        axis=axis,
    ) / (60.0 * grid_spacing)


@partial(jax.jit, static_argnames=["config"])
def fd_ohmic_interface_rhs(
    bx_interface: jnp.ndarray,
    by_interface: jnp.ndarray,
    bz_interface: jnp.ndarray,
    eta,
    dt_tilde,
    grid_spacing: float,
    config: SimulationConfig,
):
    """
    Resistive increment of the three interface fields over ``dt_tilde``.

    Args:
        bx_interface: The interface magnetic field B_x (index ``i`` is the
            ``i+1/2`` face along x).
        by_interface: The interface magnetic field B_y (index ``j`` is the
            ``j+1/2`` face along y).
        bz_interface: The interface magnetic field B_z (index ``k`` is the
            ``k+1/2`` face along z).
        eta: The constant ohmic diffusivity.
        dt_tilde: The stage-effective time step (the CT right-hand-side
            convention).
        grid_spacing: The cell size (cubic cells).
        config: The simulation configuration (3D only).

    Returns:
        ``(rhs_bx, rhs_by, rhs_bz)`` to be ADDED to the CT right-hand side.
    """
    if config.dimensionality != 3:
        raise NotImplementedError("fd_ohmic_interface_rhs: 3D only")

    # Cell-centred field, as the CT path derives it.
    magnetic_field_x = interp_face_to_center(bx_interface, XAXIS)
    magnetic_field_y = interp_face_to_center(by_interface, YAXIS)
    magnetic_field_z = interp_face_to_center(bz_interface, ZAXIS)

    # J = curl B at cell centres.
    current_x = (
        _central_first_derivative(magnetic_field_z, YAXIS, grid_spacing)
        - _central_first_derivative(magnetic_field_y, ZAXIS, grid_spacing)
    )
    current_y = (
        _central_first_derivative(magnetic_field_x, ZAXIS, grid_spacing)
        - _central_first_derivative(magnetic_field_z, XAXIS, grid_spacing)
    )
    current_z = (
        _central_first_derivative(magnetic_field_y, XAXIS, grid_spacing)
        - _central_first_derivative(magnetic_field_x, YAXIS, grid_spacing)
    )

    # Edge-centred resistive EMF: E_x on x-edges (j+1/2, k+1/2), etc.
    emf_x = eta * interp_center_to_face(interp_center_to_face(current_x, YAXIS), ZAXIS)
    emf_y = eta * interp_center_to_face(interp_center_to_face(current_y, XAXIS), ZAXIS)
    emf_z = eta * interp_center_to_face(interp_center_to_face(current_z, XAXIS), YAXIS)

    # dB/dt = -curl E, each component landing on its own face:
    #   dB_x/dt at (i+1/2, j, k) = -(d_y E_z - d_z E_y), with E_z on
    #   (i+1/2, j+1/2, k) and E_y on (i+1/2, j, k+1/2), so the int6 derivative
    #   along y (z) maps j+1/2 -> j (k+1/2 -> k) exactly onto the x-face.
    dt_over_dx = dt_tilde / grid_spacing
    rhs_bx = -dt_over_dx * (
        finite_difference_int6(emf_z, YAXIS)
        - finite_difference_int6(emf_y, ZAXIS)
    )
    rhs_by = -dt_over_dx * (
        finite_difference_int6(emf_x, ZAXIS)
        - finite_difference_int6(emf_z, XAXIS)
    )
    rhs_bz = -dt_over_dx * (
        finite_difference_int6(emf_y, XAXIS)
        - finite_difference_int6(emf_x, YAXIS)
    )
    return rhs_bx, rhs_by, rhs_bz
