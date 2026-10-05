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
the four operators above and is computed in
``examples/scripts/forward/mhd/turbulence/make_calibration_model.py``.
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

XAXIS, YAXIS, ZAXIS = 0, 1, 2


def _central_first_derivative(field, axis, dx):
    """Sixth-order central first derivative along ``axis``."""
    return _stencil_add(
        field,
        indices=(3, 2, 1, -1, -2, -3),
        factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
        axis=axis,
    ) / (60.0 * dx)


@partial(jax.jit, static_argnames=["config"])
def fd_ohmic_interface_rhs(bx_interface, by_interface, bz_interface, eta,
                           dt_tilde, grid_spacing, config: SimulationConfig):
    """Resistive increment of the three interface fields over ``dt_tilde``.

    Args:
        bx_interface, by_interface, bz_interface: interface magnetic fields
            (index ``i`` is the ``i+1/2`` face of the respective axis).
        eta: constant ohmic diffusivity.
        dt_tilde: stage-effective time step (the CT RHS convention).
        grid_spacing: cell size (cubic cells).
        config: simulation configuration (3D only).

    Returns:
        ``(rhs_bx, rhs_by, rhs_bz)`` to be ADDED to the CT right-hand side.
    """
    if config.dimensionality != 3:
        raise NotImplementedError("fd_ohmic_interface_rhs: 3D only")
    dx = grid_spacing

    # Cell-centred field, as the CT path derives it.
    Bx = interp_face_to_center(bx_interface, XAXIS)
    By = interp_face_to_center(by_interface, YAXIS)
    Bz = interp_face_to_center(bz_interface, ZAXIS)

    # J = curl B at cell centres.
    d = _central_first_derivative
    Jx = d(Bz, YAXIS, dx) - d(By, ZAXIS, dx)
    Jy = d(Bx, ZAXIS, dx) - d(Bz, XAXIS, dx)
    Jz = d(By, XAXIS, dx) - d(Bx, YAXIS, dx)

    # Edge-centred resistive EMF: E_x on x-edges (j+1/2, k+1/2), etc.
    Ex = eta * interp_center_to_face(interp_center_to_face(Jx, YAXIS), ZAXIS)
    Ey = eta * interp_center_to_face(interp_center_to_face(Jy, XAXIS), ZAXIS)
    Ez = eta * interp_center_to_face(interp_center_to_face(Jz, XAXIS), YAXIS)

    # dB/dt = -curl E, each component landing on its own face:
    #   dB_x/dt at (i+1/2, j, k) = -(d_y E_z - d_z E_y), with E_z on
    #   (i+1/2, j+1/2, k) and E_y on (i+1/2, j, k+1/2), so the int6 derivative
    #   along y (z) maps j+1/2 -> j (k+1/2 -> k) exactly onto the x-face.
    dtdx = dt_tilde / dx
    rhs_bx = -dtdx * (finite_difference_int6(Ez, YAXIS)
                      - finite_difference_int6(Ey, ZAXIS))
    rhs_by = -dtdx * (finite_difference_int6(Ex, ZAXIS)
                      - finite_difference_int6(Ez, XAXIS))
    rhs_bz = -dtdx * (finite_difference_int6(Ey, XAXIS)
                      - finite_difference_int6(Ex, YAXIS))
    return rhs_bx, rhs_by, rhs_bz
