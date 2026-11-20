"""
The isothermal collapse setups that validate the sink particles
(``astronomix/test_setups/self_gravity``): each builds the initial state its
docstring describes. The runs themselves are in
``examples/scripts/validation/sink_particles/hard_baseline.py``.
"""

# numerics
import numpy as np

# testing
import pytest

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import ISOTHERMAL

# astronomix containers
from astronomix import SimulationConfig, SimulationParams, SinkParticleConfig

# astronomix functions
from astronomix import get_registered_variables
from astronomix.test_setups.self_gravity.bonnor_ebert_sphere3D import (
    BonnorEbertSettings,
    bonnor_ebert_mass,
    bonnor_ebert_radius,
    setup_bonnor_ebert_sphere,
)
from astronomix.test_setups.self_gravity.rotating_cloud_core3D import (
    RotatingCloudCoreSettings,
    setup_rotating_cloud_core,
)
from astronomix.test_setups.self_gravity.singular_isothermal_sphere3D import (
    SingularIsothermalSphereSettings,
    setup_singular_isothermal_sphere,
)
from astronomix.test_setups.self_gravity.turbulent_cluster3D import (
    TurbulentClusterSettings,
    setup_turbulent_cluster,
)

pytestmark = pytest.mark.x64

NUM_CELLS = 32
CELL_SIZE = 1.0 / NUM_CELLS

#: The middle cell, on whose centre the spheres are centred.
CENTRAL_CELL = (NUM_CELLS // 2,) * 3


def _build(setup, settings):
    """Build a setup at 32³ with sink particles on.

    Returns:
        ``(density, velocity, config)``: the density (n, n, n), the velocity
        (n, n, n, 3) and the finalized configuration.
    """
    config = SimulationConfig(
        num_cells=NUM_CELLS,
        progress_bar=False,
        sink_particle_config=SinkParticleConfig(sink_particles=True),
    )
    state, config, _ = setup(config, SimulationParams(), settings)
    registered_variables = get_registered_variables(config)
    primitive_state = state.primitive_state
    assert jnp.all(jnp.isfinite(primitive_state)), "the initial state is not finite"
    assert config.equation_of_state == ISOTHERMAL, "the setup is not isothermal"
    density = primitive_state[registered_variables.density_index]
    velocity = jnp.stack(
        [primitive_state[index] for index in registered_variables.velocity_index],
        axis=-1,
    )
    return density, velocity, config


def _distance_from_central_cell():
    """The distance of every cell centre from the centre of the middle cell."""
    centers = (np.arange(NUM_CELLS) + 0.5) * CELL_SIZE
    x, y, z = np.meshgrid(centers, centers, centers, indexing="ij")
    middle = (NUM_CELLS // 2 + 0.5) * CELL_SIZE
    return np.sqrt((x - middle) ** 2 + (y - middle) ** 2 + (z - middle) ** 2)


def test_lane_emden_solution_gives_critical_bonnor_ebert_sphere():
    """The tabulated Lane–Emden solution gives the critical Bonnor–Ebert
    sphere: density contrast 14.04 at ξ = 6.451, and the dimensionless mass
    M G^(3/2) P^(1/2) / c_s⁴ = 1.182 at the external pressure P = ρ(R) c_s²."""
    contrast = 14.04
    edge_radius = float(bonnor_ebert_radius(contrast))
    assert abs(edge_radius - 6.451) < 5e-3, f"ξ(R) at contrast 14.04 is {edge_radius}, not 6.451"

    cloud_radius = 1.0
    mass = float(bonnor_ebert_mass(contrast, cloud_radius, 1.0, 1.0))
    central_density = edge_radius**2 / (4.0 * np.pi * cloud_radius**2)
    external_pressure = central_density / contrast
    dimensionless_mass = mass * np.sqrt(external_pressure)
    np.testing.assert_allclose(
        dimensionless_mass,
        1.182,
        rtol=2e-3,
        err_msg="the mass of the critical Bonnor–Ebert sphere is not 1.182 c_s⁴ / (G^(3/2) P^(1/2))",
    )


def test_bonnor_ebert_sphere_profile():
    """The Bonnor–Ebert sphere peaks at its enhanced central density f ρ_c in
    the middle cell, and the gas inside R holds f times the mass of the
    sphere."""
    settings = BonnorEbertSettings()
    density, velocity, _ = _build(setup_bonnor_ebert_sphere, settings)

    edge_radius = float(bonnor_ebert_radius(settings.density_contrast))
    central_density = (
        settings.density_enhancement
        * edge_radius**2
        / (4.0 * np.pi * settings.cloud_radius**2)
    )
    peak = np.unravel_index(int(jnp.argmax(density)), density.shape)
    assert peak == CENTRAL_CELL, f"the density peaks in cell {peak}, not the middle cell"
    np.testing.assert_allclose(
        density[CENTRAL_CELL], central_density, rtol=1e-12,
        err_msg="the middle cell does not hold the enhanced central density",
    )

    inside = _distance_from_central_cell() <= settings.cloud_radius
    mass_inside = float(jnp.sum(jnp.where(inside, density, 0.0))) * CELL_SIZE**3
    expected_mass = settings.density_enhancement * float(
        bonnor_ebert_mass(settings.density_contrast, settings.cloud_radius, 1.0, 1.0)
    )
    np.testing.assert_allclose(
        mass_inside, expected_mass, rtol=0.02,
        err_msg="the gas inside R does not hold f times the mass of the Bonnor–Ebert sphere",
    )
    assert jnp.all(velocity == 0.0), "the gas of the Bonnor–Ebert sphere is not at rest"


def test_singular_isothermal_sphere_profile():
    """The singular isothermal sphere peaks in the middle cell, and the gas
    inside R holds (2 A c_s² / G) (R − r_c arctan(R / r_c))."""
    settings = SingularIsothermalSphereSettings()
    density, velocity, _ = _build(setup_singular_isothermal_sphere, settings)

    peak = np.unravel_index(int(jnp.argmax(density)), density.shape)
    assert peak == CENTRAL_CELL, f"the density peaks in cell {peak}, not the middle cell"

    core_radius = settings.core_radius_in_cells * CELL_SIZE
    expected_mass = 2.0 * settings.overdensity_factor * (
        settings.cloud_radius
        - core_radius * np.arctan(settings.cloud_radius / core_radius)
    )
    inside = _distance_from_central_cell() <= settings.cloud_radius
    mass_inside = float(jnp.sum(jnp.where(inside, density, 0.0))) * CELL_SIZE**3
    np.testing.assert_allclose(
        mass_inside, expected_mass, rtol=0.03,
        err_msg="the gas inside R does not hold the mass of the singular isothermal sphere",
    )
    assert jnp.all(velocity == 0.0), "the gas of the singular isothermal sphere is not at rest"


def test_rotating_cloud_core_rotation_and_perturbation():
    """The rotating core turns as a solid body about z inside R and is at
    rest outside; its density is ρ_0 (1 ± a) along x and y; the defaults
    give α ≈ 0.26 and β ≈ 0.16."""
    settings = RotatingCloudCoreSettings()
    density, velocity, _ = _build(setup_rotating_cloud_core, settings)

    centers = (np.arange(NUM_CELLS) + 0.5) * CELL_SIZE - (NUM_CELLS // 2 + 0.5) * CELL_SIZE
    x, y, _ = np.meshgrid(centers, centers, centers, indexing="ij")
    inside = _distance_from_central_cell() <= settings.cloud_radius
    expected_velocity = np.where(
        inside[..., None],
        settings.rotation_rate * np.stack([-y, x, np.zeros_like(x)], axis=-1),
        0.0,
    )
    np.testing.assert_allclose(
        velocity, expected_velocity, rtol=1e-12, atol=1e-12,
        err_msg="the velocity is not a solid-body rotation inside R and zero outside",
    )

    middle = NUM_CELLS // 2
    along_x = density[middle + 4, middle, middle]
    along_y = density[middle, middle + 4, middle]
    np.testing.assert_allclose(
        [along_x, along_y],
        [settings.cloud_density * 1.1, settings.cloud_density * 0.9],
        rtol=1e-12,
        err_msg="the m = 2 perturbation is not ρ_0 (1 + a) along x and ρ_0 (1 − a) along y",
    )

    thermal_ratio = 15.0 / (8.0 * np.pi * settings.cloud_density * settings.cloud_radius**2)
    rotational_ratio = settings.rotation_rate**2 / (4.0 * np.pi * settings.cloud_density)
    np.testing.assert_allclose(
        [thermal_ratio, rotational_ratio], [0.26, 0.16], rtol=0.01,
        err_msg="the defaults do not give α ≈ 0.26 and β ≈ 0.16",
    )


def test_turbulent_cluster_velocity_field():
    """The turbulent cluster is uniform, its velocity has zero mean and the
    rms Mach number of the settings, and the Mach number scales the pattern
    without changing it."""
    settings = TurbulentClusterSettings()
    density, velocity, _ = _build(setup_turbulent_cluster, settings)

    np.testing.assert_allclose(
        density, settings.mean_density, rtol=1e-14, err_msg="the density is not uniform"
    )
    mach_number = float(jnp.sqrt(jnp.mean(jnp.sum(velocity**2, axis=-1))))
    np.testing.assert_allclose(
        mach_number, settings.mach_number, rtol=1e-12,
        err_msg="the rms Mach number differs from the settings",
    )
    mean_velocity = np.asarray(jnp.mean(velocity, axis=(0, 1, 2)))
    np.testing.assert_allclose(
        mean_velocity, 0.0, atol=1e-12, err_msg="the velocity field has a mean"
    )

    _, doubled_velocity, _ = _build(
        setup_turbulent_cluster, settings._replace(mach_number=2.0 * settings.mach_number)
    )
    np.testing.assert_allclose(
        doubled_velocity, 2.0 * velocity, rtol=1e-12, atol=1e-12,
        err_msg="doubling the Mach number does not double the same velocity field",
    )
