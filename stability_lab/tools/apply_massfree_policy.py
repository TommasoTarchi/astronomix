"""One-off: common speed for mass-carrying fields, own speed for mass-free ones."""
import re

# ---------------------------------------------------------------- module
path = "astronomix/_finite_difference/_interface_fluxes/_weno_positivity.py"
src = open(path).read()
s = src.index("def admissible_speed_fraction(")
e = src.index("def positivity_preserving_interface_flux(")
src = src[:s] + src[e:]
s = src.index("def admissible_speed_fraction_local(")
e = src.index("def positivity_preserving_flux_local(")
src = src[:s] + src[e:]
src = src.replace('''def stencil_maximum(cell_field):''', '''def mass_free_modes(config: SimulationConfig) -> tuple:
    """Indices of the characteristic fields that carry no mass and no
    energy-coupled density, in the mode order of the eigensystem modules: the
    shear waves of the Euler equations (both equations of state) and the
    Alfven waves of isothermal MHD. They keep their own splitting speed.

    Args:
        config: The simulation configuration.

    Returns:
        A tuple of mode indices.
    """
    if config.mhd:
        return (1, 4) if config.equation_of_state == ISOTHERMAL else ()
    if config.equation_of_state == ISOTHERMAL:
        return tuple(range(1, config.dimensionality))
    return tuple(range(2, config.dimensionality + 1))


def stencil_maximum(cell_field):''')
src = src.replace("from astronomix.option_classes.simulation_config import IDEAL_GAS\n",
                  "from astronomix.option_classes.simulation_config import IDEAL_GAS, ISOTHERMAL\n")
old_doc_start = src.index("* the fraction ``eta`` of the way from the common speed")
old_doc_end = src.index("* the face value is pulled toward its upwind state")
src = src[:old_doc_start] + '''* every field that carries mass is split with the common speed ``alpha``,
  the stencil's spectral radius. That makes the frozen-basis splitting
  monotone for every cell of the stencil, however differently the basis
  represents it (a low-density cell with a large fast speed next to dense
  gas). Per-field speeds chosen only to keep the split states admissible
  are positive but not robust: Mach-10 MHD turbulence still blows up
  (commit 96a191f). Fields that carry no mass (hydrodynamic shear waves,
  isothermal-MHD Alfven waves) keep their own speed. Their ``z`` leaves the
  density unchanged and only raises the pressure, so vortical modes keep the
  default dissipation;
''' + src[old_doc_end:]
src = src.replace('''Splitting every characteristic field ``s`` with its own speed
``alpha_s <= alpha`` makes each split flux a scaled vector''', '''Splitting characteristic field ``s`` with speed ``alpha_s <= alpha``
makes each split flux a scaled vector''')
open(path, "w").write(src)

# ---------------------------------------------------------------- native
path = "astronomix/_finite_difference/_interface_fluxes/_weno.py"
src = open(path).read()
src = src.replace('''    admissible_speed_fraction,
    positivity_preserving_interface_flux,''', '''    mass_free_modes,
    positivity_preserving_interface_flux,''')
s = src.index("        # Pre-pass: the shift z_m of every stencil cell's split states if")
e = src.index("            speed_fraction = jax.lax.stop_gradient(speed_fraction)\n") + len("            speed_fraction = jax.lax.stop_gradient(speed_fraction)\n")
src = src[:s] + '''        # Fields that carry no mass keep their own splitting speed.
        keeps_own_speed = jnp.array([mode in mass_free_modes(config) for mode in range(num_modes)])
        right_neighbour = _shift(conserved_state, -1, axis=1)
        zero = jnp.zeros_like(F_interface)
''' + src[e:]
old = '''            amx = common_speed - speed_fraction * (common_speed - amx)
'''
assert old in src
src = src.replace(old, '''            amx = jnp.where(keeps_own_speed[mode], amx, common_speed)
''')
src = src.replace('''        # The reference splitting speed: the largest wave speed anywhere on
        # the stencil. With it every split flux is a scaled admissible state.''', '''        # The common splitting speed of the fields that carry mass: the
        # largest wave speed anywhere on the stencil (see _weno_positivity.py).''')
open(path, "w").write(src)

# ---------------------------------------------------------------- pallas
path = "astronomix/_finite_difference/_interface_fluxes/_weno_pallas.py"
src = open(path).read()
src = src.replace('''from astronomix._finite_difference._interface_fluxes._weno_positivity import (
    admissible_speed_fraction_local,
    positivity_preserving_flux_local,
)''', '''from astronomix._finite_difference._interface_fluxes._weno_positivity import (
    mass_free_modes,
    positivity_preserving_flux_local,
)''')
prepass = re.compile(
    r"\n(?P<ind> +)# Pre-pass: the shifts of every stencil cell's split states with every\n"
    r"(?:(?P=ind).*\n|\n)*?"
    r"(?P=ind)speed_fraction = admissible_speed_fraction_local\(\n"
    r"(?:(?P=ind) .*\n)*?"
    r"(?P=ind)\)\n"
)
src, count = prepass.subn("\n", src)
assert count == 3, count
old_amx = re.compile(r"(?P<ind> +)if positivity_preserving:\n(?P=ind)    amx = common_speed - speed_fraction \* \(common_speed - amx\)\n")
src, count = old_amx.subn(lambda m: f"{m.group('ind')}if positivity_preserving and mode not in own_speed_modes:\n{m.group('ind')}    amx = common_speed\n", src)
assert count == 3, count
# shift accumulation only for the own-speed modes: wrap the shift block
for name, ind in [("_weno_flux_hydro_pallas_local(", "                "), ("_weno_flux_mhd_iso_pallas_local(", "                "), ("_weno_mhd_flux_from_window(", "            ")]:
    start = src.index("def " + name)
    block_start = src.index(ind + "speed_offset = amx - common_speed\n", start)
    block_end = src.index(ind + "minus_shift = add_right_correction(minus_shift, mode, relative_offset * qproj[3])\n", block_start)
    block_end += len(ind + "minus_shift = add_right_correction(minus_shift, mode, relative_offset * qproj[3])\n")
    block = src[block_start:block_end]
    indented = "".join("    " + line + "\n" if line.strip() else "\n" for line in block.splitlines())
    src = src[:block_start] + ind + "if mode in own_speed_modes:\n" + indented + src[block_end:]
# own-speed tables
for name, anchor in [("_weno_flux_hydro_pallas_local(", "    positivity_preserving = config.weno_positivity_preserving\n"),
                     ("_weno_flux_mhd_iso_pallas_local(", "    positivity_preserving = config.weno_positivity_preserving\n")]:
    start = src.index("def " + name)
    pos = src.index(anchor, start) + len(anchor)
    src = src[:pos] + "    # fields that carry no mass keep their own splitting speed\n    own_speed_modes = mass_free_modes(config) if positivity_preserving else ()\n" + src[pos:]
# ideal MHD: no mass-free fields
start = src.index("def _weno_mhd_flux_from_window(")
pos = src.index("    if positivity_preserving:\n", start)
src = src[:pos] + "    own_speed_modes = ()  # every ideal-MHD field carries mass or energy\n" + src[pos:]
open(path, "w").write(src)
print("ok")
