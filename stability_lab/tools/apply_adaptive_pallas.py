"""One-off: minimal admissible splitting speeds in the three Pallas WENO kernels."""
import re

path = "astronomix/_finite_difference/_interface_fluxes/_weno_pallas.py"
src = open(path).read()

src = src.replace('''from astronomix._finite_difference._interface_fluxes._weno_positivity import (
    mass_free_modes,
    positivity_preserving_flux_local,
)''', '''from astronomix._finite_difference._interface_fluxes._weno_positivity import (
    admissible_speed_fraction_local,
    positivity_preserving_flux_local,
)''')

PREPASS = '''
            # Pre-pass: the shift of the two upwind split states with every
            # field on its own speed, and the largest fraction eta of the way
            # there that keeps them admissible (see _weno_positivity.py).
            full_plus_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]
            full_minus_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]
            for mode in range(num_modes):
                relative_offset = (alpha_for_mode(mode) - common_speed) / safe_speed
                full_plus_shift = add_right_correction(
                    full_plus_shift, mode, relative_offset * left_project(mode, q_stencil[2])
                )
                full_minus_shift = add_right_correction(
                    full_minus_shift, mode, relative_offset * left_project(mode, q_stencil[3])
                )
            speed_fraction = admissible_speed_fraction_local(
                q_stencil[2], q_stencil[3], f_stencil[2], f_stencil[3], common_speed,
                full_plus_shift, full_minus_shift, {gm1}, rhomin, {pgmin},
                ideal_gas={ideal_gas}, magnetic_slots={slots},
            )
'''

def patch_kernel(body, gm1, pgmin, ideal_gas, slots, indent_fix=""):
    # drop the own-speed-mode table
    body = re.sub(r"\n    # [^\n]*\n    # \(the shear waves; see _weno_positivity.mass_free_modes\)\n    own_speed_modes = mass_free_modes\(config\) if positivity_preserving else \(\)\n", "\n", body)
    body = re.sub(r"\n    # the Alfven waves keep their own splitting speed under PP-WENO\n    own_speed_modes = mass_free_modes\(config\) if positivity_preserving else \(\)\n", "\n", body)
    assert "own_speed_modes = " not in body, "table"
    # pre-pass after the shift initialisation
    old = '''            minus_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]
'''
    assert body.count(old) == 1, "init"
    body = body.replace(old, old + PREPASS.format(gm1=gm1, pgmin=pgmin, ideal_gas=ideal_gas, slots=slots), 1)
    old = '''            if positivity_preserving and mode not in own_speed_modes:
                amx = common_speed
'''
    assert body.count(old) == 1, "amx"
    body = body.replace(old, '''            if positivity_preserving:
                amx = common_speed - speed_fraction * (common_speed - amx)
''', 1)
    old_block = re.search(r"                if mode in own_speed_modes:\n((?:                    .*\n)+)", body)
    assert old_block, "shift block"
    dedented = "".join(line[4:] + "\n" for line in old_block.group(1).splitlines())
    body = body.replace(old_block.group(0), dedented, 1)
    return body

# hydro forward kernel
start = src.index("def _weno_flux_hydro_pallas_local(")
end = src.index("def _weno_flux_hydro_pallas_vjp_local(")
src = src[:start] + patch_kernel(src[start:end], "gm1", "pgmin", "True", "()") + src[end:]

# iso-MHD kernel
start = src.index("def _weno_flux_mhd_iso_pallas_local(")
end = src.index("def _weno_flux_hydro_pallas_rhs(")
src = src[:start] + patch_kernel(src[start:end], "0.0", "0.0", "False", "(4, 5, 6)") + src[end:]

# ideal-MHD window function: add shifts, pre-pass and per-mode shift accumulation
start = src.index("def _weno_mhd_flux_from_window(")
end = src.index("def _weno_mhd_flux_from_window_adjoint(")
body = src[start:end]
old = '''        plus_acc = [0.5 * (flux_acc[slot] + common_speed * central_state[slot]) for slot in range(ncomp)]
        minus_acc = [0.5 * (flux_acc[slot] - common_speed * central_state[slot]) for slot in range(ncomp)]
'''
new = '''        plus_acc = [0.5 * (flux_acc[slot] + common_speed * central_state[slot]) for slot in range(ncomp)]
        minus_acc = [0.5 * (flux_acc[slot] - common_speed * central_state[slot]) for slot in range(ncomp)]
        safe_speed = jnp.maximum(common_speed, 1e-30)
        plus_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]
        minus_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]
''' + "\n".join(line[4:] for line in PREPASS.format(
    gm1="gm1", pgmin="pgmin", ideal_gas="True", slots="(4, 5, 6)").splitlines()) + "\n"
assert body.count(old) == 1, "mhd init"
body = body.replace(old, new, 1)
body = body.replace('''        # One splitting speed (the stencil's spectral radius) for every field
        # (all ideal-MHD fields carry mass or energy); split fluxes kept apart.''', '''        # The reference splitting speed (the stencil's spectral radius) and
        # the two split fluxes kept apart, as in the native kernel.''')
old = '''        amx = common_speed if positivity_preserving else alpha_for_mode(mode)
'''
new = '''        amx = alpha_for_mode(mode)
        if positivity_preserving:
            amx = common_speed - speed_fraction * (common_speed - amx)
'''
assert body.count(old) == 1, "mhd amx"
body = body.replace(old, new, 1)
old = '''        if positivity_preserving:
            plus_acc = add_right_correction(plus_acc, mode, -second)
            minus_acc = add_right_correction(minus_acc, mode, third)
            continue
'''
new = '''        if positivity_preserving:
            zero_acc = [plus_acc[0] * 0.0 for _ in range(ncomp)]
            plus_acc = add_right_correction(plus_acc, mode, -second)
            minus_acc = add_right_correction(minus_acc, mode, third)
            speed_offset = amx - common_speed
            central_projection = (
                -qproj[1] + 7.0 * qproj[2] + 7.0 * qproj[3] - qproj[4]
            ) * (1.0 / 12.0)
            central_shift = add_right_correction(zero_acc, mode, 0.5 * speed_offset * central_projection)
            plus_acc = [plus_acc[slot] + central_shift[slot] for slot in range(ncomp)]
            minus_acc = [minus_acc[slot] - central_shift[slot] for slot in range(ncomp)]
            relative_offset = speed_offset / safe_speed
            plus_shift = add_right_correction(plus_shift, mode, relative_offset * qproj[2])
            minus_shift = add_right_correction(minus_shift, mode, relative_offset * qproj[3])
            continue
'''
assert body.count(old) == 1, "mhd loop"
body = body.replace(old, new, 1)
old = '''        no_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]
        flux_acc = positivity_preserving_flux_local(
            q_stencil[2], q_stencil[3], f_stencil[2], f_stencil[3],
            plus_acc, minus_acc, no_shift, no_shift,'''
new = '''        flux_acc = positivity_preserving_flux_local(
            q_stencil[2], q_stencil[3], f_stencil[2], f_stencil[3],
            plus_acc, minus_acc, plus_shift, minus_shift,'''
assert body.count(old) == 1, "mhd final"
body = body.replace(old, new, 1)
src = src[:start] + body + src[end:]
open(path, "w").write(src)
print("ok")
