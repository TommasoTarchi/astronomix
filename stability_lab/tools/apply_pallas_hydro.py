"""One-off: add the admissible face state and PP-WENO to the Pallas hydro kernel."""
path = "astronomix/_finite_difference/_interface_fluxes/_weno_pallas.py"
src = open(path).read()

# locate the forward hydro kernel body only (the adjoint / rhs kernels are untouched)
start = src.index("def _weno_flux_hydro_pallas_local(")
end = src.index("def _weno_flux_hydro_pallas_vjp_local(")
body = src[start:end]

old = '''    omega_weights = _weno_omega_weights_z if config.weno_z else _weno_omega_weights
    tiny = 1e-14
'''
new = '''    omega_weights = _weno_omega_weights_z if config.weno_z else _weno_omega_weights
    tiny = 1e-14
    admissible_face_state = config.weno_admissible_face_state
    positivity_preserving = config.weno_positivity_preserving
    # characteristic fields that keep their own splitting speed under PP-WENO
    # (the shear waves; see _weno_positivity.mass_free_modes)
    own_speed_modes = mass_free_modes(config) if positivity_preserving else ()
'''
assert body.count(old) == 1; body = body.replace(old, new, 1)

old = '''        h_face = 0.5 * (h_i + h_j)
        v2_face = vn_face * vn_face + vt1_face * vt1_face + vt2_face * vt2_face
        c2_face = gm1 * (h_face - 0.5 * v2_face)
'''
new = '''        v2_face = vn_face * vn_face + vt1_face * vt1_face + vt2_face * vt2_face
        if admissible_face_state:
            # sound speed from the averaged pressure (see the native
            # _eigenvector_building_blocks): positive and frame independent
            c2_face = gamma * (0.5 * (p_i + p_j)) / rho_face
            h_face = c2_face / gm1 + 0.5 * v2_face
        else:
            h_face = 0.5 * (h_i + h_j)
            c2_face = gm1 * (h_face - 0.5 * v2_face)
'''
assert body.count(old) == 1; body = body.replace(old, new, 1)

old = '''        flux_acc = [
            (-f_stencil[1][slot] + 7.0 * f_stencil[2][slot] + 7.0 * f_stencil[3][slot] - f_stencil[4][slot]) * (1.0 / 12.0)
            for slot in range(ncomp)
        ]

        for mode in range(num_modes):'''
new = '''        flux_acc = [
            (-f_stencil[1][slot] + 7.0 * f_stencil[2][slot] + 7.0 * f_stencil[3][slot] - f_stencil[4][slot]) * (1.0 / 12.0)
            for slot in range(ncomp)
        ]

        if positivity_preserving:
            # One splitting speed (the stencil's spectral radius |v_n| + c) for
            # every field that carries mass, and the two split fluxes kept
            # apart, as in the native _weno_flux_x_native.
            common_speed = jnp.abs(floored_stencil[0][5]) + floored_stencil[0][11]
            for k in range(1, 6):
                common_speed = jnp.maximum(
                    common_speed, jnp.abs(floored_stencil[k][5]) + floored_stencil[k][11]
                )
            safe_speed = jnp.maximum(common_speed, 1e-30)
            central_state = [
                (-q_stencil[1][slot] + 7.0 * q_stencil[2][slot] + 7.0 * q_stencil[3][slot] - q_stencil[4][slot]) * (1.0 / 12.0)
                for slot in range(ncomp)
            ]
            plus_acc = [0.5 * (flux_acc[slot] + common_speed * central_state[slot]) for slot in range(ncomp)]
            minus_acc = [0.5 * (flux_acc[slot] - common_speed * central_state[slot]) for slot in range(ncomp)]
            plus_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]
            minus_shift = [flux_acc[0] * 0.0 for _ in range(ncomp)]

        for mode in range(num_modes):'''
assert body.count(old) == 1; body = body.replace(old, new, 1)

old = '''            amx = alpha_for_mode(mode)

            aterm_p = 0.5 * (d0 + amx * dq0)'''
new = '''            amx = alpha_for_mode(mode)
            if positivity_preserving and mode not in own_speed_modes:
                amx = common_speed

            aterm_p = 0.5 * (d0 + amx * dq0)'''
assert body.count(old) == 1; body = body.replace(old, new, 1)

old = '''            Fs = -second + third
            flux_acc = add_right_correction(flux_acc, mode, Fs)
'''
new = '''            if positivity_preserving:
                zero_acc = [plus_acc[0] * 0.0 for _ in range(ncomp)]
                plus_acc = add_right_correction(plus_acc, mode, -second)
                minus_acc = add_right_correction(minus_acc, mode, third)
                if mode in own_speed_modes:
                    # a field on its own speed also shifts the central part and
                    # the upwind cells' split states along its eigenvector
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

            Fs = -second + third
            flux_acc = add_right_correction(flux_acc, mode, Fs)

        if positivity_preserving:
            flux_acc = positivity_preserving_flux_local(
                q_stencil[2], q_stencil[3], f_stencil[2], f_stencil[3],
                plus_acc, minus_acc, plus_shift, minus_shift,
                common_speed, gm1, rhomin, pgmin,
            )
'''
assert body.count(old) == 1; body = body.replace(old, new, 1)

src = src[:start] + body + src[end:]

# imports
old = '''from astronomix._finite_difference._interface_fluxes._weno_weights import ('''
new = '''from astronomix._finite_difference._interface_fluxes._weno_positivity import (
    mass_free_modes,
    positivity_preserving_flux_local,
)
from astronomix._finite_difference._interface_fluxes._weno_weights import ('''
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new, 1)
open(path, "w").write(src)
print("ok")
