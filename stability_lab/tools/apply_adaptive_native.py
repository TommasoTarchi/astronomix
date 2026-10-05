"""One-off: minimal admissible splitting speeds in the native WENO kernel."""
path = "astronomix/_finite_difference/_interface_fluxes/_weno.py"
src = open(path).read()

src = src.replace('''from astronomix._finite_difference._interface_fluxes._weno_positivity import (
    mass_free_modes,
    positivity_preserving_interface_flux,
    stencil_maximum,
)''', '''from astronomix._finite_difference._interface_fluxes._weno_positivity import (
    admissible_speed_fraction,
    positivity_preserving_interface_flux,
    stencil_maximum,
)''')

old_start = src.index("    if positivity_preserving:\n        # One splitting speed for every field that carries mass")
old_end = src.index("    def mode_flux(mode, F_current):")
new = '''    def mode_left_row(mode):
        if config.equation_of_state == IDEAL_GAS:
            if config.mhd:
                return _eigen_L_row(conserved_state, rhomin, pgmin, gamma, registered_variables, mode, internal_energy_density=internal_energy_density, dual_eta=dual_eta, admissible_face_state=admissible_face_state)
            return _eigen_L_row_hydro(conserved_state, rhomin, pgmin, gamma, config, registered_variables, mode, internal_energy_density=internal_energy_density, dual_eta=dual_eta)
        if config.mhd:
            return _eigen_L_row_iso(conserved_state, rhomin, isothermal_sound_speed, registered_variables, mode)
        return _eigen_L_row_hydro_iso(conserved_state, rhomin, isothermal_sound_speed, config, registered_variables, mode)

    def mode_right_column(mode):
        if config.equation_of_state == IDEAL_GAS:
            if config.mhd:
                return _eigen_R_col(conserved_state, rhomin, pgmin, gamma, registered_variables, mode, internal_energy_density=internal_energy_density, dual_eta=dual_eta, admissible_face_state=admissible_face_state)
            return _eigen_R_col_hydro(conserved_state, rhomin, pgmin, gamma, config, registered_variables, mode, internal_energy_density=internal_energy_density, dual_eta=dual_eta)
        if config.mhd:
            return _eigen_R_col_iso(conserved_state, rhomin, isothermal_sound_speed, registered_variables, mode)
        return _eigen_R_col_hydro_iso(conserved_state, rhomin, isothermal_sound_speed, config, registered_variables, mode)

    if positivity_preserving:
        # The reference splitting speed: the largest wave speed anywhere on
        # the stencil. With it every split flux is a scaled admissible state.
        spectral_radius = jnp.max(
            jnp.stack([jnp.abs(mode_eigenvalues(mode)) for mode in range(num_modes)]), axis=0
        )
        common_speed = stencil_maximum(spectral_radius)
        if config.weno_ad_frozen_weights:
            common_speed = jax.lax.stop_gradient(common_speed)
        safe_common_speed = jnp.maximum(common_speed, 1e-30)

        # Pre-pass: the shift z of the two upwind split states if every field
        # kept its own speed. The largest fraction eta of the way from the
        # common to the per-field speeds that keeps both states admissible is
        # fixed here, before the reconstruction, because the speeds enter the
        # WENO weights (see _weno_positivity.py).
        right_neighbour = _shift(conserved_state, -1, axis=1)

        def accumulate_full_shift(mode, shifts):
            plus_shift, minus_shift = shifts
            left_row = mode_left_row(mode)
            right_column = mode_right_column(mode)
            own_speed = stencil_maximum(jnp.abs(mode_eigenvalues(mode)))
            relative_offset = ((own_speed - common_speed) / safe_common_speed)[None]
            left_projection = jnp.sum(left_row * conserved_state, axis=0)[None]
            right_projection = jnp.sum(left_row * right_neighbour, axis=0)[None]
            return (
                plus_shift + relative_offset * right_column * left_projection,
                minus_shift + relative_offset * right_column * right_projection,
            )

        zero = jnp.zeros_like(F_interface)
        full_plus_shift, full_minus_shift = jax.lax.fori_loop(
            0, num_modes, accumulate_full_shift, (zero, zero)
        )
        speed_fraction = admissible_speed_fraction(
            conserved_state, F, common_speed, full_plus_shift, full_minus_shift,
            params, config, registered_variables,
        )
        if config.weno_ad_frozen_weights:
            speed_fraction = jax.lax.stop_gradient(speed_fraction)

'''
src = src[:old_start] + new + src[old_end:]

old = '''        if positivity_preserving:
            amx = jnp.where(keeps_own_speed[mode], amx, common_speed)
'''
new = '''        if positivity_preserving:
            amx = common_speed - speed_fraction * (common_speed - amx)
'''
assert old in src; src = src.replace(old, new, 1)

old = '''            if not carries_shifts:
                plus_correction, minus_correction = F_current
                return (plus_correction - R_col * second[None], minus_correction + R_col * third[None])
            plus_correction'''
new = '''            plus_correction'''
assert old in src; src = src.replace(old, new, 1)
old = '''            relative_offset = speed_offset / jnp.maximum(common_speed, 1e-30)[None]'''
new = '''            relative_offset = speed_offset / safe_common_speed[None]'''
assert old in src; src = src.replace(old, new, 1)
old = '''        zero = jnp.zeros_like(F_interface)
        if carries_shifts:
            plus_correction, minus_correction, plus_owner_shift, minus_owner_shift = jax.lax.fori_loop(
                0, num_modes, mode_flux, (zero, zero, zero, zero)
            )
        else:
            plus_correction, minus_correction = jax.lax.fori_loop(
                0, num_modes, mode_flux, (zero, zero)
            )
            plus_owner_shift = minus_owner_shift = 0.0
        return positivity_preserving_interface_flux('''
new = '''        plus_face_flux, minus_face_flux, plus_owner_shift, minus_owner_shift = jax.lax.fori_loop(
            0, num_modes, mode_flux, (zero, zero, zero, zero)
        )
        # the central parts of the two split fluxes, at the common speed (the
        # fields' own speeds entered through the shifts above)
        central_flux = F_interface
        central_state = (1.0 / 12.0) * (
            -_shift(conserved_state, 1, axis=1) + 7.0 * conserved_state
            + 7.0 * right_neighbour - _shift(conserved_state, -2, axis=1)
        )
        plus_face_flux = plus_face_flux + 0.5 * (central_flux + common_speed[None] * central_state)
        minus_face_flux = minus_face_flux + 0.5 * (central_flux - common_speed[None] * central_state)
        return positivity_preserving_interface_flux('''
assert old in src; src = src.replace(old, new, 1)
old = '''            plus_correction,
            minus_correction,
            plus_owner_shift,
            minus_owner_shift,
            params,'''
new = '''            plus_face_flux,
            minus_face_flux,
            plus_owner_shift,
            minus_owner_shift,
            params,'''
assert old in src; src = src.replace(old, new, 1)
open(path, "w").write(src)
print("ok")
