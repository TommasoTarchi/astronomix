"""One-off: wire the positivity-preserving WENO into the native kernel."""
import re
path = "astronomix/_finite_difference/_interface_fluxes/_weno.py"
src = open(path).read()
old = '''from astronomix._finite_difference._interface_fluxes._weno_weights import (
    _weno_omega_weights,
    _weno_omega_weights_ad,
    _weno_omega_weights_z,
)'''
new = old + '''
from astronomix._finite_difference._interface_fluxes._weno_positivity import (
    mass_free_modes,
    positivity_preserving_interface_flux,
    stencil_maximum,
)'''
assert old in src; src = src.replace(old, new, 1)
old = '''    omega_weights = _weno_omega_weights_z if config.weno_z else _weno_omega_weights_ad
'''
new = old + '''    positivity_preserving = config.weno_positivity_preserving
    admissible_face_state = config.weno_admissible_face_state
'''
assert old in src; src = src.replace(old, new, 1)
for name in ["_eigen_L_row", "_eigen_R_col"]:
    var = "L_row" if name == "_eigen_L_row" else "R_col"
    old = f'''                {var} = {name}(conserved_state, rhomin, pgmin, gamma, registered_variables, mode, internal_energy_density=internal_energy_density, dual_eta=dual_eta)'''
    new = f'''                {var} = {name}(conserved_state, rhomin, pgmin, gamma, registered_variables, mode, internal_energy_density=internal_energy_density, dual_eta=dual_eta, admissible_face_state=admissible_face_state)'''
    assert old in src, name; src = src.replace(old, new, 1)

# the mode count moves above the loop body (the common speed needs it)
old_count = '''    
    if config.mhd:
        num_modes = 7
    else:
        num_modes = config.dimensionality + 2

    if config.equation_of_state == ISOTHERMAL:
        num_modes -= 1
    
    # I went for the for loop instead of one einsum'''
assert old_count in src
src = src.replace(old_count, '''
    # I went for the for loop instead of one einsum''', 1)

old = '''    def mode_flux(mode, F_current):
'''
assert src.count(old) == 1
new = '''    if config.mhd:
        num_modes = 7
    else:
        num_modes = config.dimensionality + 2

    if config.equation_of_state == ISOTHERMAL:
        num_modes -= 1

    def mode_eigenvalues(mode):
        if config.equation_of_state == IDEAL_GAS:
            if config.mhd:
                return _eigen_lambdas(conserved_state, rhomin, pgmin, gamma, registered_variables, mode, internal_energy_density=internal_energy_density, dual_eta=dual_eta)
            return _eigen_lambdas_hydro(conserved_state, rhomin, pgmin, gamma, config, registered_variables, mode, internal_energy_density=internal_energy_density, dual_eta=dual_eta)
        if config.mhd:
            return _eigen_lambdas_iso(conserved_state, rhomin, isothermal_sound_speed, registered_variables, mode)
        return _eigen_lambdas_hydro_iso(conserved_state, rhomin, isothermal_sound_speed, config, registered_variables, mode)

    if positivity_preserving:
        # One splitting speed for every field that carries mass: the largest
        # wave speed anywhere on the stencil. Only then is each split flux a
        # scaled admissible state (see _weno_positivity.py).
        spectral_radius = jnp.max(
            jnp.stack([jnp.abs(mode_eigenvalues(mode)) for mode in range(num_modes)]), axis=0
        )
        common_speed = stencil_maximum(spectral_radius)
        if config.weno_ad_frozen_weights:
            common_speed = jax.lax.stop_gradient(common_speed)
        keeps_own_speed = jnp.array([mode in mass_free_modes(config) for mode in range(num_modes)])

    def mode_flux(mode, F_current):
'''
src = src.replace(old, new, 1)

old = '''        lam_stack = jnp.stack([lam0, lam1, lam2, lam3, lam4, lam5], axis=0)
        amx = jnp.max(jnp.abs(lam_stack), axis=0)
'''
new = old + '''        if positivity_preserving:
            amx = jnp.where(keeps_own_speed[mode], amx, common_speed)
'''
assert old in src; src = src.replace(old, new, 1)

old = '''        if config.dimensionality == 3:
            dF = jnp.einsum('nxyz,xyz->nxyz', R_col, Fs)'''
new = '''        if positivity_preserving:
            # Keep the two split fluxes apart. A field on its own (smaller)
            # speed also shifts the central part and the upwind cells' split
            # states along its eigenvector, by (own - common) speed.
            plus_correction, minus_correction, plus_owner_shift, minus_owner_shift = F_current
            speed_offset = (amx - common_speed)[None]
            central_projection = (1.0 / 12.0) * (-q1 + 7.0 * q2 + 7.0 * q3 - q4)
            relative_offset = speed_offset / jnp.maximum(common_speed, 1e-30)[None]
            return (
                plus_correction - R_col * second[None] + 0.5 * speed_offset * R_col * central_projection[None],
                minus_correction + R_col * third[None] - 0.5 * speed_offset * R_col * central_projection[None],
                plus_owner_shift + relative_offset * R_col * q2[None],
                minus_owner_shift + relative_offset * R_col * q3[None],
            )

        if config.dimensionality == 3:
            dF = jnp.einsum('nxyz,xyz->nxyz', R_col, Fs)'''
assert src.count(old) == 1; src = src.replace(old, new, 1)

old = '''    # I went for the for loop instead of one einsum'''
new = '''    if positivity_preserving:
        zero = jnp.zeros_like(F_interface)
        plus_correction, minus_correction, plus_owner_shift, minus_owner_shift = jax.lax.fori_loop(
            0, num_modes, mode_flux, (zero, zero, zero, zero)
        )
        return positivity_preserving_interface_flux(
            conserved_state,
            F,
            common_speed,
            plus_correction,
            minus_correction,
            plus_owner_shift,
            minus_owner_shift,
            params,
            config,
            registered_variables,
        )

    # I went for the for loop instead of one einsum'''
assert src.count(old) == 1; src = src.replace(old, new, 1)
open(path, "w").write(src)
print("written")
