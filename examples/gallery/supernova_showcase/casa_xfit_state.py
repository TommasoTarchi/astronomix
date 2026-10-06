"""
The fit's evolved state at its FIRST epoch (2000) as an npz (``casa_xfit
--save-state``): the starting point of a 4D-Var in the observer frame.

The field-level modes (Y_lm ejecta modes, the wind dipole, the interior
rotation, the speed-up ...) are applied inside ``casa_xfit.make_forward``, so
the only faithful way to get the fit's 2000 state is to evolve it there; this
module writes and reads what comes out.

File format (float32 fields, (n, n, n), code units of ``_common.snr_code_units``,
grid axes = simulation frame; the line of sight is +y (``vy``)):

* hydro: ``rho``, ``vx``, ``vy``, ``vz``, ``press`` (primitive), and
  ``internal_energy`` (the dual-energy variable) when the config has one;
* composition: ``C_ej``, ``C_Fe``, ``C_Si``, ``C_O``, ``C_He``;
* shock history: ``entropy_initial``, ``shocked_fraction``,
  ``time_since_shock``, ``density_time``;
* ``var_layout`` (json: state index -> name; ``state_from_npz`` rebuilds the
  solver state exactly, in that layout);
* metadata: ``box`` (pc), ``num_cells``, ``age`` (yr since explosion at the
  epoch), ``epoch_year``, ``epoch_label``, ``t_expl``, ``theta``, ``names``
  (casa_xfit.PARAM_NAMES), the observer-frame parameters ``psi`` (roll, deg),
  ``dw`` / ``dn`` (explosion centre west / north of RA0/DEC0, arcsec),
  ``distance_kpc``, ``gamma``, ``options`` (json), ``ic``, ``argv``, and the
  IC's wind bookkeeping (``n_w``, ``r_fs_ref``, ``n_c``, ``ambient_*``,
  ``csm_composition``) so the file is also a ``casa_orlando --from-state``
  restart (``--composition``) and a casa_xfit ``--ic`` (with the identity
  physics transform: every hydro parameter at its identity value, the modes 0).
"""
import json
import sys

import numpy as np

HISTORY = ("entropy_initial", "shocked_fraction", "time_since_shock", "density_time")
#: IC keys carried over verbatim (wind / CSM bookkeeping the transforms and
#: casa_orlando read)
IC_META = ("n_w", "r_fs_ref", "n_c", "ambient_rho_w", "ambient_r_ref", "ambient_p_w", "ambient_p_slope",
           "ambient_rho_sh", "ambient_r_sh", "ambient_sigma_sh", "ambient_theta_sh", "ambient_phi_sh",
           "ambient_H_sh", "ambient_shell_rel_rms", "ambient_M_shell_msun", "ambient_M_shell_in_box_msun",
           "csm_composition", "tracer_key", "pipeline_groups", "source")


def var_layout(rv, scalar_names):
    """{state index: field name} for the registered variables."""
    lay = {rv.density_index: "rho", rv.pressure_index: "press"}
    for ax, k in zip("xyz", ("x", "y", "z")):
        lay[int(getattr(rv.velocity_index, k))] = f"v{ax}"
    if getattr(rv, "internal_energy_index", -1) >= 0:
        lay[rv.internal_energy_index] = "internal_energy"
    i0 = rv.passive_scalar_index
    for k, name in enumerate(scalar_names):
        lay[i0 + k] = name
    i_hist = i0 + rv.num_passive_scalars - len(HISTORY)
    for j, name in enumerate(HISTORY):
        lay[i_hist + j] = name
    return {int(k): v for k, v in lay.items()}


def save_state(path, state, rv, scalar_names, *, ic, theta, names, epoch_year, epoch_label, t_expl,
               options=None, extra=None):
    """Write the state (``(num_vars, n, n, n)``, host or device array)."""
    st = np.asarray(state, np.float32)
    lay = var_layout(rv, scalar_names)
    for k in range(st.shape[0]):
        lay.setdefault(k, f"var{k}")
    p = dict(zip(names, np.asarray(theta, np.float64)))
    out = {name: st[k] for k, name in lay.items()}
    out.update(var_layout=json.dumps(lay), num_vars=st.shape[0],
               box=float(ic["box"]), num_cells=int(ic["num_cells"]),
               age=float(epoch_year - t_expl), epoch_year=float(epoch_year), epoch_label=str(epoch_label),
               t_expl=float(t_expl), theta=np.asarray(theta, np.float64), names=np.array(names),
               psi=float(p.get("psi", 0.0)), dw=float(p.get("dw", 0.0)), dn=float(p.get("dn", 0.0)),
               distance_kpc=float(np.exp(p["ln_D"])) if "ln_D" in p else np.nan,
               gamma=float(ic["gamma"]) if "gamma" in ic else 5.0 / 3.0,
               options=json.dumps(options or {}), argv=" ".join(sys.argv),
               ic_age=float(ic["age"]), frame="simulation grid; LOS = +y; sky = roll psi, offset (dw, dn), "
                                            "scale D_ref / D (casa_jaxobs.project_columns)")
    for k in IC_META:
        if k in ic and k not in out:
            out[k] = np.asarray(ic[k])
    out.update(extra or {})
    np.savez_compressed(path, **out)
    return lay


def load_state(path):
    """(fields dict, metadata dict) from a ``save_state`` file."""
    d = np.load(path, allow_pickle=False)
    lay = {int(k): v for k, v in json.loads(str(d["var_layout"])).items()}
    fields = {v: np.asarray(d[v]) for v in lay.values()}
    meta = {k: d[k] for k in d.files if k not in fields}
    meta["var_layout"] = lay
    return fields, meta


def state_from_npz(path):
    """The solver state array ``(num_vars, n, n, n)`` in the saved layout."""
    fields, meta = load_state(path)
    lay = meta["var_layout"]
    return np.stack([fields[lay[k]] for k in range(int(meta["num_vars"]))])
