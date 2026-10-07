"""
Cas A's Si-rich NE jet and SW counter-jet as a traced, differentiable addition
to the 146-yr initial condition (``casa_xfit --jet on``).

Orlando's W15-IIb state carries neither jet, but Cas A has both: a narrow
Si/S/Ar/Ca-rich stream to the NE that pierces the forward shock and runs out to
~320" from the expansion centre, and a fainter SW counterpart to ~260". The
model adds, in the RESCALED frame (after the similarity map, the rotation of
the interior and the Y_lm ejecta modes, i.e. in the frame the observer sees,
rolled by ``psi``), a bipolar cone of cold, homologous, Si-rich ejecta:

* axis n_j: sky position angle ``jet_pa`` (deg, north through east) and
  inclination ``jet_incl`` (deg out of the sky plane, + = receding, i.e. +y);
  the counter-jet runs along -n_j, turned on the sky by ``cj_dpa`` (deg; its
  sky PA is jet_pa + 180 + cj_dpa, its inclination -jet_incl): in the 2004 Si
  band the NE jet peaks at PA 60-70, the SW counter-jet at PA ~260, i.e. ~15
  deg off antiparallel in projection (prior 0 +- 15);
* smooth angular profile w = exp(-(1 - n_j . r_hat) / (1 - cos theta_j)),
  theta_j = exp(``jet_lnth``) deg (w = 1/e at theta_j);
* homologous stream v = r / t_IC (radial), from r_in = JET_VIN_FRAC r_tip to
  r_tip = v_tip t_IC (``jet_vtip``, 1000 km/s; the counter-jet's tip
  CJ_VTIP_FRAC of it), with sigmoid ends of width JET_EDGE_FRAC r_tip; uniform
  dm/dv (rho ~ r^-2 in the cone), so the tip may lie beyond the 146-yr forward
  shock, as observed;
* mass M_NE = exp(``jet_ln_m``) Msun and M_SW = exp(``cj_lnr``) M_NE, imposed
  EXACTLY on the grid (discrete normalisation of the profile);
* composition C_ej = 1 and JET_COMPOSITION (Si-group dominated, O weak; the
  Doppler tracer = the Si-group fraction), never shocked (shock history 0);
* pressure: the local pressure plus rho_jet kT_cold / (mu m_p) (T_cold
  JET_T_COLD_K): pressure equilibrium with its surroundings, cold.

The material is ADDED to the cells (mass, momentum and every per-mass field
mixed mass-weighted), so the fields stay finite and rho only grows. The swept
CSM inside the cone ahead of the 146-yr shock is left to the hydro.

Priors (``JET_PRIOR``; literature, 2026-10-01 jet worker, report in
/export/data/lstorcks/casa_orlando150/work/ers/jet/REPORT.md):

* PA 65 +- 10 deg (NE jet PA ~60-75, SW counter-jet ~230-250; Hwang+04,
  Fesen+06, Fesen & Milisavljevic 2016);
* inclination 0 +- 10 deg: "the fastest knots lie close to the plane of the
  sky and thus exhibit relatively modest radial velocities", while the NE region
  as a whole spans -4000..+5000 km/s, its tail REDshifted (Milisavljevic &
  Fesen 2013; DeLaney+10). (Review 2026-10-02: the earlier "< ~800 km/s, Fesen
  & Milisavljevic 2016" is not in either paper's abstract; R4's -11 deg, NE
  approaching, is set by the Doppler sectors at PA 60-90 inside the rim, not by
  this literature, which if anything leans the other way);
* tip speed 15.0 +- 1.5 (1000 km/s): fastest NE knots 15,600 km/s transverse
  at 3.4 kpc (Fesen & Milisavljevic 2016; 14,000-15,000 km/s Fesen+06), i.e.
  ~15,200 at 3.32 kpc; SW 12,700 (CJ_VTIP_FRAC 0.81);
* half-angle ln(12 +- ~60 %) deg: the X-ray Si jet's opening angle is ~7 deg
  (Laming+06), the optical outer-knot fans ~40 deg half-angle (Milisavljevic &
  Fesen 2013; Fesen & Milisavljevic 2016);
* M_NE ln(0.07 Msun) +- 0.7, M_SW / M_NE ln(0.4) +- 0.7: jet + counter-jet
  ~0.1 Msun with E_kin ~1e50 erg (Fesen & Milisavljevic 2016; Laming+06 ~1e50
  erg in the NE jet; Schure+08 >= 1e48 erg; Orlando+16's Si-rich pistons
  0.040 / 0.0091 Msun, ratio 0.23). At the prior means E_kin(NE + SW) ~1.2e50.

    ./run.sh casa_jet.py test          # CPU tiny-grid: mass, positivity, JVP vs FD
"""

# ==== device selection (as a script) ====
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp

#: the jet parameters, appended at the END of casa_xfit's PARAM_NAMES
JET_NAMES = ("jet_ln_m", "jet_vtip", "jet_pa", "jet_incl", "jet_lnth", "cj_lnr", "cj_dpa")
JET_PRIOR = {"jet_ln_m": (float(np.log(0.07)), 0.7),
             "jet_vtip": (15.0, 1.5),
             "jet_pa": (65.0, 10.0),
             "jet_incl": (0.0, 10.0),
             "jet_lnth": (float(np.log(12.0)), 0.5),
             "cj_lnr": (float(np.log(0.4)), 0.7),
             "cj_dpa": (0.0, 15.0)}
JET_FD_STEPS = {"jet_ln_m": 0.1, "jet_vtip": 0.5, "jet_pa": 2.0, "jet_incl": 2.0, "jet_lnth": 0.1,
                "cj_lnr": 0.1, "cj_dpa": 2.0}
#: inner end of the stream (fraction of the tip radius / speed): ~6,800 km/s at
#: the prior tip, i.e. from about the main shell's speed (the X-ray jet starts
#: at the bright ring; the jet knots' proper motions average ~10,000 km/s)
JET_VIN_FRAC = 0.45
#: SW counter-jet tip / NE tip (12,700 / 15,600 km/s)
CJ_VTIP_FRAC = 0.81
#: width of the sigmoid ends (fraction of the tip radius)
JET_EDGE_FRAC = 0.04
#: the stream's mass fractions (casa_xfit tracked species: Fe-group, Si-group
#: = Si+S+Ar+Ca, O-group = O+Ne+Mg+C, He): S, Ar, Ca lines strong, O weak
#: (Fesen & Milisavljevic 2016; Si-rich, Fe-poor X-ray jet: Hwang+04,
#: Laming+06; Orlando+16's 'Si-rich' piston species)
JET_COMPOSITION = {"C_ej": 1.0, "C_Si": 0.70, "C_O": 0.27, "C_Fe": 0.03, "C_He": 0.0, "C_dop": 0.70}
JET_T_COLD_K = 1.0e4
JET_MU = 2.0
#: likelihood masks for a fit with the jet (casa_xfit cone angles theta = PA + 90).
#: Outline (``--outline-mask inner-arc+jet``): the cones whose model forward-shock
#: edge the jet's bow shock pushes out by > 5" at the prior jet (128^3 demo, 2004:
#: NE theta 140-180 by +8..+46", SW 340-0 by +11..+20"); the data's outline there
#: follows the rim beside the jet. Proper motions (``--pm-mask-extra``): the same
#: cones not already in the 'jet' PM mask (theta 160-210) -- JET_PM_EXTRA.
JET_OUTLINE_CONES = (140.0, 150.0, 160.0, 170.0, 180.0, 340.0, 350.0, 0.0)
JET_PM_EXTRA = "300+140+150+340+350+0"
#: the jet image term (casa_xfit ``--jet-img on``, ``jet_image_residuals``):
#: epoch-mean rate per band in 10-deg sky-PA bins of the annulus 195-215" about
#: the prior expansion centre -- beyond the rim (data r_FS <= 192" by 2018 even
#: at the jet's base, PA 70-80) and inside the 128^3 box (217" at 3.32 kpc) --
#: over NE PA 20-110 and SW PA 210-300 (the 2004 Si-band jets: NE PA 50-80, peak
#: 60-70; SW 240-270, peak 260; the off-jet bins pin the width and the halo /
#: background level). Model error JET_SIGMA (ln; the model jet is smooth, the
#: data's knotty).
JET_PROFILE = (195.0, 215.0, ((20.0, 110.0), (210.0, 300.0)), 10.0)
JET_SIGMA = 0.3
COE_ARCSEC = (-13.8, -4.2)


def jet_bin_index(WW, NN, r0, r1, sectors, dpa, centre=COE_ARCSEC):
    """Bin index (-1: none) of each pixel (W, N arcsec of RA0/DEC0): r0 <= r < r1
    about ``centre``, sky PA (north through east) in ``sectors``, in ``dpa``-deg
    bins numbered sector by sector."""
    w, n = np.asarray(WW) - centre[0], np.asarray(NN) - centre[1]
    r = np.hypot(w, n)
    pa = np.degrees(np.arctan2(-w, n)) % 360.0
    idx = np.full(r.shape, -1, int)
    k0 = 0
    for a0, a1 in sectors:
        nb = int(round((a1 - a0) / dpa))
        k = np.floor((pa - a0) / dpa).astype(int)
        sel = (k >= 0) & (k < nb) & (r >= r0) & (r < r1)
        idx[sel] = k0 + k[sel]
        k0 += nb
    return idx


#: Msun (1000 km/s)^2 -> 1e51 erg
E_UNIT_51 = 1.98892e33 * 1e16 / 1e51


def jet_off_theta():
    """The jet parameters' values in a theta that does not use the jet (the prior
    means; the forward ignores them without ``--jet on``)."""
    return [JET_PRIOR[k][0] for k in JET_NAMES]


def sky_axis(pa_deg, incl_deg, psi_deg=0.0):
    """Unit vector in the simulation frame (x = west, y = away from the observer,
    z = north) of a direction at sky PA ``pa_deg`` (north through east) and
    inclination ``incl_deg`` (+ = receding), seen through the projection roll
    ``psi`` (sim angle + psi = sky angle, from west through north)."""
    phi = jnp.deg2rad(pa_deg + 90.0 - psi_deg)
    inc = jnp.deg2rad(incl_deg)
    return jnp.stack([jnp.cos(inc) * jnp.cos(phi), jnp.sin(inc), jnp.cos(inc) * jnp.sin(phi)])


def jet_axis(p, psi_deg=0.0):
    """(NE axis, SW axis) unit vectors in the simulation frame."""
    ne = sky_axis(p["jet_pa"], p["jet_incl"], psi_deg)
    sw = sky_axis(p["jet_pa"] + 180.0 + p.get("cj_dpa", 0.0), -p["jet_incl"], psi_deg)
    return ne, sw


def jet_density(p, geom, t_ic, cell_vol, psi_deg=0.0):
    """(rho_ne, rho_sw, v_r) of the bipolar stream (code units: pc, Msun, 1000
    km/s; ``t_ic`` in code time), each with the exact grid mass of its prior-
    parametrised profile. v_r = r / t_IC is the (radial) stream speed."""
    r, X, Y, Z = geom
    r_s = jnp.maximum(r, 1e-6)
    n, n_sw = jet_axis(p, psi_deg)
    mu = (n[0] * X + n[1] * Y + n[2] * Z) / r_s
    mu_sw = (n_sw[0] * X + n_sw[1] * Y + n_sw[2] * Z) / r_s
    one_m_cos = 1.0 - jnp.cos(jnp.deg2rad(jnp.exp(p["jet_lnth"])))

    def stream(m_sun, mu_dir, v_tip):
        r_tip = v_tip * t_ic
        r_in = JET_VIN_FRAC * r_tip
        dr = JET_EDGE_FRAC * r_tip
        radial = jax.nn.sigmoid((r - r_in) / dr) * jax.nn.sigmoid((r_tip - r) / dr) / r_s ** 2
        shape = jnp.exp((mu_dir - 1.0) / one_m_cos) * radial
        norm = jnp.sum(shape) * cell_vol
        return m_sun * shape / jnp.maximum(norm, 1e-30)
    m_ne = jnp.exp(p["jet_ln_m"])
    m_sw = m_ne * jnp.exp(p["cj_lnr"])
    rho_ne = stream(m_ne, mu, p["jet_vtip"])
    rho_sw = stream(m_sw, mu_sw, CJ_VTIP_FRAC * p["jet_vtip"])
    return rho_ne, rho_sw, r / t_ic


def add_jet(f, p, geom, t_ic, cell_vol, *, psi_deg=None, gamma=5.0 / 3.0):
    """Fields ``f`` (rho, vx, vy, vz, press, per-mass scalars / histories) with
    the jet ADDED (mass-weighted mixing), and its bookkeeping
    ``{M_ne, M_sw, E_kin_ne, E_kin_sw, dE_grid}`` (Msun, 1e51 erg; E_kin: the
    stream's own kinetic energy; dE_grid: the energy the grid actually gains,
    smaller, as the mass- and momentum-conserving mixing dissipates the
    stream's kinetic energy relative to the gas it lands in) under ``f["_jet"]``. ``psi_deg``: the projection roll
    (default ``p["psi"]``), so ``jet_pa`` is the SKY position angle."""
    psi = p["psi"] if psi_deg is None else psi_deg
    r, X, Y, Z = geom
    rho_ne, rho_sw, v_r = jet_density(p, geom, t_ic, cell_vol, psi)
    rj = rho_ne + rho_sw
    rho = f["rho"]
    rho_new = rho + rj
    r_s = jnp.maximum(r, 1e-6)
    out = dict(f)
    for k, x in (("vx", X), ("vy", Y), ("vz", Z)):
        out[k] = (rho * f[k] + rj * v_r * x / r_s) / rho_new
    # cold: kT / (mu m_p) in (1000 km/s)^2
    cs2 = 1.380649e-16 * JET_T_COLD_K / (JET_MU * 1.67262192e-24) / 1e16
    out["press"] = f["press"] + rj * cs2
    skip = ("rho", "vx", "vy", "vz", "press", "_jet")
    for k in f:
        if k in skip or k.startswith("_"):
            continue
        out[k] = (rho * f[k] + rj * JET_COMPOSITION.get(k, 0.0)) / rho_new
    out["rho"] = rho_new
    ek = 0.5 * v_r ** 2 * cell_vol * E_UNIT_51
    # what the grid actually gains (kinetic + thermal, 1e51 erg): the mixing
    # conserves mass and momentum, not energy, so where the stream lands in
    # gas moving at another speed (the shocked shell) its relative kinetic
    # energy is lost, not heated -- at R4, dE_grid 3.0e49 vs E_kin 5.8e49 erg
    # (review 2026-10-02). Diagnostic only.
    v2_old = f["vx"] ** 2 + f["vy"] ** 2 + f["vz"] ** 2
    v2_new = out["vx"] ** 2 + out["vy"] ** 2 + out["vz"] ** 2
    de = (0.5 * (rho_new * v2_new - rho * v2_old) + (out["press"] - f["press"]) / (gamma - 1.0))
    out["_jet"] = dict(M_ne=jnp.sum(rho_ne) * cell_vol, M_sw=jnp.sum(rho_sw) * cell_vol,
                       E_kin_ne=jnp.sum(rho_ne * ek), E_kin_sw=jnp.sum(rho_sw * ek),
                       dE_grid=jnp.sum(de) * cell_vol * E_UNIT_51)
    return out


def _test():
    """CPU tiny grid: exact mass, positivity, JVP vs central FD."""
    jax.config.update("jax_enable_x64", True)
    n, box = 32, 7.0
    ax = (np.arange(n) + 0.5) * box / n - 0.5 * box
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    geom = tuple(jnp.asarray(a) for a in (r, X, Y, Z))
    cv = (box / n) ** 3
    rng = np.random.default_rng(0)
    f = dict(rho=jnp.asarray(0.05 + rng.random(r.shape)), vx=jnp.asarray(X / 0.18), vy=jnp.asarray(Y / 0.18),
             vz=jnp.asarray(Z / 0.18), press=jnp.asarray(1e-3 + 0 * r), C_ej=jnp.asarray(rng.random(r.shape)),
             C_Si=jnp.asarray(0.1 * rng.random(r.shape)), shocked_fraction=jnp.asarray(rng.random(r.shape)))
    t_ic = 172.4 / 977.79
    p0 = dict(zip(JET_NAMES, jet_off_theta()), psi=-13.0)
    out = add_jet(f, p0, geom, t_ic, cv)
    d = {k: float(v) for k, v in out["_jet"].items()}
    dm = float(jnp.sum(out["rho"] - f["rho"]) * cv)
    print("[jet test] bookkeeping", {k: round(v, 5) for k, v in d.items()}, "grid dM", round(dm, 5))
    # dE_grid against the direct sum over the mixed fields
    v2 = lambda o: o["vx"] ** 2 + o["vy"] ** 2 + o["vz"] ** 2  # noqa: E731
    de = float(jnp.sum(0.5 * (out["rho"] * v2(out) - f["rho"] * v2(f)) + 1.5 * (out["press"] - f["press"])) * cv * E_UNIT_51)
    assert abs(d["dE_grid"] - de) < 1e-9 * max(1.0, abs(de)) and d["dE_grid"] <= d["E_kin_ne"] + d["E_kin_sw"] + 1e-12
    assert abs(d["M_ne"] - 0.07) < 1e-9 and abs(d["M_sw"] - 0.028) < 1e-9 and abs(dm - 0.098) < 1e-9
    for k, v in out.items():
        if k != "_jet":
            assert bool(jnp.all(jnp.isfinite(v))), k
    assert bool(jnp.all(out["rho"] >= f["rho"])) and bool(jnp.all(out["press"] > 0))
    assert bool(jnp.all((out["C_ej"] >= 0) & (out["C_ej"] <= 1 + 1e-12)))
    keys = JET_NAMES + ("psi",)
    th0 = jnp.array([p0[k] for k in keys])
    # a smooth scalar of the output (an emission-measure-like moment)
    w = jnp.asarray(rng.random(r.shape))

    def obj(th):
        o = add_jet(f, dict(zip(keys, th)), geom, t_ic, cv)
        return jnp.sum(w * o["rho"] ** 2 * o["C_Si"]) + jnp.sum(w * o["vx"] * o["rho"]) + o["_jet"]["E_kin_ne"]
    steps = {**JET_FD_STEPS, "psi": 2.0}
    for i, k in enumerate(keys):
        e = jnp.zeros(len(keys)).at[i].set(1.0)
        jv = float(jax.jvp(obj, (th0,), (e,))[1])
        h = 0.02 * steps[k]
        fd = float((obj(th0 + h * e) - obj(th0 - h * e)) / (2 * h))
        rel = abs(jv - fd) / max(abs(fd), 1e-12)
        print(f"[jet test] d/d{k:9s} JVP {jv:+.6e} FD {fd:+.6e} rel {rel:.1e}")
        assert np.isfinite(jv) and rel < 1e-4, k
    print("[jet test] OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _test()
