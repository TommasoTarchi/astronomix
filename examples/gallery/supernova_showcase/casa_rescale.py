"""
Exact similarity rescaling of Orlando's Cas A state (or any saved state).

WHY. Adiabatic hydrodynamics of freely expanding ejecta running into a power-law
wind (plus a shell, plus anything else built from the same fields) has NO
intrinsic length, time or mass scale. For any L, T, M > 0 the map

    r' = L r,   t' = T t,   rho' = (M / L^3) rho,   v' = (L / T) v,
    p' = (M / L^3) (L / T)^2 p

takes a solution into a solution. Derived quantities follow:

    E' = M (L / T)^2 E          M_ej' = M M_ej          (every mass x M)
    wind rho r^2 -> (M / L) rho r^2   (so n_H(r) -> (M / L) n_H(r / L) (r / L)^2 / r^2 ...,
                                       at FIXED radius n_H'(r) = (M / L) n_H(r))
    shell: radius, width, scale height x L; density x M / L^3; mass x M
    B' = sqrt(M / L^3) (L / T) B   (B^2 ~ p; B is dynamically negligible here, beta ~ 350)
    shock history: time_since_shock x T, density_time (int rho dt) x M T / L^3
    composition, ejecta tag, shocked_fraction: unchanged (mass fractions)
    entropy label p / rho^gamma: x (M / L^3)^(1 - gamma) (L / T)^2

The rescaled state is the state of a model with energy E' at age T x (the
original age). THE AGE LABEL MUST BE T x 145.5 yr: the ejecta are homologous
(r / v = 145.5 yr in the delivered state), and r' / v' = T r / v, so only
this label keeps the ballistic convergence date equal to the explosion date.
(``casa_pluto_diff``'s legacy ``ln_sv`` is the member L = 1, T = 1 / s, M = 1
WITHOUT the relabelling; that is why the ballistic-date wall was needed.)

What breaks the symmetry: radiative cooling (only in Orlando's run before
146 yr: the cool dense shell; negligible afterwards), B (beta ~ 350), and the
grid (L changes how many cells a structure spans; the rescaling is done on
the fixed 7 pc grid, conservatively).

Implementation: the source cell edges are stretched by L and the state is
remapped CONSERVATIVELY (separable overlap matrices, as ``casa_pluto.convert``)
onto the destination grid -- mass, momentum, internal energy and every
scalar's mass are exact; kinetic energy below the destination cell is lost
and reported. Where the stretched source does not cover the destination box
(L < 1) the scaled analytic wind + shell (``ambient_*`` keys) fills the rest.
The ``ambient_*`` metadata are rescaled too, so every downstream consumer
(``casa_pluto_diff.transform_fields``, ``wind_nh``, ``casa_orlando
--from-state``) sees a consistent state.

For a resolution-matched production IC go through the raw PLUTO data instead
(``casa_pluto.py convert --sim L T M``), which rescales at the source (2048^3
run delivered at 512^3) and loses nothing to an intermediate grid.

Usage::

    CUDA_VISIBLE_DEVICES= ./run.sh casa_rescale.py --ic .../pluto146_n256.npz \\
        --L 1.40 --T 1.20 --M 1.00 --out .../pluto146_n256_sim.npz
    CUDA_VISIBLE_DEVICES= ./run.sh casa_rescale.py --ic ... --info --L 1.4 --T 1.2 --M 1.0
"""

# ==== CPU only ====
import os
if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
# ruff: noqa: E402
# ==================

# general
import argparse
import ast

# numerics
import numpy as np

# units
from astropy import units as u
import astropy.constants as const

# shared showcase helpers
from _common import CSM_COMPOSITION, GAMMA, snr_code_units


# =============================================================================
# ============ ↓ The similarity map ↓ =========================================
# =============================================================================
#: how each saved field transforms: key -> kind (see ``factors``)
FIELD_KIND = {
    "rho": "rho", "press": "press", "vx": "v", "vy": "v", "vz": "v",
    "bx": "B", "by": "B", "bz": "B",
    "C_ej": "frac", "C_Fe": "frac", "C_Si": "frac", "C_O": "frac", "C_He": "frac",
    "shocked_fraction": "frac", "time_since_shock": "t", "density_time": "rho_t",
    "entropy_initial": "entropy",
    # dual-energy internal energy density of casa_xfit / casa_4dvar save_state
    # files: volumetric like press (dropping it broke casa_xfit_state.load_state)
    "internal_energy": "press",
}
#: the per-mass (intensive) fields: remapped as rho * q
PER_MASS = ("frac", "t", "rho_t", "entropy")
MSUN_G = float(const.M_sun.cgs.value)
M_H_G = 1.6735575e-24
E_UNIT_1E51 = float((1.0 * u.Msun * (1000.0 * u.km / u.s) ** 2).to(u.erg).value) / 1e51


def factors(L, T, M, gamma=GAMMA):
    """Multiplicative factor per field kind (works with numpy or jax scalars)."""
    f_rho = M / L ** 3
    f_v = L / T
    return dict(rho=f_rho, v=f_v, press=f_rho * f_v ** 2, B=(f_rho ** 0.5) * f_v,
                frac=1.0, t=T, rho_t=f_rho * T, entropy=f_rho ** (1.0 - gamma) * f_v ** 2,
                energy=M * f_v ** 2, mass=M, length=L, time=T, wind_A=M / L)


def scale_fit(fit, L, T, M, gamma=GAMMA):
    """``casa_pluto.fit_ambient``'s dict (keys without the ``ambient_`` prefix)
    rescaled by (L, T, M)."""
    pre = {f"ambient_{k}": v for k, v in fit.items()}
    out = scale_ambient(pre, L, T, M, gamma)
    return {k[len("ambient_"):]: v for k, v in out.items()}


def scale_ambient(meta, L, T, M, gamma=GAMMA):
    """Rescale the ``ambient_*`` / ``n_w`` bookkeeping of a state dict (in place
    on a copy; returns the copy). The wind keeps its reference radius."""
    out = dict(meta)
    F = factors(L, T, M, gamma)
    if "ambient_rho_w" in out:
        out["ambient_rho_w"] = float(out["ambient_rho_w"]) * F["wind_A"]
    if "ambient_p_w" in out:
        # p'(r) = F_p p_w (r / (L r_ref))^slope
        out["ambient_p_w"] = (float(out["ambient_p_w"]) * F["press"]
                              * L ** (-float(out.get("ambient_p_slope", 0.0))))
    if "ambient_rho_sh" in out:
        out["ambient_rho_sh"] = float(out["ambient_rho_sh"]) * F["rho"]
    for k in ("ambient_r_sh", "ambient_sigma_sh", "ambient_H_sh"):
        if k in out:
            out[k] = float(out[k]) * L
    for k in ("ambient_M_shell_msun", "ambient_M_shell_in_box_msun"):
        if k in out:
            out[k] = float(out[k]) * M
    if "n_w" in out:
        out["n_w"] = float(out["n_w"]) * F["wind_A"]
    return out


def csm_composition_of(meta):
    try:
        return ast.literal_eval(str(meta["csm_composition"]))
    except (KeyError, ValueError, SyntaxError):
        return dict(CSM_COMPOSITION)


def ambient_code_fields(meta, n, box, gamma=GAMMA):
    """The analytic ambient (wind + cell-averaged shell) of ``meta`` on the
    destination grid, code units: (rho, press)."""
    from casa_pluto import ambient_density, ambient_pressure
    cu = snr_code_units()
    rho_c = (1.0 * cu.code_density).to(u.g / u.cm ** 3).value
    p_c = (1.0 * cu.code_pressure).to(u.erg / u.cm ** 3).value
    fit = {k[len("ambient_"):]: float(meta[k]) for k in meta if k.startswith("ambient_")
           and np.ndim(meta[k]) == 0 and k != "ambient_shell_rel_rms"}
    x = (np.arange(n) + 0.5) * box / n - box / 2
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    rho = ambient_density(r, X, Y, Z, fit, dx=box / n)
    p = ambient_pressure(r, fit, rho=rho)
    return rho / rho_c, p / p_c


def state_budget(d, gamma=GAMMA):
    """Mass (Msun) and energy (1e51 erg) of a state dict (code units pc / Msun / 1000 km/s)."""
    box, n = float(d["box"]), int(d["num_cells"])
    dv = (box / n) ** 3
    rho = np.asarray(d["rho"], np.float64)
    cej = np.clip(np.asarray(d.get("C_ej", np.zeros_like(rho)), np.float64), 0, 1)
    ek = 0.5 * rho * sum(np.asarray(d[k], np.float64) ** 2 for k in ("vx", "vy", "vz"))
    et = np.asarray(d["press"], np.float64) / (gamma - 1.0)
    out = dict(M_tot=rho.sum() * dv, M_ej=(cej * rho).sum() * dv,
               E_kin=ek.sum() * dv * E_UNIT_1E51, E_th=et.sum() * dv * E_UNIT_1E51,
               E_kin_ej=(cej * ek).sum() * dv * E_UNIT_1E51)
    if "shocked_fraction" in d:
        sf = np.clip(np.asarray(d["shocked_fraction"], np.float64), 0, 1)
        out["M_ej_sh"] = (cej * sf * rho).sum() * dv
        out["M_csm_sh"] = ((1 - cej) * sf * rho).sum() * dv
    out["E_tot"] = out["E_kin"] + out["E_th"]
    return out


def wind_nh3(meta, r_pc=3.0):
    """Pre-shock hydrogen density of the wind at ``r_pc`` (cm^-3), with the H
    fraction ``casa_pluto_diff.csm_hydrogen_fraction`` uses."""
    try:
        comp = ast.literal_eval(str(meta["csm_composition"]))
    except (KeyError, ValueError, SyntaxError):
        from casa_pluto import solar_csm_composition
        comp = solar_csm_composition()
    x_h = 1.0 - sum(v for k, v in comp.items() if k != "H")
    return x_h * float(meta["ambient_rho_w"]) * (float(meta["ambient_r_ref"]) / r_pc) ** 2 / M_H_G
# =============================================================================
# ============ ↑ The similarity map ↑ =========================================
# =============================================================================


# =============================================================================
# ============ ↓ Conservative rescaling of a saved state ↓ ====================
# =============================================================================
def rescale_state(d, L, T, M, n_out=None, method="conservative", gamma=GAMMA, log=print):
    """A saved state (dict of a ``casa_pluto convert`` / ``casa_orlando
    --save-state`` npz) rescaled by (L, T, M) onto an ``n_out``^3 grid of the
    same box. Returns the new dict (same format; ``age`` = T x age)."""
    from casa_pluto import overlap_matrix, _apply_separable
    box, n = float(d["box"]), int(d["num_cells"])
    n_out = int(n_out or n)
    F = factors(L, T, M, gamma)
    fields = [k for k in FIELD_KIND if k in d and np.ndim(d[k]) == 3]
    rho = np.asarray(d["rho"], np.float64)
    meta_new = scale_ambient({k: v for k, v in d.items() if np.ndim(v) < 3}, L, T, M, gamma)
    amb_rho, amb_p = ambient_code_fields(meta_new, n_out, box, gamma)
    comp = csm_composition_of(d)
    out = {}
    if method == "conservative":
        src = np.linspace(-0.5 * box, 0.5 * box, n + 1) * L
        dst = np.linspace(-0.5 * box, 0.5 * box, n_out + 1)
        W = overlap_matrix(src, dst)
        Wc = (W, W, W)
        c1 = W.sum(1)
        cover = np.einsum("a,b,c->abc", c1, c1, c1)
        remap = lambda q: _apply_separable(q, Wc)                         # noqa: E731
        mass = F["rho"] * remap(rho) + (1.0 - cover) * amb_rho
        mom = {k: F["rho"] * F["v"] * remap(rho * np.asarray(d[k], np.float64))
               for k in ("vx", "vy", "vz")}
        eint = F["press"] * remap(np.asarray(d["press"], np.float64)) + (1.0 - cover) * amb_p
        out["rho"] = mass
        for k in ("vx", "vy", "vz"):
            out[k] = mom[k] / mass
        out["press"] = eint
        for k in fields:
            kind = FIELD_KIND[k]
            if k in ("rho", "press", "vx", "vy", "vz"):
                continue
            q = np.asarray(d[k], np.float64)
            if kind == "press":                                           # internal_energy
                out[k] = F["press"] * remap(q) + (1.0 - cover) * amb_p / (gamma - 1.0)
                continue
            if kind in PER_MASS:
                fill = comp.get(k[2:], 0.0) if k.startswith("C_") and k != "C_ej" else 0.0
                if kind == "entropy":
                    fill = 0.0
                m_q = F["rho"] * F[kind] * remap(rho * q) + (1.0 - cover) * amb_rho * fill
                out[k] = m_q / mass
            else:                                                          # B: volume average
                out[k] = F[kind] * remap(q)
        # budget of what the crop dropped (L > 1: stretched source beyond the box)
        dv_src = (box / n * L) ** 3
        m_src = F["rho"] * rho.sum() * dv_src
        m_kept = F["rho"] * (remap(rho).sum()) * (box / n_out) ** 3
        log(f"[rescale] conservative remap {n}^3 (dx {box / n:.4f} pc -> stretched {box / n * L:.4f})"
            f" -> {n_out}^3 (dx {box / n_out:.4f}); source mass x M {m_src:.4f} Msun, kept in box "
            f"{m_kept:.4f} ({100 * (m_kept / m_src - 1):+.3f} %: the wind beyond the box)")
    elif method == "linear":
        from scipy.ndimage import map_coordinates
        x = (np.arange(n_out) + 0.5) * box / n_out - box / 2
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        idx = [(A / L + 0.5 * box) / (box / n) - 0.5 for A in (X, Y, Z)]
        inside = np.max(np.abs(np.stack([X, Y, Z]) / L), axis=0) <= 0.5 * box - 0.5 * box / n
        samp = lambda q: map_coordinates(np.asarray(q, np.float64), idx, order=1, mode="nearest")  # noqa: E731
        # conserved densities (rho, e_int, rho q), as casa_pluto_diff.similarity_resample
        rho_s = samp(rho)
        rho_new = np.where(inside, F["rho"] * rho_s, amb_rho)
        for k in fields:
            kind = FIELD_KIND[k]
            if k == "rho":
                out[k] = rho_new
            elif k == "press":
                out[k] = np.where(inside, F["press"] * samp(d[k]), amb_p)
            elif kind == "press":                                           # internal_energy
                out[k] = np.where(inside, F["press"] * samp(d[k]), amb_p / (gamma - 1.0))
            elif kind == "B":
                out[k] = F[kind] * samp(d[k])
            else:                              # per-mass: rho q, face value outside
                q = samp(rho * np.asarray(d[k], np.float64)) / np.maximum(rho_s, 1e-30)
                out[k] = F[kind] * q
    else:
        raise ValueError(f"method {method!r}")
    new = dict(meta_new)
    new.update({k: np.asarray(v, np.float32) for k, v in out.items()})
    new["num_cells"] = n_out
    for k in ("age", "age_target", "map_age", "ic_age"):
        if k in d:
            new[k] = float(d[k]) * T
    # casa_xfit / casa_4dvar save_state files: the state stays at its calendar
    # epoch, so the explosion date moves (age' = T age = epoch - t_expl'), in the
    # key AND in the stored theta; per-cone radii are lengths
    if "t_expl" in d and "epoch_year" in d and "age" in d:
        new["t_expl"] = float(d["epoch_year"]) - float(d["age"]) * T
        if "theta" in d and "names" in d and "t_expl" in [str(x) for x in d["names"]]:
            th = np.array(d["theta"], dtype=np.float64)
            th[[str(x) for x in d["names"]].index("t_expl")] = new["t_expl"]
            new["theta"] = th
        log(f"[rescale] save_state file: epoch {float(d['epoch_year']):.2f} kept, t_expl "
            f"{float(d['t_expl']):.2f} -> {new['t_expl']:.2f}")
    for k in ("r_fs_pc", "r_rs_pc"):
        if k in d:
            new[k] = np.asarray(d[k], np.float64) * L
    prev = [float(d.get(f"similarity_{k}", 1.0)) for k in "LTM"]
    new["similarity_L"], new["similarity_T"], new["similarity_M"] = prev[0] * L, prev[1] * T, prev[2] * M
    new["similarity_age_unscaled"] = float(d.get("similarity_age_unscaled", d["age"]))
    new["similarity_method"] = np.array(method)
    return new


def report(d_old, d_new, L, T, M, log=print):
    F = factors(L, T, M)
    b0, b1 = state_budget(d_old), state_budget(d_new)
    log(f"[rescale] L {L:.4f}  T {T:.4f}  M {M:.4f}:  E x {F['energy']:.4f}, masses x {M:.4f}, "
        f"v x {F['v']:.4f}, wind rho r^2 x {F['wind_A']:.4f}, age {float(d_old['age']):.2f} -> "
        f"{float(d_new['age']):.2f} yr")
    for k in ("M_tot", "M_ej", "M_ej_sh", "M_csm_sh", "E_kin", "E_th", "E_tot", "E_kin_ej"):
        if k in b0:
            fac = F["energy"] if k.startswith("E") else M
            log(f"   {k:9s} {b0[k]:9.4f} -> {b1[k]:9.4f}   (x{b1[k] / b0[k]:.4f}; exact map x{fac:.4f};"
                f" {100 * (b1[k] / (b0[k] * fac) - 1):+.3f} %)")
    if "ambient_rho_w" in d_old:
        log(f"   wind n_H(3 pc) {wind_nh3(d_old):.3f} -> {wind_nh3(d_new):.3f} cm^-3; shell r "
            f"{float(d_old['ambient_r_sh']):.3f} -> {float(d_new['ambient_r_sh']):.3f} pc, mass "
            f"{float(d_old['ambient_M_shell_msun']):.3f} -> {float(d_new['ambient_M_shell_msun']):.3f} Msun")
    return b0, b1
# =============================================================================
# ============ ↑ Conservative rescaling of a saved state ↑ ====================
# =============================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ic", required=True, help="state npz (casa_pluto convert / casa_orlando --save-state)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--L", type=float, default=1.0, help="length factor")
    ap.add_argument("--T", type=float, default=1.0, help="time factor (the new age is T x age)")
    ap.add_argument("--M", type=float, default=1.0, help="mass factor")
    ap.add_argument("--n", type=int, default=None, help="destination cells per axis (default: same)")
    ap.add_argument("--method", choices=("conservative", "linear"), default="conservative")
    ap.add_argument("--info", action="store_true", help="only print the implied physical parameters")
    args = ap.parse_args()
    d = dict(np.load(args.ic))
    F = factors(args.L, args.T, args.M)
    b0 = state_budget(d)
    print(f"[rescale] {args.ic}: age {float(d['age']):.2f} yr, E {b0['E_tot']:.4f}e51, M_ej {b0['M_ej']:.3f}")
    print(f"[rescale] implied: E' {b0['E_tot'] * F['energy']:.3f}e51 erg, M_ej' {b0['M_ej'] * args.M:.3f} Msun, "
          f"age' {float(d['age']) * args.T:.2f} yr, wind n_H(3 pc) {wind_nh3(d) * F['wind_A']:.3f} cm^-3, "
          f"shell at {float(d.get('ambient_r_sh', np.nan)) * args.L:.3f} pc")
    if args.info:
        return
    new = rescale_state(d, args.L, args.T, args.M, n_out=args.n, method=args.method)
    report(d, new, args.L, args.T, args.M)
    new["argv"] = np.array("casa_rescale.py " + " ".join(f"--{k} {v}" for k, v in vars(args).items()))
    if args.out:
        np.savez_compressed(args.out, **new)
        print(f"[rescale] wrote {args.out}")


if __name__ == "__main__":
    main()
