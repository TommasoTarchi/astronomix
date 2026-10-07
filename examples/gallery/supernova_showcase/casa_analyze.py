"""
Measure a saved 3D remnant against the Cas A observations.

Runs on the CPU from a ``--save-state`` npz, so a finished (or a mid-flight)
simulation can be scored without touching a GPU. Produces the diagnostics
Orlando et al. use to accept or reject a model:

  * **angle-averaged r_FS and r_RS**, using the same two criteria as the 1D
    calibration -- a density contrast against the known analytic wind, and the
    departure from homologous expansion -- so 1D and 3D are directly comparable.
    (The showcase's old estimator read the reverse shock off a median-temperature
    profile and broke as soon as the interior cooled: it has reported anything
    from 0.01 to 1.09 pc for states whose real r_RS is ~1.5 pc.)
  * **r_FS versus position angle in the plane of the sky**, which Orlando et al.
    (2022) identify as the single most discriminating diagnostic for the
    circumstellar-shell parameters. The Earth vantage point is on the -y axis,
    so the plane of the sky is (x, z).
  * the **radial density profile** against the calibrated 1D solution, which
    separates "the 3D run disagrees with the observations" from "the 3D run
    disagrees with its own 1D calibration" -- two very different failures.

Usage::

    CUDA_VISIBLE_DEVICES= ./run.sh casa_analyze.py \\
        /export/data/lstorcks/supernova_showcase/orl_n256_shell.npz \\
        --profile casa_1d_fiducial_350yr.npz --label shell
"""

# ==== CPU only ====
# The showcase scripts call autocvd when CUDA_VISIBLE_DEVICES is unset, which
# blocks waiting for a free GPU. Setting it empty (rather than leaving it None)
# is the documented way to import them for CPU-side analysis.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# ruff: noqa: E402
# ==================

# general
import argparse
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# units and constants
from astropy import units as u
import astropy.constants as const

# shared showcase helpers
from _common import (FIGURES_DIR, MASS_PER_NUCLEUS, snr_code_units, temperature_K,
                     wind_asymmetry_field)
from casa_orlando import (measure_shocks_3d, shock_speed_vs_position_angle,
                          position_angle_statistics, SHELL_THETA_DEG, SHELL_PHI_DEG)


# observational targets (see casa_calibrate_1d.py for the references)
OBS = dict(r_fs=(2.52, 0.20), r_rs=(1.58, 0.16), v_fs=(5250.0, 250.0),
           n_post=(4.0, 1.0))
DISTANCE_KPC = 3.4


def load(path):
    d = np.load(path)
    return dict(rho=np.asarray(d["rho"], dtype=np.float64),
                press=np.asarray(d["press"], dtype=np.float64),
                vx=np.asarray(d["vx"]) if "vx" in d else None,
                vy=np.asarray(d["vy"]) if "vy" in d else None,
                vz=np.asarray(d["vz"]) if "vz" in d else None,
                box=float(d["box"]), age=float(d["age"]),
                n=int(d["num_cells"]))


def coords(box, n):
    c = (np.arange(n) + 0.5) / n * box - box / 2.0
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    return np.sqrt(X ** 2 + Y ** 2 + Z ** 2), X, Y, Z


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="+", help="one or more --save-state npz files")
    ap.add_argument("--labels", nargs="*", default=None, help="legend labels")
    ap.add_argument("--profile", default=None,
                    help="the calibrated 1D profile at the same age, for comparison")
    ap.add_argument("--n-w", type=float, default=None,
                    help="wind density at r_fs_ref. Default: the value stamped in "
                         "the state, else the one in --profile, else 0.928")
    ap.add_argument("--r-fs-ref", type=float, default=None)
    ap.add_argument("--n-c", type=float, default=None)
    ap.add_argument("--profile-rmax", type=float, default=None,
                    help="radius beyond which the mapped 1D profile was HELD "
                         "constant (default: read from --profile). The shock "
                         "detector compares against an analytic r^-2 wind; "
                         "outside this radius the grid does not follow one, and "
                         "without this the corners read as a shock")
    ap.add_argument("--out", default="casa_analysis", help="figure name stem")
    args = ap.parse_args()

    profile_rmax = args.profile_rmax
    if profile_rmax is None and args.profile is not None:
        profile_rmax = float(np.load(args.profile)["r"][-1])
    if profile_rmax is None:
        print("[analyze] NOTE: no --profile/--profile-rmax, so the ambient "
              "reference is the pure r^-2 wind everywhere. Radii near the box "
              "edge should be distrusted.")

    cu = snr_code_units()
    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3).to(cu.code_density).value)
    labels = args.labels or [Path(s).stem for s in args.states]

    # The detector's ambient reference must be the wind the run actually used.
    # A CLI default of 0.928 silently mis-references a state calibrated with a
    # different n_w, so resolve it from the state, then the profile, then the CLI.
    prof_cfg = {}
    if args.profile is not None:
        dp = np.load(args.profile)
        prof_cfg = {k[4:]: float(dp[k]) for k in dp.files if k.startswith("cfg_")}

    def wind_params(d_state):
        out = {}
        for key, cli in (("n_w", args.n_w), ("r_fs_ref", args.r_fs_ref), ("n_c", args.n_c)):
            if cli is not None:
                out[key] = float(cli)
            elif key in d_state:
                out[key] = float(d_state[key])
            elif key in prof_cfg:
                out[key] = prof_cfg[key]
            else:
                out[key] = dict(n_w=0.928, r_fs_ref=2.5, n_c=0.1)[key]
        return out

    def asymmetry_reference(d_state, X, Y, Z):
        """Rebuild the --wind-asym modulation so the per-cone detector sees the
        same anisotropic ambient the run did (CALIBRATION.md Results 25-26).
        Newer states stamp the parameters; older ones only carry argv."""
        if "wind_asym_dipole" in d_state:
            a1, a2 = float(d_state["wind_asym_dipole"]), float(d_state["wind_asym_quadrupole"])
            th, ph = float(d_state["wind_asym_theta_deg"]), float(d_state["wind_asym_phi_deg"])
        elif "argv" in d_state:
            toks = str(np.asarray(d_state["argv"]).item()).split()
            def flag(name, default):
                return float(toks[toks.index(name) + 1]) if name in toks else default
            a1 = flag("--wind-asym", 0.0); a2 = flag("--wind-asym-quad", 0.0)
            th = flag("--wind-asym-theta", SHELL_THETA_DEG)
            ph = flag("--wind-asym-phi", SHELL_PHI_DEG)
        else:
            return None, (0.0, 0.0)
        if a1 == 0.0 and a2 == 0.0:
            return None, (0.0, 0.0)
        f = np.asarray(wind_asymmetry_field(X, Y, Z, dipole=a1, quadrupole=a2,
                                            theta_deg=th, phi_deg=ph))
        print(f"[analyze] {Path(str(d_state.fid.name) if hasattr(d_state, 'fid') else '')} "
              f"wind asymmetry A1 {a1:+.2f} A2 {a2:+.2f} about ({th:.0f}, {ph:.0f}) deg "
              f"-- per-cone detector referenced to it")
        return f, (a1, a2)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    ax_prof, ax_pa, ax_sc = axes
    summary = []

    for path, label in zip(args.states, labels):
        st = load(path)
        r, X, Y, Z = coords(st["box"], st["n"])
        r_safe = np.maximum(r, 0.5 * st["box"] / st["n"])
        if st["vx"] is None:
            raise SystemExit(f"{path} has no velocities -- the reverse-shock "
                             f"criterion needs them; re-save with a newer --save-state")
        v_r = (st["vx"] * X + st["vy"] * Y + st["vz"] * Z) / r_safe

        d_all = np.load(path)
        f_sh = np.asarray(d_all["shocked_fraction"]) if "shocked_fraction" in d_all else None
        f_ej = np.asarray(d_all["C_ej"]) if "C_ej" in d_all else None
        wind = wind_params(d_all)
        asym, (a1, a2) = asymmetry_reference(d_all, X, Y, Z)
        state_rmax = float(d_all["r_profile_max"]) if "r_profile_max" in d_all else np.nan
        flat_beyond = profile_rmax if profile_rmax is not None else (
            state_rmax if np.isfinite(state_rmax) else None)
        if "mass_conserved" in d_all and not bool(d_all["mass_conserved"]):
            print(f"[analyze] WARNING: {path} is stamped MASS NOT CONSERVED; its radii "
                  "are not physical")
        m = measure_shocks_3d(st["rho"], v_r, r, age_yr=st["age"], code_units=cu,
                              rho_per_n=rho_per_n, shocked_fraction=f_sh,
                              ejecta_fraction=f_ej, r_max=0.5 * st["box"],
                              ambient_flat_beyond=flat_beyond, **wind)
        angles, r_pa = shock_speed_vs_position_angle(
            st["rho"], r, X, Y, Z, rho_per_n=rho_per_n, r_max=0.5 * st["box"],
            ambient_flat_beyond=flat_beyond, asym=asym, **wind)
        pa = position_angle_statistics(angles, r_pa)

        T = temperature_K(st["rho"], st["press"], cu)
        summary.append(dict(label=label, age=st["age"], r_fs=m["r_fs"], r_rs=m["r_rs"],
                            pa_spread=pa["spread"], pa_std=pa["std"], pa_m1=pa["m1"],
                            pa_m2=pa["m2"], pa_m1_deg=pa["m1_pa_deg"], a1=a1, a2=a2,
                            n_w=wind["n_w"],
                            T_max=float(np.nanmax(T)), rho_max=float(st["rho"].max())))

        ax_prof.semilogy(m["rc"], m["rho_mean"] / rho_per_n, lw=1.4, label=label)
        ax_pa.plot(angles, r_pa, lw=1.4, label=label)

    if args.profile is not None:
        d = np.load(args.profile)
        ax_prof.semilogy(np.asarray(d["r"]), np.asarray(d["rho"]) / rho_per_n,
                         "k--", lw=1.0, label="1D calibration")

    # ambient wind for reference
    ref = wind_params(np.load(args.states[-1]))    # the last state's resolved wind
    rr = np.linspace(0.02, ref["r_fs_ref"] * 1.4, 400)
    ax_prof.semilogy(rr, ref["n_w"] * (ref["r_fs_ref"] / rr) ** 2 + ref["n_c"],
                     color="0.6", ls=":", lw=1.0, label="unshocked wind")
    ax_prof.set(xlabel="r [pc]", ylabel=r"$\langle n\rangle$ [cm$^{-3}$]",
                xlim=(0, 3.4), ylim=(1e-2, 1e3), title="angle-averaged density")
    ax_prof.legend(fontsize=8)

    for key, col in (("r_fs", "tab:red"), ("r_rs", "tab:orange")):
        val, tol = OBS[key]
        ax_pa.axhspan(val - tol, val + tol, color=col, alpha=0.15)
        ax_pa.axhline(val, color=col, ls=":", lw=1.0)
    ax_pa.set(xlabel="position angle in the plane of the sky [deg]",
              ylabel="$r_{FS}$ [pc]", title="forward shock vs position angle\n"
                                            "(red band = observed $r_{FS}$)")
    ax_pa.legend(fontsize=8)

    # scorecard
    ax_sc.axis("off")
    lines = [f"{'model':<22}{'age':>5}{'r_FS':>7}{'r_RS':>7}{'FS/RS':>7}"
             f"{'spread':>7}{'std':>6}{'m=1':>6}{'m=2':>6}{'A1':>5}"]
    lines.append("-" * 84)
    for s in summary:
        ratio = s["r_fs"] / s["r_rs"] if s["r_rs"] and np.isfinite(s["r_rs"]) else np.nan
        lines.append(f"{s['label'][:21]:<22}{s['age']:>5.0f}{s['r_fs']:>7.3f}"
                     f"{s['r_rs']:>7.3f}{ratio:>7.3f}{s['pa_spread']:>7.3f}"
                     f"{s['pa_std']:>6.3f}{s['pa_m1']:>6.3f}{s['pa_m2']:>6.3f}{s['a1']:>5.2f}")
    lines.append("-" * 84)
    lines.append(f"{'OBSERVED':<22}{350:>5}{OBS['r_fs'][0]:>7.2f}{OBS['r_rs'][0]:>7.2f}"
                 f"{OBS['r_fs'][0] / OBS['r_rs'][0]:>7.3f}{'0.2-0.4':>7}")
    lines.append("")
    lines.append("spread = max - min r_FS over position angle (pc), an extreme-value")
    lines.append("statistic quantised to ~1 cell before 2026-09-02; std, m=1 and m=2")
    lines.append("are the PA standard deviation and the lopsided / elliptical Fourier")
    lines.append("amplitudes of r_FS(PA), which a dipole / quadrupole ambient predicts.")
    lines.append("The observed 0.2-0.4 pc range is UNSOURCED in this directory; see")
    lines.append("CALIBRATION.md Result 26 before scoring against it.")
    ax_sc.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=8.5,
               va="top", ha="left", transform=ax_sc.transAxes)

    out = Path(FIGURES_DIR) / f"{args.out}.png"
    fig.savefig(out, dpi=150)
    print("\n".join(lines))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
