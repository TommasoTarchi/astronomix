"""
Orlando's Cas A state at ~150 yr (PLUTO) -> analysis and an astronomix initial condition.

S. Orlando provided one snapshot of his 3D MHD Cas A model (PLUTO 4.3, AMR at
2048^3 effective, delivered as 512^3 float32 ``.flt`` files, one per variable)
at t = 0.1486 code = **145.6 yr** after the explosion. It lives in
``/export/data/lstorcks/casa_orlando150`` (6.9 GB, downloaded from the shared
Google Drive folder with ``gdown``). This module

  * reads it (memory-mapped, one field at a time, in astronomix's ``[x, y, z]``
    index order -- PLUTO writes x fastest, so the raw array is ``[z, y, x]``);
  * documents what each of the 19 tracers holds (:data:`TRACER_KEY`) and how
    the eleven element tracers fold into the pipeline's five scalars;
  * fits the ambient medium: an r^-2 wind plus the asymmetric shell of
    Orlando et al. (2022) Eq. 1, which the snapshot reproduces to 5e-4 rms.
    The shell sits at 1.5 pc, i.e. mostly OUTSIDE PLUTO's +-1.33 pc box, so
    the fit is what lets the state be embedded in a box large enough to evolve
    to 2023;
  * remaps the state conservatively onto a larger astronomix grid (mass,
    momentum, internal energy and every scalar's mass are conserved exactly;
    kinetic energy below the target cell scale is lost and reported), fills the
    rest of the box from the fitted ambient model, and writes a npz in the same
    format ``casa_orlando.py --save-state`` does, so ``casa_orlando.py
    --from-state``, ``casa_analyze.py``, ``casa_plasma.py`` and
    ``casa_observe.py`` all consume it unchanged.

Usage::

    # CPU, a few minutes: diagnostics + figure
    CUDA_VISIBLE_DEVICES= ./run.sh casa_pluto.py analyze

    # CPU, ~5 min at 512^3: the astronomix initial condition
    CUDA_VISIBLE_DEVICES= ./run.sh casa_pluto.py convert --n 256 \\
        --out /export/data/lstorcks/casa_orlando150/work/pluto146_n256.npz

    # similarity-rescaled (casa_rescale): lengths x L, times x T, masses x M
    CUDA_VISIBLE_DEVICES= ./run.sh casa_pluto.py convert --n 256 --sim 1.40 1.20 1.00 \\
        --out /export/data/lstorcks/casa_orlando150/work/pluto146_n256_sim.npz

    # evolve it (GPU)
    ./run.sh casa_orlando.py --from-state .../pluto146_n256.npz --composition \\
        --age 342 --snapshot-ages 319 323 331 337 --positivity redistribute ...
"""

# ==== CPU only (as a script; importers such as casa_pluto_diff keep their GPU) ====
import os
if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
# ruff: noqa: E402
# ==================

# general
import argparse
import re
from pathlib import Path

# numerics
import numpy as np
from scipy.optimize import least_squares

# units and constants
from astropy import units as u
import astropy.constants as const

# shared showcase helpers
from _common import CSM_COMPOSITION, FIGURES_DIR, GAMMA, snr_code_units

PLUTO_DIR = Path("/export/data/lstorcks/casa_orlando150")
M_P = const.m_p.cgs.value
MSUN = const.M_sun.cgs.value
PC = const.pc.cgs.value
YR = (1.0 * u.yr).to(u.s).value


# =============================================================================
# ============ ↓ What the 19 tracers are ↓ ====================================
# =============================================================================
#: Identified from the data, NOT from a header (the delivery has no init.c).
#: Evidence, measured on the 146-yr snapshot:
#:
#: * ``tr1``  shock time t_sh (code time). 0 in never-shocked gas, ~0.134 just
#:   behind the forward shock (t = 0.1486), ~0.04 in the CSM next to the contact
#:   discontinuity -- the Orlando et al. (2015) ionization-age bookkeeping.
#: * ``tr2``  shell material marker: non-zero only at r = 1.41-1.58 pc in the
#:   unshocked CSM, i.e. exactly where the Eq. 1 shell sits.
#: * ``tr3``  tiny negative values almost everywhere: most likely the radiative
#:   loss record of the tabulated cooling. Not used.
#: * ``tr4``  ejecta fraction (3.25 Msun in the box).
#: * ``tr5-7`` direction-like labels correlated 0.88 with x, y, z; ``tr8`` ~ the
#:   parcel's speed at the last shock (= |v| to 2 % in unshocked ejecta). Not used.
#: * ``tr9-19`` eleven element mass fractions. They sum to 1 in pure ejecta
#:   and to 0 in the CSM (the CSM is not tagged). The identification below is by
#:   mass, radial ordering, peak fraction and co-location, against a 15 Msun IIb
#:   neutrino-driven explosion (Wongwathanarat et al. 2017, the model Orlando
#:   et al. 2021/2022 evolve). He, O, Fe, Ne, C and Ca are confident; H, Si
#:   and the three minor ones are not. CONFIRM WITH ORLANDO.
TRACER_KEY = {
    "tr1": "t_shock", "tr2": "shell", "tr3": "cooling_loss?", "tr4": "ejecta",
    "tr5": "label_x?", "tr6": "label_y?", "tr7": "label_z?", "tr8": "v_shock?",
    #       element     ejecta mass [Msun], evidence
    "tr9": "C",       # 0.157, peaks at 0.49 with He 0.22 + O 0.25: the He/C shell
    "tr10": "Fe-group",  # 0.059, innermost of all (<v0> = 3.0e3 km/s), with Fe: 'X'/Ni?
    "tr11": "H?",     # 0.181, part outermost, part mixed inward; anti-corr. with He
    "tr12": "He",     # 1.512, 97 % pure in the outermost ejecta
    "tr13": "Mg?",    # 0.038, co-located with O (0.54)
    "tr14": "Ne",     # 0.124, peaks at 0.20 inside O-rich (0.63) gas
    "tr15": "Si",     # 0.058, co-located with Fe; tr18/tr15 = 0.05 everywhere
    "tr16": "O",      # 0.562, peaks at 0.76 with Ne 0.15 and C 0.10
    "tr17": "S/Ar?",  # 0.029, inner, co-located with tr10/tr15/Fe
    "tr18": "Ca",     # 0.003, locked to Si at 5 % by mass
    "tr19": "Fe",     # 0.091, peak 0.60 in the Fe-rich knots
}

#: How the eleven element tracers fold into the pipeline's four scalars
#: (``_plasma.TRACER_SPLIT``: the "O" scalar stands for O/Ne/Mg, "Si" for
#: Si/S/Ar/Ca). Hydrogen is the remainder, so anything NOT listed here (tr11)
#: becomes hydrogen, which is what the key says it is.
PIPELINE_GROUPS = {
    "Fe": ("tr19", "tr10"),
    "Si": ("tr15", "tr17", "tr18"),
    "O": ("tr16", "tr14", "tr13", "tr9"),
    "He": ("tr12",),
}
ELEMENT_TRACERS = tuple(f"tr{i}" for i in range(9, 20))
# =============================================================================
# ============ ↑ What the 19 tracers are ↑ ====================================
# =============================================================================


# =============================================================================
# ============ ↓ Reading the PLUTO delivery ↓ =================================
# =============================================================================
class PlutoSnapshot:
    """One PLUTO ``.flt`` output: grid, units, and memory-mapped fields."""

    def __init__(self, directory=PLUTO_DIR, index=0):
        self.dir = Path(directory)
        self.index = index
        self._read_units()
        self._read_grid()
        self._read_flt_out()

    def _read_units(self):
        text = (self.dir / "definitions.h").read_text()

        def const_(name):
            return float(re.search(rf"#define\s+{name}\s+([0-9.eE+-]+)", text).group(1))

        self.unit_length = const_("UNIT_LENGTH")           # cm (PLUTO's "pc" = 3.09e18)
        self.unit_density = const_("UNIT_DENSITY")         # g cm^-3
        self.unit_velocity = const_("UNIT_VELOCITY")       # cm s^-1
        self.unit_time = self.unit_length / self.unit_velocity
        self.unit_pressure = self.unit_density * self.unit_velocity ** 2
        # PLUTO's B unit: sqrt(4 pi rho_u) v_u, with p_mag = B_code^2 / 2
        self.unit_b = np.sqrt(4.0 * np.pi * self.unit_density) * self.unit_velocity
        self.mu = const_("MU_CASA")

    def _read_grid(self):
        lines = (self.dir / "grid.out").read_text().splitlines()
        body = [ln for ln in lines if not ln.startswith("#")]
        axes, i = [], 0
        while len(axes) < 3:
            n = int(body[i]); i += 1
            edges = np.array([[float(v) for v in body[i + k].split()[1:3]] for k in range(n)])
            axes.append(edges); i += n
        self.n = axes[0].shape[0]
        # The two columns are NOT the cell edges: their difference is the fine
        # (2048^3) spacing, a quarter of the row-to-row step. The delivery is the
        # 2048^3 run reduced to 512^3, and the first column is the reduced cell's
        # centre (-1.3284 = -1.331 + 2 fine cells). Taking the columns as edges
        # made the last cell a quarter-width and lost 0.13 % of the mass.
        self.edges_cm = []
        for a in axes:
            c = a[:, 0]
            d = float(np.median(np.diff(c)))
            self.edges_cm.append(np.append(c - 0.5 * d, c[-1] + 0.5 * d) * self.unit_length)

    def _read_flt_out(self):
        line = (self.dir / "flt.out").read_text().split("\n")[self.index].split()
        self.time_code = float(line[1])
        self.age_yr = self.time_code * self.unit_time / YR
        # "<n> <t> <dt> <step> <single|multiple>_files <little|big> var..."
        self.endian = "<" if line[5] == "little" else ">"
        self.variables = line[6:]

    def field(self, name, stride=1):
        """A field as ``[x, y, z]`` float32 (a transposed memmap view, strided)."""
        n = self.n
        mm = np.memmap(self.dir / f"{name}.{self.index:04d}.flt",
                       dtype=f"{self.endian}f4", mode="r", shape=(n, n, n))
        return mm[::stride, ::stride, ::stride].transpose(2, 1, 0)

    def centers_pc(self, stride=1):
        return [0.5 * (e[:-1] + e[1:])[::stride] / PC for e in self.edges_cm]
# =============================================================================
# ============ ↑ Reading the PLUTO delivery ↑ =================================
# =============================================================================


# =============================================================================
# ============ ↓ Ambient medium: r^-2 wind + Orlando (2022) shell ↓ ===========
# =============================================================================
def shell_direction_cosine(X, Y, Z, theta_deg, phi_deg):
    """``r.D`` of Orlando et al. (2022) Eq. 1 (same convention as ``_common``)."""
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return X * np.cos(th) * np.cos(ph) - Y * np.sin(ph) + Z * np.sin(th) * np.cos(ph)


def radial_gaussian(r, r0, sigma, dx=0.0, erf=None):
    """``exp(-(r - r0)^2 / 2 sigma^2)``, averaged over a radial cell width ``dx``.

    The shell's sigma (0.022 pc) is below the cell of every whole-remnant grid
    coarser than 320^3 in 7 pc, and point-sampling a sub-cell Gaussian at cell
    centres gets its mass wrong by an amount that depends on where the centres
    fall. The cell average keeps the column (and so the mass) right at any dx.
    """
    if erf is None:
        from scipy.special import erf
    if dx <= 0.0:
        return np.exp(-0.5 * ((r - r0) / sigma) ** 2)
    s2 = np.sqrt(2.0) * sigma
    return (sigma * np.sqrt(np.pi / 2.0) / dx
            * (erf((r + 0.5 * dx - r0) / s2) - erf((r - 0.5 * dx - r0) / s2)))


def ambient_density(r, X, Y, Z, fit, dx=0.0):
    """Pre-shock ambient density (g cm^-3) from :func:`fit_ambient`'s parameters."""
    wind = fit["rho_w"] * (fit["r_ref"] / np.maximum(r, 1e-3)) ** 2
    shell = (fit["rho_sh"] * radial_gaussian(r, fit["r_sh"], fit["sigma_sh"], dx)
             * np.exp(shell_direction_cosine(X, Y, Z, fit["theta_sh"], fit["phi_sh"])
                      / fit["H_sh"]))
    return wind + shell


def ambient_pressure(r, fit, rho=None, t_max_K=1e5, mu=1.2889):
    """Pre-shock ambient pressure (dyn cm^-2): a power law fitted to the wind.

    With ``rho`` given, capped at ``t_max_K``: the fitted p ~ r^0.8 against
    rho ~ r^-2 is T ~ r^2.8, fine over the 1.2-2.3 pc it was fitted on but
    5e5 K by the box edge, which a temperature-based shock indicator then reads
    as nearly shocked. Dynamically irrelevant either way (Mach > 100).
    """
    p = fit["p_w"] * (np.maximum(r, 1e-3) / fit["r_ref"]) ** fit["p_slope"]
    if rho is not None:
        p = np.minimum(p, rho * const.k_B.cgs.value * t_max_K / (mu * M_P))
    return p


def fit_ambient(snap, stride=2):
    """Fit the unshocked CSM: wind normalisation, pressure law, and the Eq. 1 shell.

    Cells are taken as unshocked CSM when they carry no ejecta and are cold
    (T < 1e6 K at PLUTO's mu). All radii in pc, densities in g cm^-3.
    """
    xc, yc, zc = snap.centers_pc(stride)
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    rho = np.asarray(snap.field("rho", stride), dtype=np.float64) * snap.unit_density
    prs = np.asarray(snap.field("prs", stride), dtype=np.float64) * snap.unit_pressure
    ej = np.asarray(snap.field("tr4", stride), dtype=np.float64)
    T = snap.mu * M_P * prs / (rho * const.k_B.cgs.value)
    cold = (ej < 1e-3) & (T < 1e6)
    r_ref = 2.5

    # wind: away from the shell, rho r^2 is constant to 3e-4
    wind = cold & (np.abs(r - 1.5) > 0.15) & (r > 1.2)
    rho_w = float(np.median(rho[wind] * (r[wind] / r_ref) ** 2))
    lp = np.polyfit(np.log(r[wind] / r_ref), np.log(prs[wind]), 1)
    p_slope, p_w = float(lp[0]), float(np.exp(lp[1]))

    # shell: fit the excess over the wind with Eq. 1
    sel = cold & (np.abs(r - 1.5) < 0.14)
    ex = rho[sel] / rho_w * (r[sel] / r_ref) ** 2 - 1.0     # in wind units at r
    xs, ys, zs, rs = X[sel], Y[sel], Z[sel], r[sel]
    wind_at = (r_ref / rs) ** 2

    def model(q):
        ln_a, r_sh, sig, th, ph, H = q
        return (np.exp(ln_a) * np.exp(-0.5 * ((rs - r_sh) / sig) ** 2)
                * np.exp(shell_direction_cosine(xs, ys, zs, th, ph) / H)) / wind_at

    wt = 1.0 / np.sqrt(np.maximum(ex, 0.0) + 1.0)
    res = least_squares(lambda q: (model(q) - ex) * wt,
                        [np.log(25.0), 1.5, 0.02, 30.0, 50.0, 0.7],
                        bounds=([0, 1.3, 0.005, -360, -90, 0.05], [8, 1.7, 0.2, 360, 90, 10]))
    ln_a, r_sh, sig, th, ph, H = res.x
    resid = float(np.sqrt(np.mean((model(res.x) - ex) ** 2) / np.mean(ex ** 2)))
    fit = dict(rho_w=rho_w, r_ref=r_ref, p_w=p_w, p_slope=p_slope,
               rho_sh=float(np.exp(ln_a)) * rho_w, r_sh=float(r_sh),
               sigma_sh=float(sig), theta_sh=float(th), phi_sh=float(ph),
               H_sh=float(H), shell_rel_rms=resid)
    # the shell's whole mass, analytically (the angular factor integrates to
    # 4 pi sinh(r/H) / (r/H)), and how much of it the PLUTO box actually contains
    fit["M_shell_msun"] = float(fit["rho_sh"] * np.sqrt(2 * np.pi) * sig * r_sh ** 2 * PC ** 3
                                * 4 * np.pi * np.sinh(r_sh / H) / (r_sh / H) / MSUN)
    dv = (np.diff(snap.edges_cm[0])[0] * stride) ** 3
    fit["M_shell_in_box_msun"] = float(np.sum(np.maximum(
        rho[sel] - rho_w * (r_ref / rs) ** 2, 0.0)) * dv / MSUN)
    return fit


def format_ambient(fit, mu):
    n_w = fit["rho_w"] / (mu * M_P)
    n_sh = fit["rho_sh"] / (mu * M_P)
    return (f"wind n = {n_w:.3f} (r / {fit['r_ref']:.1f} pc)^-2 cm^-3 (mu = {mu:.3f}); "
            f"p ~ r^{fit['p_slope']:.2f}\n"
            f"shell (Orlando+22 Eq. 1): n_sh = {n_sh:.2f} cm^-3, r_sh = {fit['r_sh']:.4f} pc, "
            f"sigma = {fit['sigma_sh']:.4f} pc, theta = {fit['theta_sh']:.1f}, "
            f"phi = {fit['phi_sh']:.1f} deg, H = {fit['H_sh']:.3f} pc "
            f"(rel. rms residual {fit['shell_rel_rms']:.1e}); "
            f"M_shell = {fit['M_shell_msun']:.2f} Msun, of which "
            f"{fit['M_shell_in_box_msun']:.2f} inside the PLUTO box")
# =============================================================================
# ============ ↑ Ambient medium: r^-2 wind + Orlando (2022) shell ↑ ===========
# =============================================================================


# =============================================================================
# ============ ↓ Conservative remap onto an astronomix grid ↓ =================
# =============================================================================
def overlap_matrix(src_edges, dst_edges):
    """``W[a, i]`` = fraction of destination cell ``a`` covered by source cell ``i``.

    Rows sum to the covered fraction of the destination cell, so a destination
    cell average of a source density ``q`` is ``W @ q`` plus ``(1 - sum_i W)``
    times whatever fills the uncovered part.
    """
    lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1])
    hi = np.minimum(dst_edges[1:, None], src_edges[None, 1:])
    return np.clip(hi - lo, 0.0, None) / np.diff(dst_edges)[:, None]


def _apply_separable(q, W):
    """Apply per-axis overlap matrices ``(Wx, Wy, Wz)`` to a ``[x, y, z]`` field."""
    out = np.tensordot(W[0], q, axes=(1, 0))                      # [a, y, z]
    out = np.tensordot(out, W[1], axes=(1, 1))                    # [a, z, b]
    out = np.tensordot(out, W[2], axes=(1, 1))                    # [a, b, c]
    return out


def remap_to_grid(snap, box_pc, num_cells, fit, *, gamma=GAMMA, shock_time_min_yr=1.0,
                  csm=CSM_COMPOSITION, log=print, similarity=None):
    """PLUTO snapshot -> astronomix code-unit fields on a ``num_cells^3`` box.

    Conserved exactly: mass, momentum, internal energy, and the mass of every
    passive scalar (they are remapped as ``rho * C``). The kinetic energy of
    velocity structure below the target cell is not representable and is
    dropped, NOT converted into heat -- the cold ejecta carry ~1e-6 of their
    energy as heat, so thermalising the sub-cell velocity shear would heat them
    by orders of magnitude.

    ``similarity`` = (L, T, M): the exact similarity map of the adiabatic
    problem (``casa_rescale``) applied AT THE SOURCE: PLUTO's cell edges are
    stretched by L before the conservative remap, every remapped density gets
    its factor (rho x M / L^3, momentum x M / L^3 x L / T, e_int x M / L^3 (L /
    T)^2, shock time x T, B x sqrt(M / L^3) L / T), and ``fit`` must be the
    SCALED ambient (``casa_rescale.scale_fit``), which fills the rest of the box.
    The caller labels the result with age = T x 145.5 yr.
    """
    Ls, Ts, Ms = similarity if similarity is not None else (1.0, 1.0, 1.0)
    f_rho, f_v = Ms / Ls ** 3, Ls / Ts
    FAC = dict(mass=f_rho, mom0=f_rho * f_v, mom1=f_rho * f_v, mom2=f_rho * f_v,
               eint=f_rho * f_v ** 2, m_ej=f_rho, m_ej_untagged=f_rho, m_fsh=f_rho,
               m_tss=f_rho * Ts, m_dt=f_rho ** 2 * Ts,
               **{f"m_{sp}": f_rho for sp in PIPELINE_GROUPS},
               **{f"B{k}": f_rho ** 0.5 * f_v for k in (1, 2, 3)})
    cu = snr_code_units()
    rho_c = (1.0 * cu.code_density).to(u.g / u.cm ** 3).value
    v_c = (1.0 * cu.code_velocity).to(u.cm / u.s).value
    p_c = (1.0 * cu.code_pressure).to(u.erg / u.cm ** 3).value
    t_c = (1.0 * cu.code_time).to(u.s).value

    dst_edges = np.linspace(-0.5 * box_pc, 0.5 * box_pc, num_cells + 1)
    W = [overlap_matrix(e / PC * Ls, dst_edges) for e in snap.edges_cm]
    # only destination rows that touch the PLUTO box
    rows = [np.nonzero(w.sum(1) > 0)[0] for w in W]
    Wc = [w[r] for w, r in zip(W, rows)]
    cover = np.einsum("a,b,c->abc", *[w.sum(1) for w in Wc])
    sl = tuple(slice(r[0], r[-1] + 1) for r in rows)
    log(f"[pluto] remap {snap.n}^3 (dx = {np.diff(snap.edges_cm[0])[0] / PC:.5f} pc) -> "
        f"{num_cells}^3 in {box_pc:.2f} pc (dx = {box_pc / num_cells:.5f} pc); "
        f"PLUTO box covers {len(rows[0])}^3 destination cells")

    def remap(q):
        return _apply_separable(np.asarray(q, dtype=np.float64), Wc)

    # ---- source-side derived fields (cgs) ----
    rho_s = np.asarray(snap.field("rho"), dtype=np.float64) * snap.unit_density
    out = {}
    out["mass"] = remap(rho_s)
    for k, name in enumerate(("vx1", "vx2", "vx3")):
        out[f"mom{k}"] = remap(rho_s * (np.asarray(snap.field(name), dtype=np.float64)
                                        * snap.unit_velocity))
    prs_s = np.asarray(snap.field("prs"), dtype=np.float64) * snap.unit_pressure
    out["eint"] = remap(prs_s / (gamma - 1.0))

    # kinetic energy bookkeeping: what the remap cannot represent
    ke_src = 0.5 * sum(rho_s * (np.asarray(snap.field(n), dtype=np.float64)
                                * snap.unit_velocity) ** 2 for n in ("vx1", "vx2", "vx3"))
    ke_fine = float(ke_src.sum()) * np.prod([np.diff(e)[0] for e in snap.edges_cm])
    del ke_src

    ej = np.asarray(snap.field("tr4"), dtype=np.float64)
    out["m_ej"] = remap(rho_s * ej)
    elem = {t: np.asarray(snap.field(t), dtype=np.float64) for t in ELEMENT_TRACERS}
    s_sum = sum(elem.values())
    # normalise the element fractions to the ejecta fraction (they sum to 1 in
    # pure ejecta only to within 0.64-1.33 per cell); CSM gets cosmic abundances
    norm = np.where(s_sum > 1e-6, ej / np.maximum(s_sum, 1e-30), 0.0)
    for sp, members in PIPELINE_GROUPS.items():
        frac = sum(elem[t] for t in members) * norm
        # ejecta cells with no element record at all take the IIb mean below
        out[f"m_{sp}"] = remap(rho_s * (frac + (1.0 - ej) * csm.get(sp, 0.0)))
    out["m_ej_untagged"] = remap(rho_s * np.where(s_sum > 1e-6, 0.0, ej))
    del elem, s_sum, norm

    # shock history: Orlando's shock time -> the library's accumulators
    t_sh = np.asarray(snap.field("tr1"), dtype=np.float64)
    shocked = (t_sh * snap.unit_time > shock_time_min_yr * YR).astype(np.float64)
    dt_s = shocked * np.clip(snap.time_code - t_sh, 0.0, None) * snap.unit_time   # s
    out["m_fsh"] = remap(rho_s * shocked)
    out["m_tss"] = remap(rho_s * dt_s)
    # rho * Delta t at the present density, as prior_shock_history does
    out["m_dt"] = remap(rho_s * rho_s * dt_s)
    del t_sh, shocked, dt_s

    # magnetic field: volume average (not used by the hydro, kept for synchrotron)
    for k in (1, 2, 3):
        out[f"B{k}"] = remap(np.asarray(snap.field(f"Bx{k}"), dtype=np.float64) * snap.unit_b)

    # ---- destination grid, analytic fill of the uncovered fraction ----
    xc = 0.5 * (dst_edges[:-1] + dst_edges[1:])
    X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    rho_amb = ambient_density(r, X, Y, Z, fit, dx=box_pc / num_cells)
    p_amb = ambient_pressure(r, fit, rho=rho_amb, mu=snap.mu)

    def fill(key, amb):
        full = amb.copy() if np.ndim(amb) else np.full(r.shape, float(amb))
        full[sl] = FAC.get(key, 1.0) * out[key] + (1.0 - cover) * full[sl]
        return full

    mass = fill("mass", rho_amb)
    fields = dict(rho=mass / rho_c)
    for k, name in enumerate(("vx", "vy", "vz")):
        fields[name] = fill(f"mom{k}", 0.0) / mass / v_c
    fields["press"] = fill("eint", p_amb / (gamma - 1.0)) * (gamma - 1.0) / p_c
    fields["C_ej"] = np.clip(fill("m_ej", 0.0) / mass, 0.0, 1.0)
    for sp in PIPELINE_GROUPS:
        fields[f"C_{sp}"] = np.clip(fill(f"m_{sp}", rho_amb * csm.get(sp, 0.0)) / mass, 0.0, 1.0)
    fields["shocked_fraction"] = np.clip(fill("m_fsh", 0.0) / mass, 0.0, 1.0)
    fields["time_since_shock"] = fill("m_tss", 0.0) / mass / t_c
    fields["density_time"] = fill("m_dt", 0.0) / mass / rho_c / t_c
    for k, name in zip((1, 2, 3), ("bx", "by", "bz")):
        fields[name] = fill(f"B{k}", 0.0)          # Gauss

    # ---- conservation report ----
    dv_dst = (box_pc / num_cells * PC) ** 3
    ke_dst = float(np.sum(0.5 * mass * sum((fields[n] * v_c) ** 2 for n in ("vx", "vy", "vz")))
                   * dv_dst)
    ke_dst_in = float(np.sum((0.5 * mass * sum((fields[n] * v_c) ** 2
                                                for n in ("vx", "vy", "vz")))[sl]) * dv_dst)
    # source totals x the map's factors (M for masses, M (L / T)^2 for energies)
    m_src = float(rho_s.sum()) * np.prod([np.diff(e)[0] for e in snap.edges_cm]) / MSUN * Ms
    ke_fine = ke_fine * Ms * f_v ** 2
    m_dst_in = float(out["mass"].sum()) * f_rho * dv_dst / MSUN
    log(f"[pluto] mass in PLUTO box {m_src:.4f} Msun -> {m_dst_in:.4f} on the new grid "
        f"({100 * (m_dst_in / m_src - 1):+.2e} %); whole new box {mass.sum() * dv_dst / MSUN:.3f} Msun")
    log(f"[pluto] kinetic energy {ke_fine:.4e} -> {ke_dst_in:.4e} erg "
        f"({100 * (ke_dst_in / ke_fine - 1):+.3f} %, sub-cell shear dropped); "
        f"whole box {ke_dst:.4e} erg")
    untag = float(out["m_ej_untagged"].sum()) * f_rho * dv_dst / MSUN
    log(f"[pluto] ejecta {float(np.sum(fields['C_ej'] * mass)) * dv_dst / MSUN:.4f} Msun, "
        f"of which {untag:.4f} carry no element record (-> hydrogen)")
    return fields, dict(cover_slices=sl, code_units=cu)
# =============================================================================
# ============ ↑ Conservative remap onto an astronomix grid ↑ =================
# =============================================================================


# =============================================================================
# ============ ↓ Diagnostics on the native snapshot ↓ =========================
# =============================================================================
def shock_radii_by_direction(snap, stride=2, n_dirs=400, seed=0):
    """r_FS (outermost shocked gas), r_CD (outermost ejecta) and r_RS (innermost
    shocked ejecta) along random rays, from the tracers rather than a contrast."""
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_dirs, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    xc = snap.centers_pc(stride)[0]
    dx = xc[1] - xc[0]
    t_sh = np.asarray(snap.field("tr1", stride), dtype=np.float32)
    ej = np.asarray(snap.field("tr4", stride), dtype=np.float32)
    rr = np.arange(0.02, xc[-1], 0.25 * dx)
    out = np.full((n_dirs, 3), np.nan)
    for i, d in enumerate(v):
        pts = (d[:, None] * rr[None, :] - xc[0]) / dx
        ts = map_coordinates(t_sh, pts, order=1, mode="nearest")
        e = map_coordinates(ej, pts, order=1, mode="nearest")
        shocked = ts > 1e-3
        if shocked.any():
            out[i, 0] = rr[np.nonzero(shocked)[0][-1]]
        ej_idx = np.nonzero(e > 0.5)[0]
        if ej_idx.size:
            out[i, 1] = rr[ej_idx[-1]]
        sh_ej = np.nonzero(shocked & (e > 0.5))[0]
        if sh_ej.size:
            out[i, 2] = rr[sh_ej[0]]
    return v, out


def cmd_analyze(args):
    snap = PlutoSnapshot(args.dir)
    s = args.stride
    print(f"[pluto] {snap.dir}: {snap.n}^3, t = {snap.time_code:.4f} code = "
          f"{snap.age_yr:.1f} yr; box +-{snap.edges_cm[0][-1] / PC:.4f} pc; "
          f"variables {' '.join(snap.variables)}")
    print(f"[pluto] units: L = {snap.unit_length:.3e} cm, rho = {snap.unit_density:.4e} g/cc, "
          f"v = {snap.unit_velocity:.1e} cm/s, t = {snap.unit_time / YR:.2f} yr, "
          f"B = {snap.unit_b:.3f} G; mu = {snap.mu}")

    fit = fit_ambient(snap, stride=2)
    print("[pluto] ambient:\n  " + format_ambient(fit, snap.mu).replace("\n", "\n  "))

    xc, yc, zc = snap.centers_pc(s)
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    dv = (np.diff(snap.edges_cm[0])[0] * s) ** 3
    rho = np.asarray(snap.field("rho", s), dtype=np.float64) * snap.unit_density
    prs = np.asarray(snap.field("prs", s), dtype=np.float64) * snap.unit_pressure
    v = [np.asarray(snap.field(n, s), dtype=np.float64) * snap.unit_velocity
         for n in ("vx1", "vx2", "vx3")]
    ej = np.asarray(snap.field("tr4", s), dtype=np.float64)
    t_sh = np.asarray(snap.field("tr1", s), dtype=np.float64)
    shocked = t_sh * snap.unit_time > YR
    T = snap.mu * M_P * prs / (rho * const.k_B.cgs.value)
    m = rho * dv / MSUN
    ke = 0.5 * rho * sum(c ** 2 for c in v) * dv
    th = prs / (GAMMA - 1.0) * dv
    print(f"[pluto] mass {m.sum():.3f} Msun (ejecta {np.sum(m * ej):.3f}, shocked ejecta "
          f"{np.sum(m * ej * shocked):.3f}, shocked CSM {np.sum(m * (1 - ej) * shocked):.3f})")
    print(f"[pluto] energy: kinetic {ke.sum():.3e}, thermal {th.sum():.3e}, total "
          f"{ke.sum() + th.sum():.3e} erg (ejecta KE {np.sum(ke * ej):.3e})")
    print(f"[pluto] max |v| {np.sqrt(sum(c ** 2 for c in v)).max() / 1e5:.0f} km/s; "
          f"shocked gas: EM-weighted T = "
          f"{np.sum(T * rho ** 2 * shocked) / np.sum(rho ** 2 * shocked):.3e} K (mu = {snap.mu})")

    print("[pluto] element tracers (ejecta-weighted mass, shocked fraction, "
          "mass-weighted <r>):")
    for t in ELEMENT_TRACERS:
        f = np.asarray(snap.field(t, s), dtype=np.float64)
        mt = m * f * ej
        print(f"    {t:5s} {TRACER_KEY[t]:9s} {mt.sum():.4f} Msun, shocked "
              f"{np.sum(mt * shocked) / mt.sum():.2f}, <r> {np.sum(mt * r) / mt.sum():.3f} pc")
    for sp, members in PIPELINE_GROUPS.items():
        tot = sum(np.asarray(snap.field(t, s), dtype=np.float64) for t in members)
        print(f"    pipeline C_{sp:2s} = {'+'.join(members):18s} {np.sum(m * tot * ej):.4f} Msun "
              f"({np.sum(m * tot * ej * shocked):.4f} shocked)")

    dirs, radii = shock_radii_by_direction(snap, stride=2)
    for k, name in enumerate(("r_FS", "r_CD", "r_RS")):
        q = radii[:, k]
        print(f"[pluto] {name}: mean {np.nanmean(q):.3f}, min {np.nanmin(q):.3f}, "
              f"max {np.nanmax(q):.3f}, std {np.nanstd(q):.3f} pc")
    # plane of the sky is (x, z) with Earth on -y (Orlando's convention)
    pa = np.rad2deg(np.arctan2(dirs[:, 2], dirs[:, 0])) % 360
    sky = np.abs(dirs[:, 1]) < 0.3
    print(f"[pluto] r_FS in the plane of the sky ({sky.sum()} rays): "
          f"{np.nanmean(radii[sky, 0]):.3f} +- {np.nanstd(radii[sky, 0]):.3f} pc")

    b = [np.asarray(snap.field(f"Bx{k}", s), dtype=np.float64) * snap.unit_b for k in (1, 2, 3)]
    B = np.sqrt(sum(c ** 2 for c in b)) * 1e6
    for name, sel in (("unshocked wind", (~shocked) & (ej < 0.01) & (r > 1.2)),
                      ("shocked CSM", shocked & (ej < 0.1)),
                      ("shocked ejecta", shocked & (ej > 0.9))):
        beta = prs[sel] / ((B[sel] * 1e-6) ** 2 / (8 * np.pi))
        print(f"[pluto] B in {name}: median {np.median(B[sel]):.1f} uG, 90% "
              f"{np.quantile(B[sel], 0.9):.0f} uG; plasma beta median {np.median(beta):.3g}")

    if args.figure:
        _analysis_figure(snap, fit, radii, dirs, pa, args.figure)


def _analysis_figure(snap, fit, radii, dirs, pa, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = 1
    k = snap.n // 2
    ext = [snap.edges_cm[0][0] / PC, snap.edges_cm[0][-1] / PC] * 2

    def sl(name):
        # the y = 0 plane is the plane of the sky (x, z)
        return np.asarray(snap.field(name, s)[:, k, :], dtype=np.float64).T

    rho = sl("rho") * snap.unit_density / (snap.mu * M_P)
    T = snap.mu * M_P * sl("prs") * snap.unit_pressure / (sl("rho") * snap.unit_density
                                                          * const.k_B.cgs.value)
    panels = [("n [cm$^{-3}$]", np.log10(rho), "viridis"),
              ("T [K]", np.log10(np.maximum(T, 1e3)), "inferno"),
              ("ejecta fraction", sl("tr4"), "magma"),
              ("Fe (tr19)", sl("tr19"), "cividis"),
              ("O (tr16)", sl("tr16"), "cividis"),
              ("Si (tr15)", sl("tr15"), "cividis"),
              ("shock age [yr]", np.where(sl("tr1") > 1e-3,
                                          (snap.time_code - sl("tr1")) * snap.unit_time / YR,
                                          np.nan), "plasma"),
              ("|B| [$\\mu$G]", np.log10(1e6 * snap.unit_b * np.sqrt(
                  sl("Bx1") ** 2 + sl("Bx2") ** 2 + sl("Bx3") ** 2) + 1e-3), "magma")]
    fig, axs = plt.subplots(3, 4, figsize=(18, 13.5), layout="constrained")
    for ax, (title, img, cmap) in zip(axs.flat, panels):
        im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap)
        ax.set_title(("log " if title.startswith(("n", "T", "|B")) else "") + title)
        ax.set_xlabel("x [pc]"); ax.set_ylabel("z [pc]")
        plt.colorbar(im, ax=ax, fraction=0.046)
    ax = axs.flat[8]
    for kk, name in enumerate(("r_FS", "r_CD", "r_RS")):
        ax.hist(radii[:, kk], bins=40, histtype="step", label=name)
    ax.axvline(fit["r_sh"], color="k", ls=":", label="shell r_sh")
    ax.set_xlabel("radius along 400 random rays [pc]"); ax.legend()
    ax = axs.flat[9]
    sky = np.abs(dirs[:, 1]) < 0.3
    ax.scatter(pa[sky], radii[sky, 0], s=8, label="r_FS")
    ax.scatter(pa[sky], radii[sky, 2], s=8, label="r_RS")
    ax.set_xlabel("angle in the (x, z) plane [deg]"); ax.set_ylabel("pc"); ax.legend()
    ax.set_title("shock radii near the plane of the sky")
    ax = axs.flat[10]
    th = np.linspace(0, 2 * np.pi, 361)
    for ph_deg in (0.0,):
        Xs, Zs = fit["r_sh"] * np.cos(th), fit["r_sh"] * np.sin(th)
        dens = ambient_density(fit["r_sh"], Xs, 0 * Xs, Zs, fit) / (snap.mu * M_P)
        ax.plot(np.rad2deg(th), dens)
    ax.set_yscale("log"); ax.set_xlabel("angle in the (x, z) plane [deg]")
    ax.set_ylabel("n at r_sh [cm$^{-3}$]"); ax.set_title("fitted shell, plane of the sky")
    axs.flat[11].axis("off")
    axs.flat[11].text(0, 0.5, format_ambient(fit, snap.mu).replace("; ", "\n").replace(", ", "\n"),
                      fontsize=9, family="monospace", va="center")
    fig.suptitle(f"Orlando's Cas A state at {snap.age_yr:.1f} yr (PLUTO, y = 0 slice)")
    fig.savefig(out_path, dpi=110)
    print(f"[pluto] wrote {out_path}")
# =============================================================================
# ============ ↑ Diagnostics on the native snapshot ↑ =========================
# =============================================================================


def solar_csm_composition():
    """Solar (Anders & Grevesse) mass fractions per PIPELINE scalar.

    Derived from ``_plasma.SOLAR_NUMBER_RATIO_TO_H`` -- the table the X-ray
    model normalises abundances to -- so "solar" here and there cannot drift:
    the "O" scalar stands for O+Ne+Mg and "Si" for Si+S+Ar+Ca, as in
    ``_plasma.TRACER_SPLIT``.
    """
    from _plasma import ATOMIC, SOLAR_NUMBER_RATIO_TO_H as R
    w = {"H": ATOMIC["H"][0]}
    w.update({el: R[el] * ATOMIC[el][0] for el in R})
    tot = sum(w.values())
    X = {el: v / tot for el, v in w.items()}
    return {"He": X["He"], "O": X["O"] + X["Ne"] + X["Mg"],
            "Si": X["Si"] + X["S"] + X["Ar"] + X["Ca"], "Fe": X["Fe"]}


def recompose_csm(fields, old=CSM_COMPOSITION, new=None):
    """Swap the circumstellar part of every composition scalar, exactly.

    The scalars are advected linearly, so every cell holds
    ``C_ej * X_ej + (1 - C_ej) * X_csm`` whatever mixing happened; the CSM part
    can therefore be replaced after the run with no hydro re-run. Needed
    because ``_common.CSM_COMPOSITION`` is not solar (Si-group 4x, Fe 2.7x),
    and the shocked shell carries ~60 % of the thermal emission measure.
    """
    new = new or solar_csm_composition()
    out = dict(fields)
    f_csm = np.clip(1.0 - np.asarray(fields["C_ej"], dtype=np.float64), 0.0, 1.0)
    for sp in PIPELINE_GROUPS:
        k = f"C_{sp}"
        out[k] = np.clip(np.asarray(fields[k], np.float64)
                         + f_csm * (new[sp] - old.get(sp, 0.0)), 0.0, 1.0).astype(np.float32)
    return out


def cmd_recompose(args):
    new = solar_csm_composition()
    print("[pluto] CSM composition by mass: " + ", ".join(
        f"{sp} {CSM_COMPOSITION.get(sp, 0):.4g} -> {new[sp]:.4g}" for sp in PIPELINE_GROUPS))
    for path in args.states:
        d = dict(np.load(path))
        d = recompose_csm(d, new=new)
        d["csm_composition"] = np.array(repr(new))
        out = path[:-4] + "_solarcsm.npz"
        np.savez_compressed(out, **d)
        print(f"[pluto] wrote {out}")


def cmd_convert(args):
    snap = PlutoSnapshot(args.dir)
    fit = fit_ambient(snap, stride=2)
    print("[pluto] ambient:\n  " + format_ambient(fit, snap.mu).replace("\n", "\n  "))
    sim = tuple(args.sim) if args.sim is not None else None
    age_yr = float(snap.age_yr)
    if sim is not None:
        # the exact similarity map at the source (casa_rescale): scaled ambient,
        # stretched PLUTO cells, and the age label T x 145.5 yr
        from casa_rescale import scale_fit, factors
        L_, T_, M_ = sim
        fit = scale_fit(fit, L_, T_, M_)
        age_yr *= T_
        F = factors(L_, T_, M_)
        print(f"[pluto] similarity L {L_:.4f} T {T_:.4f} M {M_:.4f}: E x {F['energy']:.4f}, masses x {M_:.4f}, "
              f"age {snap.age_yr:.2f} -> {age_yr:.2f} yr; scaled ambient:\n  "
              + format_ambient(fit, snap.mu).replace("\n", "\n  "))
    fields, _ = remap_to_grid(snap, args.box, args.n, fit, similarity=sim)
    # the ambient reference casa_orlando's detectors use: n in ITS convention
    # (1.4 m_p per nucleus), with no constant ISM term
    from _common import MASS_PER_NUCLEUS
    n_w_ours = fit["rho_w"] / (MASS_PER_NUCLEUS * M_P)
    np.savez_compressed(
        args.out, **{k: v.astype(np.float32) for k, v in fields.items()},
        box=float(args.box), age=age_yr, age_target=age_yr,
        num_cells=int(args.n), map_age=age_yr,
        **({} if sim is None else dict(similarity_L=float(sim[0]), similarity_T=float(sim[1]),
                                       similarity_M=float(sim[2]),
                                       similarity_age_unscaled=float(snap.age_yr),
                                       similarity_method=np.array("conservative at the PLUTO source"))),
        argv=np.array("casa_pluto.py convert " + " ".join(
            f"--{k} {v}" for k, v in vars(args).items() if k not in ("func",))),
        git_commit=np.array(""), mass_conserved=True,
        wind_asym_dipole=0.0, wind_asym_quadrupole=0.0,
        wind_asym_theta_deg=0.0, wind_asym_phi_deg=0.0,
        n_w=float(n_w_ours), r_fs_ref=float(fit["r_ref"]), n_c=0.0,
        r_profile_max=np.nan, gamma=float(GAMMA),
        source=np.array(str(snap.dir)),
        tracer_key=np.array(repr(TRACER_KEY)), pipeline_groups=np.array(repr(PIPELINE_GROUPS)),
        **{f"ambient_{k}": v for k, v in fit.items()})
    print(f"[pluto] wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=str(PLUTO_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="diagnostics on the native snapshot")
    a.add_argument("--stride", type=int, default=2)
    a.add_argument("--figure", default=str(FIGURES_DIR / "pluto146_analysis.png"))
    a.set_defaults(func=cmd_analyze)
    c = sub.add_parser("convert", help="write an astronomix-grid npz")
    c.add_argument("--n", type=int, default=256)
    c.add_argument("--box", type=float, default=7.0)
    c.add_argument("--out", required=True)
    c.add_argument("--sim", type=float, nargs=3, default=None, metavar=("L", "T", "M"),
                   help="exact similarity rescaling at the source (casa_rescale): lengths x L, "
                        "times x T (age label T x 145.5 yr), masses x M")
    c.set_defaults(func=cmd_convert)
    rc = sub.add_parser("recompose", help="solar CSM composition in saved states")
    rc.add_argument("states", nargs="+")
    rc.set_defaults(func=cmd_recompose)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
