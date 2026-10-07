"""
Forward-shock displacement control of the Cas A 4D-Var (``casa_4dvar --warp``).

Why: about 50 of the outline chi2 is 20-30 deg forward-shock structure (Stage-3
physics section 3) that the smooth B^(1/2) state increments cannot move: a smooth
increment cannot create the shock jump ahead of the shock. This control moves
the outer remnant (the shocked-CSM shell, the forward shock and the CSM just
ahead of it) along the radial direction:

    x  ->  x + delta(n) f(|x| / R(n)) n,        n = x / |x|,

    delta(n) = sigma_pc * sqrt(4 pi) / (L + 1) * sum_{l <= L, m} xi_lm Y_lm(n)

(real orthonormal Y_lm, (L + 1)^2 = 81 coefficients for L = 8; white xi gives a
pointwise rms of delta equal to sigma_pc, and the prior is 0.5 |xi|^2).

* ``R(n)``: the background's forward-shock radius, the outermost radius where the
  hot indicator (T > 1e7 K, ``casa_pluto_diff.shocked_indicator``) crosses 1/2
  along ~3000 rays, least-squares fitted by real Y_lm up to ``r_lfit`` (fixed:
  it is a property of the background, not of the control).
* ``f(s)``, s = r / R(n): the radial taper, 0 for s < s0, a C^1 smoothstep up to
  1 at s1, 1 on [s1, s2] (the shell and the shock jump), a smoothstep back to 0
  at s3 (the CSM ahead). Default (s0, s1, s2, s3) = (0.78, 0.97, 1.05, 1.35).
* No folding: along each ray r' = r + delta f(r / R) must be monotone, i.e.
  1 + delta f'(s) / R > 0. The raw delta is passed through a smooth saturation
  delta / (1 + (delta / L)^4)^(1/4) with L = ``fold_frac`` x the fold limit of
  its sign (inward: the ramp, R w_up / 1.5; outward: the decay, R w_down / 1.5),
  which is the identity to 1.5 % below L / 2.

Resampling (the warp is a GATHER): the new state at an output cell y (|y| = r')
is the old state at the source radius r_src on the same ray, r_src + delta
f(r_src / R) = r' (5 Newton steps), taken by trilinear interpolation of EVERY
field (hydro, composition, histories: all intensive, so the shocked shell keeps
its density and pressure while it is stretched or compressed). Density and
pressure (``log_fields``) are interpolated in log space: across the forward
shock p jumps by ~1e3 and rho by ~4, so a linear p interpolant is dominated by
the hot side, T = p / rho stays above 1e7 K over ~97 % of the cell next to the
shock and any sub-cell displacement pushes the model's hot edge (the outline
estimator) outward by up to a cell (measured at 64^3: +3" bias for either sign
of delta). In log space the threshold crossing sits mid-cell. Only the cells
with s0 < s < s3 are gathered; elsewhere the state is untouched, and at xi = 0
the map is the identity bit for bit (source = cell centre exactly, weights 1/0;
the log fields as x_c + x_c expm1(interp(ln x) - ln x_c), whose argument is then exactly 0).

Interpolation kernel (``kernel``):

* ``"cubic"`` (default): separable Catmull-Rom (Keys a = -1/2) cubic
  convolution, 64 taps. It interpolates (identity at the cell centres) and is
  C^1 in the sample position, with derivative = the central difference at a
  centre, so plain AD is the EXACT derivative of the map everywhere and J has
  no kink at xi = 0. It overshoots a step by up to ~7 %: rho, p, e are
  interpolated in log space (no sign change; the overshoot is a factor), and
  the bounded fields (mass fractions, shocked fraction, histories >= 0) are
  clipped to their ranges afterwards (``bounds``).
* ``"linear"``: trilinear (monotone, 8 taps) with a custom JVP. Its exact
  position derivative is piecewise constant and switches at the cell centres,
  so at xi = 0 (every sample exactly at a centre) it is a one-sided difference
  in every cell at once: a coherent kink of J at the start point. The tangent
  uses instead the central-difference gradient of each field, trilinearly
  interpolated to the sample position (exact at xi = 0; an approximation at
  generic xi: on the CPU test a white-noise-weighted functional of the warped
  state disagrees with FD at O(1), the cone r_FS by ~30 %).

Unshocked CSM ahead of the shock (``upstream=``, casa_4dvar default
``--warp-upstream keep``): the gathered unshocked gas is rescaled by the
background's local radial power law so the wind profile stays where it was
while the shock and shell move (``Warp._setup_upstream``).

The dual-energy internal energy and the entropy label are not special-cased
here: the caller (``casa_4dvar_control.Control.state_on``) rebuilds g = p /
(gamma - 1) and s0 from the warped background, so they stay consistent. The
remnant mask of the B^(1/2) increment and the never-shocked-CSM weight can be
warped with the state (``Warp.apply(..., extra=)``), so the controlled region
moves with the shock.
"""
import math

import numpy as np
import jax
import jax.numpy as jnp

ARCSEC_PC_PER_KPC = 1e3 * np.pi / (180.0 * 3600.0)       # pc per arcsec per kpc
DEFAULT_TAPER = (0.78, 0.97, 1.05, 1.35)


# =============================================================================
# ============ ↓ Real spherical harmonics ↓ ===================================
# =============================================================================
def sh_names(lmax):
    return [f"w{l}_{m}" for l in range(lmax + 1) for m in range(-l, l + 1)]


def _sh_norm(l, m):
    return math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m))


def real_sh(lmax, nx, ny, nz):
    """Real orthonormal spherical harmonics Y_lm, l <= lmax, m = -l..l (in that
    order), of UNIT vectors (nx, ny, nz) (numpy or jax arrays): Cartesian
    recursions only (no angles, smooth at the poles). Y_l,m>0 ~ cos(m phi),
    Y_l,m<0 ~ sin(|m| phi); no Condon-Shortley phase."""
    # C_m + i S_m = (nx + i ny)^m = sin^m(theta) e^{i m phi}
    C, S = [nx * 0 + 1], [nx * 0]
    for m in range(lmax):
        C.append(nx * C[m] - ny * S[m])
        S.append(nx * S[m] + ny * C[m])
    # Q_l^m(t): P_l^m(t) = sin^m(theta) Q_l^m(t)
    Q = {}
    for m in range(lmax + 1):
        Q[(m, m)] = nz * 0 + float(np.prod(np.arange(1, 2 * m, 2))) if m else nz * 0 + 1.0
        if m + 1 <= lmax:
            Q[(m + 1, m)] = (2 * m + 1) * nz * Q[(m, m)]
        for l in range(m + 2, lmax + 1):
            Q[(l, m)] = ((2 * l - 1) * nz * Q[(l - 1, m)] - (l + m - 1) * Q[(l - 2, m)]) / (l - m)
    out = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            c = _sh_norm(l, am) * (1.0 if m == 0 else math.sqrt(2.0))
            out.append(c * Q[(l, am)] * (1.0 if m == 0 else (C[am] if m > 0 else S[am])))
    return out


def sh_eval(coef, lmax, nx, ny, nz):
    """sum_k coef_k Y_k(n) without materialising the basis."""
    ys = real_sh(lmax, nx, ny, nz)
    acc = coef[0] * ys[0]
    for k in range(1, len(ys)):
        acc = acc + coef[k] * ys[k]
    return acc


def fibonacci_sphere(n_dirs):
    k = np.arange(n_dirs) + 0.5
    th = np.arccos(1.0 - 2.0 * k / n_dirs)
    ph = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])       # (3, N)
# =============================================================================
# ============ ↑ Real spherical harmonics ↑ ===================================
# =============================================================================


# =============================================================================
# ============ ↓ Forward-shock radius of the background ↓ =====================
# =============================================================================
def ray_outer_radius(field, box, n, dirs, *, thr=0.5, r_min=0.3, r_max=None, step_frac=0.25):
    """Outermost radius (code length) along each ray (``dirs`` (3, N) unit
    vectors from the box centre) where ``field`` (n^3, host) crosses ``thr``
    (trilinear samples every step_frac cells, linear interpolation of the
    crossing); NaN if it never exceeds thr."""
    from scipy.ndimage import map_coordinates
    dx = box / n
    r_max = r_max or 0.5 * box - dx
    rr = np.arange(r_min, r_max, step_frac * dx)
    pts = dirs[:, :, None] * rr[None, None, :]
    idx = (pts + 0.5 * box) / dx - 0.5
    prof = map_coordinates(np.asarray(field, np.float64), idx.reshape(3, -1), order=1,
                           mode="nearest").reshape(dirs.shape[1], rr.size)
    above = prof > thr
    out = np.full(dirs.shape[1], np.nan)
    for i in range(dirs.shape[1]):
        j = np.nonzero(above[i])[0]
        if j.size == 0:
            continue
        j = j.max()
        if j + 1 < rr.size:
            f0, f1 = prof[i, j], prof[i, j + 1]
            out[i] = rr[j] + (f0 - thr) / max(f0 - f1, 1e-30) * (rr[j + 1] - rr[j])
        else:
            out[i] = rr[j]
    return out


def fit_fs_radius(hot, box, n, *, lfit=8, n_dirs=3000, thr=0.5, ridge=1e-8):
    """(Y_lm coefficients of R(n) up to ``lfit``, diagnostics) from the hot
    indicator of the background (host arrays)."""
    dirs = fibonacci_sphere(n_dirs)
    R = ray_outer_radius(hot, box, n, dirs, thr=thr)
    ok = np.isfinite(R)
    A = np.stack(real_sh(lfit, *dirs), 1)[ok]
    c = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ R[ok])
    res = R[ok] - A @ c
    diag = dict(n_dirs=int(n_dirs), n_ok=int(ok.sum()), R_mean=float(np.mean(R[ok])),
                R_min=float(np.min(R[ok])), R_max=float(np.max(R[ok])), fit_rms=float(np.std(res)),
                fit_maxabs=float(np.max(np.abs(res))), lfit=int(lfit))
    return c, diag
# =============================================================================
# ============ ↑ Forward-shock radius of the background ↑ =====================
# =============================================================================


# =============================================================================
# ============ ↓ Trilinear gather with a smooth position tangent ↓ ============
# =============================================================================
_CORNERS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]


def _setup(pos, n):
    """Clipped positions, lower corner indices (int32, <= n - 2) and fractions;
    at pos == n - 1 the fraction is exactly 1 (so a sample at any cell centre is
    exact); ``inside``: the position was not clipped."""
    p = jnp.clip(pos, 0.0, n - 1.0)
    i0 = jnp.clip(jnp.floor(p), 0, n - 2)
    fr = p - i0
    inside = (pos >= 0.0) & (pos <= n - 1.0)
    return i0.astype(jnp.int32), fr, inside


def _flat(i, j, k, n):
    return (i * n + j) * n + k


def _trilinear_raw(xf, pos, n):
    """xf (nv, n^3), pos (3, M) float index coordinates -> (nv, M)."""
    i0, fr, _ = _setup(pos, n)
    out = None
    for (a, b, c) in _CORNERS:
        w = (fr[0] if a else 1 - fr[0]) * (fr[1] if b else 1 - fr[1]) * (fr[2] if c else 1 - fr[2])
        v = w * xf[:, _flat(i0[0] + a, i0[1] + b, i0[2] + c, n)]
        out = v if out is None else out + v
    return out


def _trilinear_grad(xf, pos, n):
    """(nv, 3, M): the central-difference index-space gradient of each field,
    trilinearly interpolated to pos (zero where pos was clipped)."""
    i0, fr, inside = _setup(pos, n)
    grads = []
    for ax in range(3):
        acc = None
        for (a, b, c) in _CORNERS:
            w = (fr[0] if a else 1 - fr[0]) * (fr[1] if b else 1 - fr[1]) * (fr[2] if c else 1 - fr[2])
            ijk = [i0[0] + a, i0[1] + b, i0[2] + c]
            up = list(ijk)
            dn = list(ijk)
            up[ax] = jnp.clip(ijk[ax] + 1, 0, n - 1)
            dn[ax] = jnp.clip(ijk[ax] - 1, 0, n - 1)
            span = (up[ax] - dn[ax]).astype(xf.dtype)
            g = (xf[:, _flat(*up, n)] - xf[:, _flat(*dn, n)]) / jnp.maximum(span, 1.0)
            v = w * g
            acc = v if acc is None else acc + v
        grads.append(jnp.where(inside[ax], acc, 0.0))
    return jnp.stack(grads, 1)


def _is_zero(t):
    SZ = getattr(jax.custom_derivatives, "SymbolicZero", None)
    return SZ is not None and isinstance(t, SZ)


def trilinear(xf, pos, n):
    """Trilinear gather of xf (nv, n^3) at index positions pos (3, M); tangent
    with respect to pos = the interpolated central-difference gradient (see the
    module docstring). ``n`` static."""
    return _TRILINEAR(xf, pos, n)


def _build():
    f = jax.custom_jvp(lambda xf, pos, n: _trilinear_raw(xf, pos, n), nondiff_argnums=(2,))

    def jvp(n, primals, tangents):
        xf, pos = primals
        xd, pd = tangents
        out = _trilinear_raw(xf, pos, n)
        tan = None
        if not _is_zero(xd):
            tan = _trilinear_raw(xd, pos, n)
        if not _is_zero(pd):
            tp = jnp.sum(_trilinear_grad(xf, pos, n) * pd[None], 1)
            tan = tp if tan is None else tan + tp
        if tan is None:
            tan = jnp.zeros_like(out)
        return out, tan
    try:
        f.defjvp(jvp, symbolic_zeros=True)
    except TypeError:                     # older jax: tangents instantiated as zeros
        f.defjvp(jvp)
    return f


_TRILINEAR = _build()


def _keys_weights(t):
    """Catmull-Rom (Keys a = -1/2) weights of the taps i0 - 1 .. i0 + 2 at
    fraction t in [0, 1]: exactly (0, 1, 0, 0) at t = 0."""
    return (t * (-0.5 + t * (1.0 - 0.5 * t)),
            1.0 + t * t * (-2.5 + 1.5 * t),
            t * (0.5 + t * (2.0 - 1.5 * t)),
            t * t * (-0.5 + 0.5 * t))


def tricubic(xf, pos, n):
    """Separable Catmull-Rom gather of xf (nv, n^3) at index positions pos (3,
    M) -> (nv, M); edge taps replicate the boundary cell; positions are clipped
    to [0, n - 1] (derivative 0 beyond). C^1 in pos: plain AD."""
    p = jnp.clip(pos, 0.0, n - 1.0)
    i0 = jnp.clip(jnp.floor(jax.lax.stop_gradient(p)), 0, n - 2)
    t = p - i0
    i0 = i0.astype(jnp.int32)
    W = [_keys_weights(t[a]) for a in range(3)]
    I = [[jnp.clip(i0[a] + o, 0, n - 1) for o in (-1, 0, 1, 2)] for a in range(3)]
    out = None
    for a in range(4):
        for b in range(4):
            wab = W[0][a] * W[1][b]
            iab = (I[0][a] * n + I[1][b]) * n
            for c in range(4):
                v = (wab * W[2][c]) * xf[:, iab + I[2][c]]
                out = v if out is None else out + v
    return out
# =============================================================================
# ============ ↑ Trilinear gather with a smooth position tangent ↑ ============
# =============================================================================


# =============================================================================
# ============ ↓ The warp ↓ ===================================================
# =============================================================================
def smoothstep(t):
    t = jnp.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def dsmoothstep(t):
    t = jnp.clip(t, 0.0, 1.0)
    return 6.0 * t * (1.0 - t)


class Warp:
    """The forward-shock displacement for one background state.

    ``hot``: the background's hot indicator (n^3, host); ``box``: box length
    (code length = pc); ``sigma_pc``: pointwise prior rms of delta; ``lmax``:
    angular cut-off of delta; ``taper``: (s0, s1, s2, s3); ``fold_frac``: the
    saturation level as a fraction of the fold limit; ``r_lfit``: angular
    cut-off of the fitted R(n); ``log_fields``: state indices interpolated in
    log space (casa_4dvar: rho, press, internal_energy); ``bounds``: {index:
    (lo, hi)} clipped after the interpolation (hi may be None); ``kernel``:
    "cubic" (Catmull-Rom, exact AD) or "linear" (trilinear, custom JVP)."""

    def __init__(self, hot, box, n, *, sigma_pc, lmax=8, taper=DEFAULT_TAPER, fold_frac=0.85, r_lfit=8,
                 dtype=jnp.float32, n_newton=5, log_fields=(), bounds=None, kernel="cubic", upstream=None):
        s0, s1, s2, s3 = (float(v) for v in taper)
        if not (0.0 < s0 < s1 <= s2 < s3):
            raise ValueError(f"taper must satisfy 0 < s0 < s1 <= s2 < s3, got {taper}")
        self.box, self.n, self.dx = float(box), int(n), float(box) / int(n)
        self.sigma_pc, self.lmax = float(sigma_pc), int(lmax)
        self.taper = (s0, s1, s2, s3)
        self.fold_frac = float(fold_frac)
        self.dtype = dtype
        self.n_newton = int(n_newton)
        self.n_coef = (self.lmax + 1) ** 2
        self.names = sh_names(self.lmax)
        self.c_norm = math.sqrt(4.0 * math.pi) / (self.lmax + 1)
        self.r_lfit = int(r_lfit)
        self.r_coef, self.r_diag = fit_fs_radius(hot, box, n, lfit=r_lfit)
        # cell geometry (host, float64)
        c = (np.arange(self.n) + 0.5) * self.dx - 0.5 * self.box
        Xg, Yg, Zg = np.meshgrid(c, c, c, indexing="ij")
        r = np.sqrt(Xg ** 2 + Yg ** 2 + Zg ** 2)
        rs = np.maximum(r, 1e-12)
        R = np.asarray(sh_eval(self.r_coef, self.r_lfit, Xg / rs, Yg / rs, Zg / rs))
        s = r / R
        act = np.nonzero(((s > s0) & (s < s3)).ravel())[0]
        self.act_np = act
        self.n_act = int(act.size)
        ii = np.stack(np.unravel_index(act, (self.n,) * 3)).astype(np.float64)       # (3, M)
        self._idx = ii
        self._nhat = np.stack([Xg.ravel()[act], Yg.ravel()[act], Zg.ravel()[act]]) / rs.ravel()[act]
        self._r = r.ravel()[act]
        self._R = R.ravel()[act]
        self.w_up, self.w_dn = s1 - s0, s3 - s2
        self.act = jnp.asarray(act, jnp.int32)
        self.idx = jnp.asarray(ii, dtype)
        self.nhat = jnp.asarray(self._nhat, dtype)
        self.r = jnp.asarray(self._r, dtype)
        self.R = jnp.asarray(self._R, dtype)
        self.R_cell_np = R                   # (n, n, n) fitted forward-shock radius (diagnostics)
        self.log_fields = tuple(log_fields)
        self.bounds = dict(bounds or {})
        if kernel not in ("cubic", "linear"):
            raise ValueError(f"kernel {kernel!r}")
        self.kernel = kernel
        self.upstream = None
        if upstream is not None:
            self._setup_upstream(**upstream)

    # ---- the unshocked CSM ahead of the shock -------------------------------------------
    def _setup_upstream(self, rho, press, t_per_code, i_rho, i_press, i_e=None, s_fit=(1.06, 1.30), lfit=4,
                        log_t_shock=7.0, width=0.1):
        """Keep the unshocked CSM's radial profile where it was: without this,
        the decay of the taper ahead of the shock resamples the wind from other
        radii (an outward push fills r with the denser gas from r - u: +2 u / r
        for a r^-2 wind, i.e. ~+6 % at 5\"), which then changes the shock's
        later speed. The gathered unshocked gas is rescaled by the background's
        local power law along its ray, rho -> rho (r / r_src)^b_rho(n), p ->
        p (r / r_src)^b_p(n) (b: least-squares d ln / d ln r of the background
        over s_fit along ~3000 rays, Y_lm-smoothed to l <= lfit), weighted by
        the unshocked indicator 1 - sigmoid((log10 T - 7) / 0.1) of the gathered
        gas. The shocked shell is untouched; at xi = 0 the factor is exactly 1."""
        from scipy.ndimage import map_coordinates
        dirs = fibonacci_sphere(3000)
        R = self.R_dirs(dirs)
        k = np.linspace(s_fit[0], s_fit[1], 25)
        rr = R[None, :] * k[:, None]                                             # (K, N)
        ok_r = rr < 0.5 * self.box - 1.5 * self.dx
        idx = (dirs[:, None, :] * rr[None] + 0.5 * self.box) / self.dx - 0.5
        slopes = {}
        for nm, f in (("rho", rho), ("press", press)):
            prof = np.log(np.maximum(map_coordinates(np.asarray(f, np.float64), idx.reshape(3, -1), order=1,
                                                     mode="nearest").reshape(rr.shape), 1e-300))
            b = np.full(dirs.shape[1], np.nan)
            for j in range(dirs.shape[1]):
                m = ok_r[:, j]
                if m.sum() >= 5:
                    b[j] = np.polyfit(np.log(rr[m, j]), prof[m, j], 1)[0]
            ok = np.isfinite(b)
            A = np.stack(real_sh(lfit, *dirs), 1)
            c = np.linalg.lstsq(A[ok], b[ok], rcond=None)[0]
            slopes[nm] = (c, float(np.nanmedian(b)), float(np.nanstd(b)))
        self._b_rho = np.asarray(sh_eval(slopes["rho"][0], lfit, *self._nhat))
        self._b_p = np.asarray(sh_eval(slopes["press"][0], lfit, *self._nhat))
        self.b_rho = jnp.asarray(self._b_rho, self.dtype)
        self.b_p = jnp.asarray(self._b_p, self.dtype)
        self.upstream = dict(t_per_code=float(t_per_code), i_rho=int(i_rho), i_press=int(i_press),
                             i_e=None if i_e is None else int(i_e), log_t_shock=float(log_t_shock),
                             width=float(width), s_fit=tuple(s_fit), lfit=int(lfit),
                             b_rho_median=slopes["rho"][1], b_rho_sd=slopes["rho"][2],
                             b_p_median=slopes["press"][1], b_p_sd=slopes["press"][2])

    # ---- the radial profile ---------------------------------------------------------
    def f(self, s):
        s0, s1, s2, s3 = self.taper
        return smoothstep((s - s0) / self.w_up) * (1.0 - smoothstep((s - s2) / self.w_dn))

    def fprime(self, s):
        s0, s1, s2, s3 = self.taper
        up, dn = smoothstep((s - s0) / self.w_up), smoothstep((s - s2) / self.w_dn)
        return dsmoothstep((s - s0) / self.w_up) / self.w_up * (1.0 - dn) \
            - up * dsmoothstep((s - s2) / self.w_dn) / self.w_dn

    def limits(self, R):
        """(inward, outward) saturation levels (code length) at forward-shock radius R."""
        return self.fold_frac * R * self.w_up / 1.5, self.fold_frac * R * self.w_dn / 1.5

    # ---- the angular displacement ------------------------------------------------------
    def delta_raw(self, xi, nx, ny, nz):
        a = jnp.asarray(self.sigma_pc * self.c_norm, xi.dtype) * xi
        return sh_eval(a, self.lmax, nx, ny, nz)

    def saturate(self, d, R):
        lin, lout = self.limits(R)
        L = jnp.where(d < 0, lin, lout)
        q = d / L
        return d / jnp.sqrt(jnp.sqrt(1.0 + (q * q) * (q * q)))

    def _geometry(self, xi):
        """(idx, nhat, r, R, xi) through an optimization barrier: the geometry
        is a jit constant, and without the barrier XLA constant-folds the 81 Y_lm
        (and the taper) over every active cell at compile time (~1 min on the
        CPU at 128^3)."""
        xi = jnp.asarray(xi, self.dtype)
        (idx, nhat, r, R), xi = jax.lax.optimization_barrier(((self.idx, self.nhat, self.r, self.R), xi))
        return idx, nhat, r, R, xi

    def upstream_slopes(self, xi):
        """(b_rho, b_p) at the active cells, through an optimization barrier."""
        (br, bp), _ = jax.lax.optimization_barrier(((self.b_rho, self.b_p), xi))
        return br, bp

    def delta(self, xi, geo=None):
        """Saturated delta (code length) at the active cells."""
        _, nhat, _, R, xi = geo or self._geometry(xi)
        return self.saturate(self.delta_raw(xi, *nhat), R)

    def delta_dirs(self, xi, dirs):
        """Saturated delta at arbitrary unit directions (3, N) (host numpy)."""
        dirs = np.asarray(dirs, np.float64)
        R = np.asarray(sh_eval(self.r_coef, self.r_lfit, *dirs))
        d = np.asarray(sh_eval(np.asarray(xi, np.float64) * self.sigma_pc * self.c_norm, self.lmax, *dirs))
        lin, lout = self.limits(R)
        L = np.where(d < 0, lin, lout)
        return d / (1.0 + (d / L) ** 4) ** 0.25

    def R_dirs(self, dirs):
        return np.asarray(sh_eval(self.r_coef, self.r_lfit, *np.asarray(dirs, np.float64)))

    # ---- the map ----------------------------------------------------------------------
    def source_shift(self, xi, geo=None):
        """u = r' - r_src (code length) at the active cells: r_src + delta f(r_src / R) = r'."""
        geo = geo or self._geometry(xi)
        d = self.delta(xi, geo)
        _, _, r, R, _ = geo
        rs = r - d * self.f(r / R)
        for _ in range(self.n_newton):
            g = rs + d * self.f(rs / R) - r
            rs = rs - g / (1.0 + d * self.fprime(rs / R) / R)
        return r - rs

    def positions(self, xi, with_shift=False):
        geo = self._geometry(xi)
        u = self.source_shift(xi, geo)
        pos = geo[0] - geo[1] * (u / self.dx)[None]
        return (pos, u, geo) if with_shift else pos

    def apply(self, x, xi, extra=None, log_fields=None, extra_bounds=(0.0, 1.0)):
        """The warped state (nv, n, n, n); ``extra`` (ne, n, n, n): further fields
        warped with it (e.g. the remnant mask), returned as a second output;
        ``log_fields``: indices of (positive) fields interpolated in log space
        (default: ``self.log_fields``, set by the caller, e.g. rho and p);
        ``extra_bounds``: the range the extra fields are clipped to."""
        nv = x.shape[0]
        full = x if extra is None else jnp.concatenate([x, extra.astype(x.dtype)], 0)
        # a constant background (the 4D-Var's x_b) would otherwise be log'd and gathered at compile time
        full, xi = jax.lax.optimization_barrier((full, jnp.asarray(xi, self.dtype)))
        xf = full.reshape(full.shape[0], -1)
        lf = tuple(self.log_fields if log_fields is None else log_fields)
        if lf:
            li = jnp.asarray(lf)
            tiny = jnp.asarray(np.finfo(np.dtype(x.dtype)).tiny, x.dtype)
            xg = xf.at[li].set(jnp.log(jnp.maximum(xf[li], tiny)))
        else:
            xg = xf
        gather = tricubic if self.kernel == "cubic" else trilinear
        pos, u, geo = self.positions(xi, with_shift=True)
        vals = gather(xg, pos.astype(x.dtype), self.n)
        if lf:
            dl = vals[li] - xg[li][:, self.act]                   # gathered ln x - ln x_c
            up = self.upstream
            if up is not None:
                if not {up["i_rho"], up["i_press"]} <= set(lf):
                    raise ValueError("the upstream correction needs rho and p among the log fields")
                pos_of = {k: j for j, k in enumerate(lf)}
                r = geo[2]
                ln_ratio = -jnp.log1p(-u / r).astype(x.dtype)     # ln(r' / r_src); exactly 0 at xi = 0
                log10_t = (vals[up["i_press"]] - vals[up["i_rho"]]) / np.log(10.0) + np.log10(up["t_per_code"])
                cold = jax.nn.sigmoid(-(log10_t - up["log_t_shock"]) / up["width"])
                b_rho, b_p = self.upstream_slopes(xi)
                corr = {up["i_rho"]: b_rho, up["i_press"]: b_p}
                if up["i_e"] is not None and up["i_e"] in pos_of:
                    corr[up["i_e"]] = b_p
                for k, b in corr.items():
                    dl = dl.at[pos_of[k]].add(cold * b.astype(x.dtype) * ln_ratio)
            xc = xf[li][:, self.act]
            # x_c (1 + expm1(.)), not x_c exp(.): XLA:CPU's f32 exp(0) need not be exactly 1, expm1(0) is 0
            vals = vals.at[li].set(xc + xc * jnp.expm1(dl))
        bnd = dict(self.bounds)
        if extra is not None and extra_bounds is not None:
            bnd.update({nv + k: extra_bounds for k in range(extra.shape[0])})
        for k, (lo, hi) in bnd.items():
            vals = vals.at[k].set(jnp.clip(vals[k], lo, hi))
        out = xf.at[:, self.act].set(vals).reshape(full.shape)
        if extra is None:
            return out
        return out[:nv], out[nv:]

    def background_part(self, xi):
        return {"b_warp": xi}

    def describe(self):
        d = self.r_diag
        lin, lout = self.limits(d["R_mean"])
        upd = self.upstream
        ups = ("; upstream kept (d ln rho / d ln r median %.2f sd %.2f, p %.2f sd %.2f)" % (
            upd["b_rho_median"], upd["b_rho_sd"], upd["b_p_median"], upd["b_p_sd"])) if upd else "; upstream resampled"
        return (f"warp l <= {self.lmax} ({self.n_coef} coef, {self.kernel}), sigma {self.sigma_pc:.4f} pc, "
                f"taper {self.taper}, "
                f"{self.n_act} active cells ({self.n_act / self.n ** 3:.3f}); R(n) l <= {d['lfit']}: mean "
                f"{d['R_mean']:.3f} [{d['R_min']:.3f}, {d['R_max']:.3f}] pc, fit rms {d['fit_rms']:.4f} pc; "
                f"saturation at <R>: inward {lin:.3f} pc, outward {lout:.3f} pc{ups}")


def field_rules(names):
    """{log_fields, bounds} for a casa_xfit_state variable list: rho, p, e in
    log space; mass fractions and the shocked fraction in [0, 1]; the shock
    histories >= 0."""
    names = list(names)
    log_fields = [k for k, nm in enumerate(names) if nm in ("rho", "press", "internal_energy")]
    bounds = {}
    for k, nm in enumerate(names):
        if nm.startswith("C_") or nm == "shocked_fraction":
            bounds[k] = (0.0, 1.0)
        elif nm in ("time_since_shock", "density_time"):
            bounds[k] = (0.0, None)
    return dict(log_fields=log_fields, bounds=bounds)


def sigma_pc_from_arcsec(sigma_arcsec, distance_kpc):
    return float(sigma_arcsec) * float(distance_kpc) * ARCSEC_PC_PER_KPC
# =============================================================================
# ============ ↑ The warp ↑ ===================================================
# =============================================================================
