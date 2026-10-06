"""
Control-variable transform of the Cas A full-state 4D-Var (``casa_4dvar``).

The control vector ``z`` (float64 on the host, whitened: its background term is
0.5 |z|^2 except for the global parameters, which carry the xfit priors) is

* ``chi`` (5, m, m, m), m = n / 2: the half-resolution field of (delta ln rho,
  delta v_x, delta v_y, delta v_z [units of ``v_ref``], delta ln p);
* ``xi_csm`` (10,): the unshocked-CSM ln-density modulation, real Y_lm with
  l <= 2 (9 amplitudes, sigma ``sigma_csm``) and a radial power-law slope
  (``sigma_slope``, delta ln rho = s ln(r / r_ref));
* ``xi_g``: the free global parameters (casa_xfit names), g = g_b + s_k xi_k
  (``s_k``: a preconditioning scale, not a prior);
* ``xi_w`` (optional, ``warp=``; stage 4): the forward-shock displacement
  coefficients of ``casa_4dvar_warp.Warp`` ((L + 1)^2 real Y_lm, whitened),
  appended after the globals so that a control vector without them is a prefix (``pad_z``);
* ``chi_f`` (optional, ``fine_ell=``; 2026-09-27): a second, FINE-scale white
  field on the same half-resolution grid (5, m, m, m), smoothed with ``fine_ell``
  coarse cells and scaled by ``fine_sigma`` x sigma_k -- the two-level control
  (the smooth ``chi`` carries the large-scale shape, ``chi_f`` the filament-scale
  knots at a smaller amplitude); appended LAST (a control vector without it is a
  prefix, ``pad_z``).

With a warp the background is first displaced radially (the shocked shell, the
forward shock and the CSM just ahead), then the increments below are applied
about the warped background, with the remnant mask and the never-shocked-CSM
weight warped along (``warp_mask``), so the controlled band follows the shock.

The state increment is the square-root background covariance

    u_k = sigma_k * Mask * Smooth_ell(Prolong(chi_k)) / norm
          [+ fine_sigma * sigma_k * Mask * Smooth_fine_ell(Prolong(chi_f_k)) / norm_f],

``Prolong`` = 2x piecewise-constant, ``Smooth_ell`` = Gaussian of standard
deviation ``ell`` COARSE cells applied by FFT on the (periodic) box, ``norm``
such that white chi gives a pointwise standard deviation sigma_k where
Mask = 1; ``Mask`` = the remnant interior (ejecta or shocked gas) dilated to
just beyond the forward shock, with a smooth edge.

x0 from the background x_b (the saved 2000 state):

    rho = rho_b exp(u_lnrho + q),  p = p_b exp(u_lnp + q),  v = v_b + v_ref u_v,
    q   = U_csm (sum_lm a_lm Y_lm(n) + s ln(r / r_ref))   (isothermal CSM modulation)
    g   = p / (gamma - 1)                                  (dual energy)
    s0  = s0_b + delta ln p - gamma delta ln rho           (entropy label: the
                                                            shock test stays as it was)

with U_csm = (1 - C_ej)(1 - shocked_fraction)(1 - Mask), the never-shocked
circumstellar gas beyond the controlled region. Composition and the remaining history fields are carried unchanged.
"""
import numpy as np
import jax
import jax.numpy as jnp

import casa_xfit_shard as SH

FIELDS = ("lnrho", "vx", "vy", "vz", "lnp")
#: real spherical harmonics l <= 2 in Cartesian form (O(1) amplitude, as casa_xfit.YLM)
CSM_YLM = {
    "c00": lambda x, y, z: jnp.ones_like(x),
    "c11x": lambda x, y, z: x, "c11y": lambda x, y, z: y, "c10": lambda x, y, z: z,
    "c22xy": lambda x, y, z: 3 * x * y, "c21yz": lambda x, y, z: 3 * y * z,
    "c20": lambda x, y, z: 1.5 * z * z - 0.5, "c21xz": lambda x, y, z: 3 * x * z,
    "c22c": lambda x, y, z: 1.5 * (x * x - y * y),
}
CSM_NAMES = tuple(CSM_YLM) + ("c_slope",)


def gaussian_filter_fft(a, sigma_cells):
    """Periodic Gaussian smoothing (standard deviation ``sigma_cells``) of the
    last three axes (numpy or jax arrays)."""
    xp = jnp if isinstance(a, jax.Array) else np
    n = a.shape[-3:]
    k2 = sum(np.meshgrid(*[(2 * np.pi * np.fft.fftfreq(m)) ** 2 for m in n[:2]], indexing="ij"))
    kz = (2 * np.pi * np.fft.rfftfreq(n[2])) ** 2
    G = np.exp(-0.5 * sigma_cells ** 2 * (k2[..., None] + kz[None, None]))
    out = xp.fft.irfftn(xp.fft.rfftn(a, axes=(-3, -2, -1)) * xp.asarray(G, dtype=a.dtype),
                        s=n, axes=(-3, -2, -1))
    return out.astype(a.dtype)


def prolong2(a):
    """2x piecewise-constant prolongation of the last three axes."""
    for ax in (-3, -2, -1):
        a = jnp.repeat(a, 2, axis=ax)
    return a


def bhalf_norm(n, ell_coarse):
    """1 / (pointwise rms of Smooth(Prolong(white)))."""
    e = np.zeros((n, n, n))
    e[:2, :2, :2] = 1.0
    h = gaussian_filter_fft(e, 2.0 * ell_coarse)
    return 1.0 / np.sqrt(np.sum(h ** 2) / 8.0)


def remnant_mask(c_ej, shocked, *, dilate_cells=3.0, edge_cells=1.5, thresh=1e-3):
    """Smooth [0, 1] mask: ejecta or shocked gas, dilated by ~``dilate_cells``
    (fine cells) beyond the forward shock, Gaussian edge ``edge_cells``."""
    core = ((np.asarray(c_ej) > thresh) | (np.asarray(shocked) > thresh)).astype(np.float64)
    grown = (gaussian_filter_fft(core, dilate_cells / 2.0) > 0.02).astype(np.float64)
    return np.clip(gaussian_filter_fft(grown, edge_cells), 0.0, 1.0)


def sanitize_entropy_label(fields, gamma=5.0 / 3.0, window=None):
    """Load-time safety net for the shock-history label ``entropy_initial``:
    re-seed a non-finite or runaway label (more than ``window`` nats, default
    the library's ``ENTROPY_LABEL_WINDOW``, from the current specific entropy)
    with the current entropy, i.e. "not shocked relative to now" -- the reset
    the library applies every step (``sanitize_entropy_label``). A background
    written before that fix can hold NaN labels (448^3 R4b, 2026-10-04: all of
    them), which make every reverse-mode gradient NaN through the first scalar
    advection. Returns ``(fields, n_reset)``; ``fields`` unchanged if clean."""
    if "entropy_initial" not in fields:
        return fields, 0
    if window is None:
        from astronomix._fluid_equations._passive_scalars import ENTROPY_LABEL_WINDOW
        window = ENTROPY_LABEL_WINDOW
    s0 = np.asarray(fields["entropy_initial"])
    rho = np.maximum(np.asarray(fields["rho"], np.float64), 1e-30)
    p = np.maximum(np.asarray(fields["press"], np.float64), 1e-30)
    s_now = np.log(p) - gamma * np.log(rho)
    with np.errstate(invalid="ignore"):
        bad = ~(np.abs(s_now - s0) <= window)
    n = int(bad.sum())
    if n:
        fields = dict(fields, entropy_initial=np.where(bad, s_now, s0).astype(s0.dtype))
    return fields, n


def block_average_state(fields, n_out):
    """Conservative block average of a saved state's fields to ``n_out``^3 (CPU
    smoke tests): mass, momentum and energy densities averaged, primitive
    fields rebuilt; scalars / history mass-weighted (entropy label: from the
    mass-weighted mean of exp(s0))."""
    n = fields["rho"].shape[0]
    f = n // n_out
    if f * n_out != n:
        raise ValueError(f"{n} not divisible by {n_out}")

    def B(a):
        return np.asarray(a, np.float64).reshape(n_out, f, n_out, f, n_out, f).mean((1, 3, 5))
    rho = np.asarray(fields["rho"], np.float64)
    out = {"rho": B(rho)}
    for k in ("vx", "vy", "vz"):
        out[k] = B(rho * fields[k]) / out["rho"]
    ekin = 0.5 * rho * sum(np.asarray(fields[k], np.float64) ** 2 for k in ("vx", "vy", "vz"))
    ekin_b = 0.5 * out["rho"] * sum(out[k] ** 2 for k in ("vx", "vy", "vz"))
    gm = 5.0 / 3.0
    etot = B(np.asarray(fields["press"], np.float64) / (gm - 1.0) + ekin)
    out["press"] = np.maximum((gm - 1.0) * (etot - ekin_b), 1e-3 * B(fields["press"]))
    for k, v in fields.items():
        if k in out or k in ("internal_energy",):
            continue
        if k == "entropy_initial":
            out[k] = np.log(B(rho * np.exp(np.clip(np.asarray(v, np.float64), -80, 80))) / out["rho"])
        else:
            out[k] = B(rho * np.asarray(v, np.float64)) / out["rho"]
    if "internal_energy" in fields:
        out["internal_energy"] = out["press"] / (gm - 1.0)
    return {k: v.astype(np.float32) for k, v in out.items()}


class Control:
    """The control-to-state map for one background state (see the module
    docstring). ``fields``: the saved state's fields (``casa_xfit_state.load_state``);
    ``layout``: {state index: name}; ``geom``: (r, X, Y, Z) of the solver grid
    (code length = pc); ``globals_b``: {name: background value} of the FREE
    global parameters, ``globals_scale``: their preconditioning scale."""

    def __init__(self, fields, layout, geom, *, ell=2.0, sigma=(0.3, 0.5, 0.5, 0.5, 0.3), v_ref=1.0,
                 sigma_csm=0.3, sigma_slope=0.5, r_ref=None, csm=True, globals_b=None, globals_scale=None,
                 gamma=5.0 / 3.0, dtype=jnp.float32, mask_dilate=3.0, mask_edge=1.5, warp=None,
                 warp_mask=True, fine_ell=None, fine_sigma=0.3):
        self.layout = {int(k): v for k, v in layout.items()}
        self.names = [self.layout[k] for k in range(len(self.layout))]
        self.idx = {v: k for k, v in self.layout.items()}
        n = fields["rho"].shape[0]
        if n % 2:
            raise ValueError("the control grid is n / 2: n must be even")
        self.n, self.m = n, n // 2
        self.dtype = dtype
        self.gamma = gamma
        # multi-GPU (casa_xfit_shard): SH.put places these split along x
        # (jnp.asarray on one device, as before)
        self.xb = SH.put(np.stack([np.asarray(fields[nm], np.float32) for nm in self.names]), SH.STATE, dtype)
        self.mask_np = remnant_mask(fields["C_ej"], fields["shocked_fraction"], dilate_cells=mask_dilate,
                                    edge_cells=mask_edge)
        self.mask = SH.put(self.mask_np, SH.FIELD, dtype)
        cej = np.clip(np.asarray(fields["C_ej"], np.float64), 0.0, 1.0)
        sf = np.clip(np.asarray(fields["shocked_fraction"], np.float64), 0.0, 1.0)
        # never-shocked circumstellar gas (the wind prior's weight), and the
        # modulation's support: that gas OUTSIDE the controlled region (the
        # band just ahead of the forward shock belongs to chi)
        self.u_csm_np = (1.0 - cej) * (1.0 - sf)
        self.w_csm_np = self.u_csm_np * (1.0 - self.mask_np)
        self.u_csm = SH.put(self.w_csm_np, SH.FIELD, dtype)
        self.ell = float(ell)
        self.sigma = tuple(float(s) for s in sigma)
        self.v_ref = float(v_ref)
        self.norm = float(bhalf_norm(n, ell))
        self.csm = bool(csm)
        r, X, Y, Z = geom
        rs = jnp.maximum(r, 1e-3)
        nx, ny, nz = X / rs, Y / rs, Z / rs
        if r_ref is None:            # the mean radius of the shocked / unshocked boundary
            w = self.mask_np * (1.0 - self.mask_np)
            r_ref = float(np.sum(np.asarray(r) * w) / max(np.sum(w), 1e-30))
        self.r_ref = float(r_ref)
        self.basis = jnp.stack([fn(nx, ny, nz) for fn in CSM_YLM.values()]
                               + [jnp.log(rs / self.r_ref)]).astype(dtype)          # (10, n, n, n)
        if SH.active():
            self.basis = jax.device_put(self.basis, SH.sharding(SH.STATE))
        self.csm_sigma = jnp.asarray([sigma_csm] * len(CSM_YLM) + [sigma_slope], dtype)
        self.gnames = tuple(globals_b or {})
        self.g_b = np.array([globals_b[k] for k in self.gnames], np.float64)
        self.g_s = np.array([(globals_scale or {}).get(k, 1.0) for k in self.gnames], np.float64)
        self.n_chi = 5 * self.m ** 3
        self.n_csm = len(CSM_NAMES) if self.csm else 0
        self.warp = warp
        self.warp_mask = bool(warp_mask)
        self.n_warp = warp.n_coef if warp is not None else 0
        self.off_g = self.n_chi + self.n_csm
        self.off_w = self.off_g + len(self.gnames)
        # the fine-scale second level (two-level control; None: off, the layout unchanged)
        self.fine_ell = None if not fine_ell else float(fine_ell)
        self.fine_sigma = float(fine_sigma) if self.fine_ell else 0.0
        self.norm_f = float(bhalf_norm(n, self.fine_ell)) if self.fine_ell else 0.0
        self.n_fine = 5 * self.m ** 3 if self.fine_ell else 0
        self.off_f = self.off_w + self.n_warp
        self.size = self.off_f + self.n_fine

    def lift(self, lifted):
        """Register the background arrays with a ``casa_xfit_shard.Lifted``
        (multi-GPU: sharded jit arguments instead of HLO constants)."""
        lifted.add(self, "xb", SH.STATE).add(self, "mask", SH.FIELD).add(self, "u_csm", SH.FIELD)
        lifted.add(self, "basis", SH.STATE)
        return lifted

    # ---- flat vector <-> pieces -------------------------------------------------
    def split(self, z):
        m = self.m
        chi = z[:self.n_chi].reshape(5, m, m, m)
        xi_csm = z[self.n_chi:self.n_chi + self.n_csm]
        xi_g = z[self.off_g:self.off_w]
        return chi, xi_csm, xi_g

    def warp_of(self, z):
        """The warp coefficients xi_w (None without a warp)."""
        return z[self.off_w:self.off_w + self.n_warp] if self.n_warp else None

    def fine_of(self, z):
        """The fine-scale field chi_f (5, m, m, m) (None without the second level)."""
        if not self.n_fine:
            return None
        m = self.m
        return z[self.off_f:self.off_f + self.n_fine].reshape(5, m, m, m)

    def pad_z(self, z):
        """A control vector of an older layout (no warp / no fine level: a prefix)
        padded with zero warp coefficients and a zero fine field."""
        z = np.asarray(z, np.float64)
        if z.size == self.size - self.n_fine - self.n_warp and (self.n_warp or self.n_fine):
            z = np.concatenate([z, np.zeros(self.n_warp + self.n_fine)])
        elif z.size == self.size - self.n_warp and self.n_warp and not self.n_fine:
            z = np.concatenate([z, np.zeros(self.n_warp)])
        elif z.size == self.size - self.n_fine and self.n_fine:
            z = np.concatenate([z, np.zeros(self.n_fine)])
        return z

    def globals_of(self, xi_g):
        """{name: value} of the free globals (traced)."""
        return {k: jnp.asarray(self.g_b[i], xi_g.dtype) + jnp.asarray(self.g_s[i], xi_g.dtype) * xi_g[i]
                for i, k in enumerate(self.gnames)}

    def globals_np(self, z):
        xi_g = np.asarray(z, np.float64)[self.off_g:self.off_w]
        return {k: float(self.g_b[i] + self.g_s[i] * xi_g[i]) for i, k in enumerate(self.gnames)}

    # ---- the transform ------------------------------------------------------------
    def _smooth(self, chi, ell):
        if SH.active():         # x-split prolongation + distributed FFT (one shard_map)
            return SH.smooth_prolong(chi.astype(self.dtype), 2.0 * ell)
        return gaussian_filter_fft(prolong2(chi.astype(self.dtype)), 2.0 * ell)

    def increments(self, chi, mask=None, chi_f=None):
        """(5, n, n, n) state increments u_k (sigma-scaled, masked; ``mask``:
        another state's remnant mask, default this background's; ``chi_f``: the
        fine-scale level, if the control has one)."""
        sm = self._smooth(chi, self.ell)
        sig = jnp.asarray(self.sigma, self.dtype)[:, None, None, None]
        mk = (self.mask if mask is None else mask)[None]
        if chi_f is None or not self.n_fine:
            return sig * self.norm * mk * sm
        smf = self._smooth(chi_f, self.fine_ell)
        return sig * mk * (self.norm * sm + (self.fine_sigma * self.norm_f) * smf)

    def csm_lnrho(self, xi_csm, u_csm=None):
        u_csm = self.u_csm if u_csm is None else u_csm
        if not self.csm:
            return jnp.zeros_like(u_csm)
        a = self.csm_sigma * xi_csm.astype(self.dtype)
        return u_csm * jnp.tensordot(a, self.basis, 1)

    def state(self, chi, xi_csm, xi_w=None, chi_f=None):
        """x0 (num_vars, n, n, n) for the control pieces (``xi_w``: the warp
        coefficients, applied to the background BEFORE the increments; ``chi_f``:
        the fine-scale level)."""
        if xi_w is None or self.warp is None:
            return self.state_on(self.xb, chi, xi_csm, chi_f=chi_f)
        if self.warp_mask:
            xw, ex = self.warp.apply(self.xb, xi_w, extra=jnp.stack([self.mask, self.u_csm]))
            return self.state_on(xw, chi, xi_csm, mask=ex[0], u_csm=ex[1], chi_f=chi_f)
        return self.state_on(self.warp.apply(self.xb, xi_w), chi, xi_csm, chi_f=chi_f)

    def state_z(self, z):
        """x0 for a full control vector (all pieces, the warp and the fine level included)."""
        chi, xi_csm, _ = self.split(z)
        return self.state(chi, xi_csm, self.warp_of(z), self.fine_of(z))

    def state_on(self, xb, chi, xi_csm, mask=None, u_csm=None, chi_f=None):
        """The same transform about another background state ``xb`` (e.g. a
        weak-constraint sub-window start, ``casa_4dvar_wc``) with ITS remnant
        mask and never-shocked-CSM weight (defaults: this background's)."""
        u = self.increments(chi, mask, chi_f)
        q = self.csm_lnrho(xi_csm, u_csm)
        dlr, dlp = u[0] + q, u[4] + q
        I = self.idx
        x = xb.at[I["rho"]].set(xb[I["rho"]] * jnp.exp(dlr))
        x = x.at[I["press"]].set(xb[I["press"]] * jnp.exp(dlp))
        for k, name in enumerate(("vx", "vy", "vz")):
            x = x.at[I[name]].set(xb[I[name]] + self.v_ref * u[1 + k])
        if "internal_energy" in I:
            x = x.at[I["internal_energy"]].set(x[I["press"]] / (self.gamma - 1.0))
        if "entropy_initial" in I:
            x = x.at[I["entropy_initial"]].set(xb[I["entropy_initial"]] + dlp - self.gamma * dlr)
        return SH.cstate(x)

    def background_parts(self, chi, xi_csm, xi_w=None, chi_f=None):
        out = {"b_state": chi.ravel(), **({"b_csm": xi_csm} if self.csm else {})}
        if xi_w is not None and self.n_warp:
            out["b_warp"] = xi_w
        if chi_f is not None and self.n_fine:
            out["b_fine"] = chi_f.ravel()
        return out

    # ---- directions -----------------------------------------------------------------
    def random_direction(self, seed, *, what="chi", rms=1.0):
        """A random control direction (float64 host vector): ``what`` in
        {"chi", "globals", "csm", "warp", "fine", "all"}; per-component rms ``rms`` on the
        selected part (so eps * d is a eps-sigma increment)."""
        rng = np.random.default_rng(1000 + seed)
        d = np.zeros(self.size)
        sl = {"chi": slice(0, self.n_chi), "csm": slice(self.n_chi, self.n_chi + self.n_csm),
              "globals": slice(self.off_g, self.off_w), "warp": slice(self.off_w, self.off_f),
              "fine": slice(self.off_f, self.size), "all": slice(0, self.size)}[what]
        d[sl] = rms * rng.normal(size=d[sl].size)
        return d
