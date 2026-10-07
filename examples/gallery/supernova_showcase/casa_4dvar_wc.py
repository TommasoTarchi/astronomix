"""
Weak-constraint (multiple-shooting) 4D-Var of Cas A: sub-windows of <= 5.4 yr,
an independent state control at each sub-window start, a model-error penalty
at every boundary, shared global parameters (``casa_4dvar --wc``).

Why: the strong-constraint ``casa_4dvar`` sees J through an 18-yr forecast from
the 2000 state; discrete solver paths near the reverse shock make it rough
(J jumps ~0.03-0.1 at ulp-level control changes) and the stage-3 analysis has
no descent along -g (review_v). Exact tangents are unusable beyond ~4 yr at
128^3, and the approximate ones are bounded but ~20 % biased. Breaking the
window into sub-windows keeps every forecast (and adjoint) within ~5 yr.

Control z = [z_strong, chi_1, ..., chi_K]: z_strong is the strong-constraint
control (``casa_4dvar_control``: chi_0 at 2000, the unshocked-CSM Y_lm
modulation, the globals); chi_k (5, m, m, m) is a whitened half-resolution
increment of the state at the boundary t_k through the SAME B^(1/2) shape
(Gaussian ell, sigma per field, remnant mask of the sub-window's reference
state) about a fixed reference state x_k^ref (the strong-constraint analysis
forecast to t_k, ``make_refs``): x_k = x_k^ref (+) U_k chi_k, with the CSM
modulation change (xi - xi_ref) applied to x_k's never-shocked gas (static CSM:
the same Y_lm field as at 2000).

    J = J_strong,b(z_strong) + 0.5 |r_obs|^2 + sum_k [0.5 |x_k^- - x_k|^2_Q + 0.5 |P_H chi_k|^2]

r_obs: the same casa_xfit residuals over all training epochs, epoch e computed
from the trajectory of the sub-window containing it; x_k^-: the forecast from
x_{k-1} (x_0 for k = 1) arriving at t_k. Everything runs in ONE ``lax.scan``
over the event list (epochs and boundaries: integrate, reset at a boundary,
observe at an epoch), so the solver is compiled once and memory stays that of
the strong-constraint window.

Model-error norm |d|_Q, Q^(1/2) = alpha U (same smoothing operator, amplitude
alpha x the background sigma): on the coarse grid U U^T ~ sigma^2 c^2 S S^T with
S the Gaussian of sd ell (symbol G(k)) and c^2 = 1 / mean(G^2) (pointwise sd
sigma), so

    |d|_Q^2 = sum_f sum_{k: G(k) >= g_cut} |FFT(R M d_f)|_k^2 / (N alpha^2 sigma_f^2 c^2 G(k)^2)

(R = 2^3 block average, M = the sub-window's remnant mask, N = m^3). The inverse
of a Gaussian covariance explodes at high k (G ~ e^-79 at Nyquist for ell = 2),
so it is evaluated on its well-conditioned band G >= g_cut (a reduced-rank Q):
sub-correlation-length mismatch, which is where the forecast differences are
discrete-path noise (shock positions flipping by a cell), is not penalised by
the continuity term; the matching high-k part of chi_k gets the unit background
0.5 |P_H chi_k|^2 instead (without it those components would be free). The
misfit variables are those of the control: ln rho, v / v_ref, and ln(p + p_c)
with p_c = ``p_c_frac`` x the mean pressure of the reference's shocked gas (cold
ejecta near the pressure floor carry no dynamical information; their huge
relative ln p changes would otherwise dominate the norm).

alpha (default 0.3): the model error accumulated over one <= 5 yr sub-window,
relative to the 2000 background uncertainty (sigma = 0.3 / 0.5 / 0.3 in ln rho,
v / 1000 km/s, ln p). 0.3 sigma = 9 % in rho and p, 150 km/s in v (~3 % of the
forward-shock speed) per sub-window: the size of the known model deficits
over a few years (128^3 numerical diffusion at the shocks, the missing
CR-modified-shock and CSM physics that make R sit on its priors) and of the
approximate-tangent bias (~20 %) times the analysis increments. alpha -> 0
recovers the strong constraint; alpha -> 1 would let each sub-window fit
nearly independently (losing the dynamical link). The penalty weight 1/alpha^2
= 11 x the background's keeps the model the dominant constraint on x_k.
"""
import numpy as np
import jax
import jax.numpy as jnp

import casa_xfit as X
import casa_4dvar_control as C


def coarse_gaussian_symbol(m, ell):
    """G(k) of the Gaussian of sd ``ell`` (cells) on an m^3 periodic grid (fftn layout)."""
    k = 2.0 * np.pi * np.fft.fftfreq(m)
    k2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
    return np.exp(-0.5 * ell ** 2 * k2)


def qnorm_weights(m, ell, sigma, alpha, gcut):
    """(W2 (5, m, m, m), keep (m, m, m)): |d|_Q^2 = sum W2 |fftn(d_coarse)|^2."""
    G = coarse_gaussian_symbol(m, ell)
    c2 = 1.0 / np.mean(G ** 2)
    keep = G >= gcut
    N = float(m) ** 3
    W2 = np.stack([np.where(keep, 1.0 / (N * alpha ** 2 * s ** 2 * c2 * np.maximum(G, 1e-30) ** 2), 0.0)
                   for s in sigma]).astype(np.float32)
    return W2, keep


def event_steps(years_sorted, t0_year, bounds, first_at_x0):
    """The scan steps: dict of numpy arrays (time [yr], reset flag, boundary
    index, observe flag, epoch index into the sorted epochs) for the epochs
    (except the first if it is observed at x0) and the boundaries."""
    ys = np.asarray(years_sorted, np.float64)
    bounds = sorted(float(b) for b in bounds)
    for b in bounds:
        if b <= t0_year + 1e-6:
            raise ValueError(f"boundary {b} not after the state epoch {t0_year}")
        if np.any(np.abs(ys - b) < 1e-3):
            raise ValueError(f"boundary {b} coincides with an epoch")
    ev = [(y, 0, i) for i, y in enumerate(ys) if not (i == 0 and first_at_x0)] + \
         [(b, 1, k) for k, b in enumerate(bounds)]
    ev.sort(key=lambda e: (e[0], e[1]))
    t = np.array([e[0] for e in ev])
    reset = np.array([e[1] == 1 for e in ev])
    kidx = np.array([e[2] if e[1] == 1 else 0 for e in ev], np.int32)
    obsf = ~reset
    # boundary steps carry the NEXT epoch's inputs (not observed; flags off)
    ep = np.zeros(len(ev), np.int32)
    nxt = len(ys) - 1
    for j in range(len(ev) - 1, -1, -1):
        if obsf[j]:
            nxt = ev[j][2]
        ep[j] = ev[j][2] if obsf[j] else nxt
    sub = np.searchsorted(np.asarray(bounds), ys, side="right")          # sub-window of each epoch
    return dict(t=t, dt_yr=np.diff(np.concatenate([[t0_year], t])), reset=reset, k=kidx, obs=obsf, ep=ep,
                bounds=np.asarray(bounds), sub_of_epoch=sub)


def reference_fields(ref, layout):
    lay = {int(k): v for k, v in layout.items()}
    return {lay[i]: np.asarray(ref[i]) for i in range(len(lay))}


def make_refs(base, ctrl, z_strong, bounds, dtype=jnp.float32):
    """The reference states x_k^ref at the boundaries: the forecast from the
    strong-constraint control ``z_strong`` through the same event steps as the
    weak-constraint scan (resets off). Returns (K, num_vars, n, n, n) float32."""
    core = base.core
    ys = np.sort(np.asarray(base.obs["years"], np.float64))
    steps = event_steps(ys, base.t0_year, bounds, base.first_at_x0)
    chi, xi, _ = ctrl.split(jnp.asarray(z_strong, dtype))
    x = ctrl.state(chi, xi)
    integ = jax.jit(core.integrator(x.shape))
    refs = []
    last = int(np.nonzero(steps["reset"])[0].max())
    for j in range(last + 1):
        x = integ(x, jnp.asarray(steps["dt_yr"][j] * core.yr, dtype))
        if steps["reset"][j]:
            refs.append(np.asarray(jax.block_until_ready(x), np.float32))
            print(f"[wc] reference state at {steps['t'][j]:.2f} (boundary {int(steps['k'][j]) + 1})", flush=True)
    return np.stack(refs)


class WCWindow:
    """J and its gradient of the weak-constraint problem over the epochs of
    ``base`` (a ``casa_4dvar.Window``: data subset, forward core, likelihood
    settings). ``refs``: (K, nv, n, n, n) reference states at ``bounds``;
    ``xi_ref``: the CSM control the references were made with."""

    def __init__(self, base, bounds, refs, *, xi_ref, alpha=0.3, gcut=0.1, p_c_frac=0.01, dtype=jnp.float32):
        self.base, self.ctrl, self.dtype = base, base.ctrl, dtype
        ctrl = self.ctrl
        self.labels, self.obs, self.img = base.labels, base.obs, base.img
        ys = np.sort(np.asarray(base.obs["years"], np.float64))
        self.steps = event_steps(ys, base.t0_year, bounds, base.first_at_x0)
        self.bounds = self.steps["bounds"]
        self.K = len(self.bounds)
        if refs.shape[0] != self.K:
            raise ValueError(f"{refs.shape[0]} reference states for {self.K} boundaries")
        self.alpha, self.gcut, self.p_c_frac = float(alpha), float(gcut), float(p_c_frac)
        self.size = ctrl.size + self.K * ctrl.n_chi
        self.rough = np.zeros(self.size, bool)
        self.rough[:ctrl.n_chi] = True
        self.rough[ctrl.size:] = True
        self.xi_ref = np.asarray(xi_ref, np.float64)
        lay = ctrl.layout
        masks, ucsm, pc = [], [], []
        for k in range(self.K):
            f = reference_fields(refs[k], lay)
            mk = C.remnant_mask(f["C_ej"], f["shocked_fraction"])
            cej = np.clip(np.asarray(f["C_ej"], np.float64), 0.0, 1.0)
            sf = np.clip(np.asarray(f["shocked_fraction"], np.float64), 0.0, 1.0)
            masks.append(mk)
            ucsm.append((1.0 - cej) * (1.0 - sf) * (1.0 - mk))
            hot = sf > 0.5
            pc.append(self.p_c_frac * float(np.mean(np.asarray(f["press"], np.float64)[hot])) if hot.any()
                      else 0.0)
        W2, keep = qnorm_weights(ctrl.m, ctrl.ell, ctrl.sigma, self.alpha, self.gcut)
        self.n_keep = int(keep.sum())
        self.aux = dict(refs=jnp.asarray(refs, dtype), masks=jnp.asarray(np.stack(masks), dtype),
                        ucsm=jnp.asarray(np.stack(ucsm), dtype), pc=jnp.asarray(pc, dtype),
                        W2=jnp.asarray(W2, dtype), hk=jnp.asarray(~keep, dtype))
        self.mask_frac = [float(np.mean(mk)) for mk in masks]
        self.p_c = pc

    # ---- pieces ---------------------------------------------------------------------
    def split(self, z):
        ctrl = self.ctrl
        m = ctrl.m
        return z[:ctrl.size], z[ctrl.size:].reshape(self.K, 5, m, m, m)

    def boundary_states(self, z, aux):
        zs, chis = self.split(z)
        _, xi, _ = self.ctrl.split(zs)
        dxi = xi - jnp.asarray(self.xi_ref, xi.dtype)
        return jnp.stack([self.ctrl.state_on(aux["refs"][k], chis[k], dxi, aux["masks"][k], aux["ucsm"][k])
                          for k in range(self.K)])

    def qnorm(self, st, xk, mk, pck, W2):
        """|st - xk|_Q^2 (the model-error norm, see the module docstring)."""
        ctrl = self.ctrl
        I, vr, m = ctrl.idx, ctrl.v_ref, ctrl.m
        tiny = jnp.asarray(1e-30, st.dtype)

        def h(s):
            return jnp.stack([jnp.log(jnp.maximum(s[I["rho"]], tiny)), s[I["vx"]] / vr, s[I["vy"]] / vr,
                              s[I["vz"]] / vr, jnp.log(jnp.maximum(s[I["press"]], 0.0) + pck + tiny)])
        d = (h(st) - h(xk)) * mk[None]
        dc = d.reshape(5, m, 2, m, 2, m, 2).mean((2, 4, 6))
        F = jnp.fft.fftn(dc, axes=(1, 2, 3))
        return jnp.sum((jnp.real(F) ** 2 + jnp.imag(F) ** 2) * W2)

    def forward(self, x0, Xs, p, aux):
        """(model dict, continuity chi2 per boundary (K,))."""
        base, core = self.base, self.base.core
        integrate = core.integrator(x0.shape)
        xs = core.xs_all
        S = self.steps
        ep_idx = jnp.asarray(S["ep"])
        xs_steps = jax.tree.map(lambda v: jnp.asarray(v)[ep_idx], xs)
        obsf = jnp.asarray(S["obs"])
        for key in ("has", "dop"):
            if key in xs_steps:
                xs_steps[key] = jnp.logical_and(xs_steps[key], obsf)
        ep0 = jax.tree.map(lambda v: v[0], xs)
        shapes = jax.eval_shape(lambda st, e: core.observer(p)(st, e), x0, ep0)

        @jax.checkpoint
        def seg(st, dt, reset, k, of, ep, pp, Xs_, masks, pcs, W2):
            st_int = integrate(st, dt)
            xk = Xs_[k]
            pen = jax.lax.cond(reset, lambda: self.qnorm(st_int, xk, masks[k], pcs[k], W2),
                               lambda: jnp.zeros((), st.dtype))
            st_new = jnp.where(reset, xk, st_int)
            out = jax.lax.cond(of, lambda: core.observer(pp)(st_new, ep),
                               lambda: jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), shapes))
            return st_new, (out, pen)

        def body(st, xs_):
            dt, reset, k, of, ep = xs_
            return seg(st, dt, reset, k, of, ep, p, Xs, aux["masks"], aux["pc"], aux["W2"])

        dts = jnp.asarray(S["dt_yr"] * core.yr, self.dtype)
        _, (outs, pens) = jax.lax.scan(body, x0, (dts, jnp.asarray(S["reset"]), jnp.asarray(S["k"]), obsf,
                                                  xs_steps))
        oi = np.nonzero(S["obs"])[0]
        outs = tuple(o[oi] for o in outs)
        if base.first_at_x0:
            first = core.observer(p)(x0, ep0)
            outs = tuple(jnp.concatenate([f[None], b], 0) for f, b in zip(first, outs))
        ri = np.nonzero(S["reset"])[0]
        cont = pens[ri][np.argsort(S["k"][ri])]
        return core.assemble(p, outs), cont

    def parts_chi2(self, z, dz, aux):
        """({part: chi2}, model, x0): the background terms at z, the model
        terms (observations, wind prior on x0, continuity) at z + dz."""
        base, ctrl = self.base, self.ctrl
        zm = z + dz
        zs, chis = self.split(z)
        zsm, _ = self.split(zm)
        chi0, xi, xg = ctrl.split(zs)
        chi0m, xim, xgm = ctrl.split(zsm)
        x0 = ctrl.state(chi0m, xim)
        Xs = self.boundary_states(zm, aux)
        g = ctrl.globals_of(xg)
        p = base.params(g)
        model, cont = self.forward(x0, Xs, p, aux)
        theta = jnp.stack([p[k] for k in X.PARAM_NAMES])
        parts = X.residual_parts(model, base.obs, base.img, theta, base.largs)
        for k in base.drop_parts:
            parts.pop(k, None)
        if ctrl.gnames:
            parts["prior_glob"] = jnp.stack([(g[k] - base.prior[k][0]) / base.prior[k][1] for k in ctrl.gnames])
        if base.wind_prior:
            parts["wind_nh"] = jnp.atleast_1d((base.wind_nh(x0) - base.wind_nh_mu) / base.wind_nh_sd)
        parts.update(ctrl.background_parts(chi0, xi))
        chi2 = {k: jnp.sum(v ** 2) for k, v in parts.items()}
        N = float(ctrl.m) ** 3
        for k in range(self.K):
            chi2[f"cont{k + 1}"] = cont[k]
            Fk = jnp.fft.fftn(chis[k], axes=(1, 2, 3))
            chi2[f"b_hk{k + 1}"] = jnp.sum((jnp.real(Fk) ** 2 + jnp.imag(Fk) ** 2) * aux["hk"]) / N
        return chi2, model, x0

    def objective(self, z, dz, aux):
        chi2, _, _ = self.parts_chi2(z, dz, aux)
        return 0.5 * sum(chi2.values()), chi2

    def compile(self, grad=True):
        aux = self.aux
        vg = jax.jit(jax.value_and_grad(self.objective, has_aux=True))
        val = jax.jit(self.objective)
        self.vg = (lambda z, dz: vg(z, dz, aux)) if grad else None
        self.val = lambda z, dz: val(z, dz, aux)
        return self

    def states_np(self, z):
        """(x0, (K, nv, ...) boundary states) of a control (host arrays)."""
        zj = jnp.asarray(z, self.dtype)
        zs, _ = self.split(zj)
        chi0, xi, _ = self.ctrl.split(zs)
        return np.asarray(self.ctrl.state(chi0, xi)), np.asarray(self.boundary_states(zj, self.aux))
