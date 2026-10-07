"""
Tangent-growth probe for forward-mode AD through the Cas A FD solver.

Propagates the JVP of a state (``casa_pluto_diff`` IC format) with respect to
one parameter of ``casa_pluto_diff.transform_fields`` in strides, and prints,
per stride, the largest and the 99.9th-percentile |d rho / d theta|, where the
largest sits (radius, temperature, ejecta fraction) and how much of the tangent
"mass" sits in each region. This is how the NaN tangents were traced to the
positivity limiter, the WENO weights / eigensystem, the LLF speed, and then to
growth in the shocked interior (PLUTO150.md section 5).

    pq sub -t a100 -n 1 -- ./run.sh casa_pluto_jvp_probe.py --ic IC.npz \\
        --param ln_sv --stride 1 --strides 12 [--filter-sigma 1.0]
"""

# ==== GPU selection ====
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as u
import astropy.constants as const

from astronomix import (PositivityConfig, SimulationParams, finalize_config,
                        get_helper_data, get_registered_variables, time_integration)
from _common import (GAMMA, MASS_PER_NUCLEUS, make_fd_config,
                     snr_code_units)
from casa_pluto_diff import PARAM_NAMES, PRIOR, THETA0, make_initial_state, tangent_filter


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ic", required=True)
    ap.add_argument("--param", default="ln_sv", choices=PARAM_NAMES)
    ap.add_argument("--theta", type=float, nargs="+", default=None,
                    help="padded with the prior means if shorter")
    ap.add_argument("--stride", type=float, default=2.0, help="years per stride")
    ap.add_argument("--strides", type=int, default=20)
    ap.add_argument("--filter-sigma", type=float, default=0.0,
                    help="conservative Gaussian low-pass on the tangent after each "
                         "stride (cells); 0 = off")
    ap.add_argument("--pos-off", nargs="*", default=[],
                    choices=("pp", "coldcrush_blend", "dual_energy"),
                    help="switch individual cold-gas protections off, to bisect "
                         "where the tangent is amplified")
    args = ap.parse_args()
    if args.theta is not None and len(args.theta) < len(PARAM_NAMES):
        args.theta = list(args.theta) + [PRIOR[k][0] for k in PARAM_NAMES[len(args.theta):]]

    ic = dict(np.load(args.ic))
    x64 = bool(jax.config.jax_enable_x64)
    if x64:
        # promote everything: float32 IC arrays would otherwise keep the whole
        # computation in float32 even with x64 enabled
        ic = {k: (np.asarray(v, np.float64) if np.asarray(v).dtype == np.float32 else v)
              for k, v in ic.items()}
    box, n = float(ic["box"]), int(ic["num_cells"])
    cu = snr_code_units()
    rho_c = float((1 * cu.code_density).to(u.g / u.cm ** 3).value)
    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3).to(cu.code_density).value)
    p_per_n = float((const.k_B * 1e4 * u.K / u.cm ** 3).to(cu.code_pressure).value)
    yr = float((1 * u.yr).to(cu.code_time).value)
    tpc = float((0.6 * const.m_p * cu.code_velocity ** 2 / const.k_B).to(u.K).value)
    theta = jnp.asarray(args.theta if args.theta is not None else THETA0,
                        dtype=jnp.float64 if x64 else jnp.float32)
    t = jnp.zeros_like(theta).at[PARAM_NAMES.index(args.param)].set(1.0)

    full = dict(coldcrush_blend="coldcrush_blend" not in args.pos_off, coldcrush_blend_factor=8.0)
    config = make_fd_config(box, n, dual_energy="dual_energy" not in args.pos_off,
                            progress_bar=False,
                            positivity_config=PositivityConfig(**full),
                            weno_ad_frozen_weights=True,
                            weno_positivity_preserving="pp" not in args.pos_off)
    rv = get_registered_variables(config)
    hd = get_helper_data(config)
    c = hd.geometric_centers
    X, Y, Z = c[..., 0] - box / 2, c[..., 1] - box / 2, c[..., 2] - box / 2
    r = jnp.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    s0, ds0 = jax.jvp(lambda th: make_initial_state(ic, th, config=config, rv=rv,
                                                     geom=(r, X, Y, Z), rho_c=rho_c),
                      (theta,), (t,))
    cfg = finalize_config(config, s0.shape)
    seg = SimulationParams(gamma=GAMMA, C_cfl=0.3, t_end=1.0,
                           minimum_density=0.1 * rho_per_n * 1e-3,
                           minimum_pressure=0.1 * p_per_n * 1e-2,
                           minimum_specific_pressure=p_per_n / rho_per_n)

    def advance(q, dt):
        q = time_integration(q, cfg, seg._replace(t_end=dt), rv)
        return tangent_filter(q, args.filter_sigma) if args.filter_sigma > 0 else q

    step = jax.jit(lambda s, ds, dt: jax.jvp(lambda q: advance(q, dt), (s,), (ds,)))
    ej = np.asarray(ic["C_ej"])            # ejecta tag at the start (region labels)
    rr = np.asarray(r)
    s, ds = s0, ds0
    age = float(ic["age"])
    print(f"[probe] d/d {args.param}, {n}^3, from {age:.1f} yr, strides of {args.stride} yr, "
          f"tangent filter sigma = {args.filter_sigma} cells, protections off: {args.pos_off}, "
          f"float{'64' if x64 else '32'}")
    for _ in range(args.strides):
        s1, ds1 = step(s, ds, args.stride * yr)
        age += args.stride
        a = np.abs(np.asarray(ds1[0]))
        nb = int((~np.isfinite(a)).sum())
        a = np.where(np.isfinite(a), a, 0.0)
        i = np.unravel_index(int(np.argmax(a)), a.shape)
        T = tpc * np.asarray(s1[rv.pressure_index]) / np.asarray(s1[0])
        hot = T > 1e7
        tot = a.sum() + 1e-300
        print(f"AGE {age:6.1f}: max|drho| {a.max():.3e} 99.9% {np.percentile(a, 99.9):.3e} "
              f"at r={rr[i]:.2f} T={T[i]:.1e} ej0={ej[i]:.2f}; |tangent| share: "
              f"hot ejecta {a[hot & (ej > 0.5)].sum() / tot:.2f}, hot CSM "
              f"{a[hot & (ej <= 0.5)].sum() / tot:.2f}, cold {a[~hot].sum() / tot:.2f}; "
              f"NaN {nb}", flush=True)
        if nb:
            break
        s, ds = s1, ds1


if __name__ == "__main__":
    main()
