"""
Data side of the Cas A 4D-Var (``casa_4dvar``): the casa_xfit data dicts for
all epochs, their restriction to an epoch subset (training window, held-out
epochs), and the held-out (validation) residuals.

The held-out epochs are scored AGAINST THE TRAINING SET: an image block's
brightness relative to the block's training-epoch mean (``casa_xfit.
image_residuals``' temporal statistic with the training statics), a spectrum's
shape relative to the training-mean shape, and the outline radius per cone
relative to the training epochs' mean offset -- i.e. a prediction test of the
evolution, not a refit of the static structure.
"""
import time

import numpy as np

import casa_pluto_diff as PD
import casa_xfit as X
import casa_xfit_obs2 as O2

#: per-epoch keys, the proper-motion refit and the epoch subset live in
#: casa_xfit (its --exclude-epochs uses them); re-exported unchanged
OBS_EPOCH_KEYS, IMG_EPOCH_KEYS, SPEC_EPOCH_KEYS = X.OBS_EPOCH_KEYS, X.IMG_EPOCH_KEYS, X.SPEC_EPOCH_KEYS
refit_pm = X.refit_pm
subset = X.subset_epochs
heldout_parts = X.heldout_parts
#: the multi-scale image likelihood (casa_xfit.fine_image_residuals; default off).
#: casa_4dvar enables it with three hooks: ``DA.add_fine_arguments(ap)`` in its
#: argparse, ``DA.load_all(opts, ..., fine_block=a.fine_block)``, and
#: ``**DA.fine_likelihood_args(a)`` in its ``likelihood_args`` namespace; then
#: ``X.residual_parts`` returns ``image_fine`` (+ ``image_struct``) as well.
add_fine_arguments = X.add_fine_arguments
FINE_ARG_NAMES = ("fine_block", "sigma_fine", "fine_weight", "struct_sectors", "sigma_struct")


def fine_likelihood_args(a):
    """The multi-scale likelihood settings of an argparse namespace (``add_fine_arguments``)
    as a dict for casa_xfit's likelihood args (fine_block 0 = off if absent)."""
    return {k: getattr(a, k, 0 if k in ("fine_block", "struct_sectors") else None) for k in FINE_ARG_NAMES}


def load_all(opts, *, block=16, r_max=150.0, table_dir=None, pm_exclude=None, fine_block=0):
    """(obs, img) for every usable epoch, the casa_xfit ``main`` recipe
    (``pm_exclude``: epochs dropped from the proper-motion slopes, ``refit_pm``;
    ``fine_block``: also the fine-block counts of the multi-scale image term)."""
    t0 = time.time()
    obs = PD.load_observations(pm_files=opts.pm_files, pm_mask=opts.pm_mask)
    # the stage-4 cone masks (--outline-mask, --pm-mask-extra; refit_pm keeps the latter)
    X.apply_obs_masks(obs, opts)
    if pm_exclude:
        n0 = int(np.isfinite(obs["pm"]).sum())
        obs = refit_pm(obs, opts.pm_files, opts.pm_mask, set(pm_exclude))
        print(f"[4dvar] proper motions refitted without {sorted(pm_exclude)}: {n0} -> "
              f"{int(np.isfinite(obs['pm']).sum())} cones", flush=True)
    if opts.obs == "v2":
        obs["doppler"] = O2.load_doppler_data()
    kw = {} if table_dir is None else dict(table_dir=table_dir)
    img = X.load_image_data(obs["epochs"], obs["years"], block=block, r_max=r_max, obs=opts.obs,
                            history=getattr(opts, "history", True), spec_kinds=X.spec_table_kinds(opts),
                            fine_block=fine_block, **kw)
    img["spec"] = X.load_spectrum_data(obs["epochs"], img)
    X.attach_responses(obs, img, opts)
    # the stage-4 measured background (--background; nothing for 'rate')
    X.attach_stage4(obs, img, opts)
    print(f"[4dvar] data: {len(obs['epochs'])} epochs {obs['epochs'][0]}-{obs['epochs'][-1]} "
          f"({time.time() - t0:.0f} s)", flush=True)
    return obs, img


def split_epochs(obs, *, t_end, holdout=("2019", "2022"), t_start=None):
    """(train_idx, hold_idx): training = epochs <= ``t_end`` (years) not held out."""
    yrs = np.asarray(obs["years"])
    hold = np.array([e in holdout for e in obs["epochs"]])
    ok = (yrs <= t_end + 1e-6) & ~hold
    if t_start is not None:
        ok &= yrs >= t_start - 1e-6
    return np.nonzero(ok)[0], np.nonzero(hold)[0]
