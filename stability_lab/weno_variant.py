"""Map the WENO_VARIANT environment variable onto SimulationConfig options.

    baseline  current scheme
    face      weno_admissible_face_state
    pp        weno_positivity_preserving (implies the face state)
"""

import os


def weno_variant_kwargs():
    variant = os.environ.get("WENO_VARIANT", "baseline")
    if variant == "baseline":
        return {}
    if variant == "face":
        return dict(weno_admissible_face_state=True)
    if variant == "pp":
        return dict(weno_positivity_preserving=True)
    raise ValueError(f"unknown WENO_VARIANT {variant!r}")


def weno_variant_name():
    return os.environ.get("WENO_VARIANT", "baseline")
