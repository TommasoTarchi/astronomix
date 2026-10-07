"""
The validation cases of the VL2 scheme against AthenaPK.

The case table and the matching astronomix configurations live in the library,
``astronomix.test_setups.mhd.athenapk_vl2_cases``, so the regression pytest can
import them directly; this module re-exports them for the comparison scripts in
this directory.
"""

# astronomix constants
from astronomix.test_setups.mhd.athenapk_vl2_cases import (  # noqa: F401
    CASES,
    CASES_BY_NAME,
    REGRESSION_CASES,
)

# astronomix containers
from astronomix.test_setups.mhd.athenapk_vl2_cases import ValidationCase  # noqa: F401

# astronomix functions
from astronomix.test_setups.mhd.athenapk_vl2_cases import (  # noqa: F401
    astronomix_setup,
    initial_primitive_state,
)
