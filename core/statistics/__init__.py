"""Statistics utilities for survey adjustment.

This package contains small, dependency-light statistical helpers used by the
least-squares adjustment engine:
- Distribution functions (normal, chi-square)
- Global/local tests
- Reliability measures (redundancy, MDB, external reliability)

No SciPy dependency is required.
"""

from .distributions import normal_ppf, chi2_cdf, chi2_ppf
from .tests import chi_square_global_test, standardized_residuals, local_outlier_threshold
from .reliability import redundancy_numbers, mdb_values, external_reliability

__all__ = [
    "normal_ppf",
    "chi2_cdf",
    "chi2_ppf",
    "chi_square_global_test",
    "standardized_residuals",
    "local_outlier_threshold",
    "redundancy_numbers",
    "mdb_values",
    "external_reliability",
]
