"""survey_adjustment.core.statistics.tests

Statistical tests and residual standardization used in least-squares adjustment.

Includes:
- Global chi-square test for overall model consistency
- Standardized residual computation
- Helper to compute local outlier thresholds
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

from .distributions import chi2_cdf, chi2_ppf, normal_ppf
from ..results.adjustment_result import ChiSquareTestResult


def chi_square_global_test(
    vTPv: float,
    dof: int,
    alpha: float,
    a_priori_variance: float = 1.0,
) -> ChiSquareTestResult:
    """Run the global chi-square test.

    Test statistic:
        T = v^T P v / sigma0_apriori^2

    Decision (two-sided):
        chi2_{alpha/2, dof} <= T <= chi2_{1-alpha/2, dof}

    Args:
        vTPv: residual sum of squares with weights (v^T P v)
        dof: degrees of freedom
        alpha: significance level
        a_priori_variance: a priori variance of unit weight (sigma0^2)

    Returns:
        ChiSquareTestResult (with p-value and pass/fail)
    """
    if dof <= 0:
        raise ValueError("dof must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if a_priori_variance <= 0:
        raise ValueError("a_priori_variance must be positive")

    test_stat = float(vTPv) / float(a_priori_variance)

    lower = chi2_ppf(alpha / 2.0, dof)
    upper = chi2_ppf(1.0 - alpha / 2.0, dof)

    # Two-sided p-value around the center of the distribution
    cdf = chi2_cdf(test_stat, dof)
    p_value = float(2.0 * min(cdf, 1.0 - cdf))

    conf = 1.0 - alpha
    passed = bool(lower <= test_stat <= upper)

    return ChiSquareTestResult(
        test_statistic=test_stat,
        critical_lower=lower,
        critical_upper=upper,
        confidence_level=conf,
        passed=passed,
        p_value=p_value,
        degrees_of_freedom=int(dof),
    )


def standardized_residuals(
    residuals: np.ndarray,
    sigma0_hat: float,
    qvv_diag: np.ndarray,
) -> np.ndarray:
    """Compute standardized residuals.

    w_i = v_i / (sigma0_hat * sqrt(qvv_ii))

    Args:
        residuals: residual vector v (length m)
        sigma0_hat: sqrt(v^T P v / dof)
        qvv_diag: diagonal of Qvv (cofactor of residuals)

    Returns:
        vector w of standardized residuals
    """
    if sigma0_hat <= 0:
        raise ValueError("sigma0_hat must be positive")
    denom = sigma0_hat * np.sqrt(np.maximum(qvv_diag, 1e-30))
    return residuals / denom


def local_outlier_threshold(alpha_local: float) -> float:
    """Two-sided local outlier threshold for standardized residuals.

    k = Phi^{-1}(1 - alpha_local/2)

    Args:
        alpha_local: local significance level

    Returns:
        threshold k
    """
    if not (0.0 < alpha_local < 1.0):
        raise ValueError("alpha_local must be in (0,1)")
    return float(normal_ppf(1.0 - alpha_local / 2.0))
