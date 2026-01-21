"""survey_adjustment.core.statistics.distributions

Distribution helpers (no SciPy).

Implemented:
- Standard normal PPF via stdlib ``statistics.NormalDist``
- Chi-square CDF/PPF via regularized incomplete gamma + safeguarded Newton

These routines are used for statistical tests in least-squares adjustment.

Chi-square:
  If X ~ ChiSquare(df), then X = 2 * Gamma(a=df/2, scale=1).
  CDF is regularized lower incomplete gamma P(a, x/2).

References (algorithms):
- Numerical Recipes / Cephes style implementations for incomplete gamma.
- Wilson-Hilferty transformation for initial chi-square quantile guess.

The goal here is stable behavior for typical survey-network degrees of freedom
(from ~1 up to a few thousand) without adding heavy dependencies.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Tuple


# ----------------------------
# Normal
# ----------------------------

_NORMAL = NormalDist()


def normal_ppf(p: float) -> float:
    """Standard normal quantile (inverse CDF).

    Args:
        p: probability in (0, 1)

    Returns:
        z such that P(Z <= z) = p
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    return float(_NORMAL.inv_cdf(p))


# ----------------------------
# Incomplete gamma (regularized)
# ----------------------------

_DEF_EPS = 1e-14
_DEF_MAX_IT = 2000


def _gammainc_lower_reg(a: float, x: float, eps: float = _DEF_EPS, max_it: int = _DEF_MAX_IT) -> float:
    """Regularized lower incomplete gamma P(a, x).

    Computes:
      P(a,x) = 1/Gamma(a) * integral_0^x t^{a-1} e^{-t} dt

    Uses:
      - series expansion for x < a+1
      - continued fraction for x >= a+1

    Args:
        a: shape parameter (>0)
        x: integration limit (>=0)

    Returns:
        P(a, x) in [0, 1]
    """
    if a <= 0.0:
        raise ValueError("a must be positive")
    if x <= 0.0:
        return 0.0

    # Compute common factor: e^{-x} x^a / Gamma(a)
    # Use logs for stability.
    gln = math.lgamma(a)

    if x < a + 1.0:
        # Series representation
        ap = a
        summ = 1.0 / a
        delt = summ
        for _ in range(max_it):
            ap += 1.0
            delt *= x / ap
            summ += delt
            if abs(delt) < abs(summ) * eps:
                break
        # P(a,x)
        return summ * math.exp(-x + a * math.log(x) - gln)

    # Continued fraction for Q(a,x) = 1 - P(a,x)
    # Modified Lentz's method
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / max(b, tiny)
    h = d

    for i in range(1, max_it + 1):
        an = -float(i) * (float(i) - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    q = h * math.exp(-x + a * math.log(x) - gln)
    p = 1.0 - q

    # Clip due to rounding
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


# ----------------------------
# Chi-square
# ----------------------------


def chi2_cdf(x: float, df: int) -> float:
    """CDF of chi-square distribution.

    Args:
        x: value (>=0)
        df: degrees of freedom (>0)

    Returns:
        P(X <= x)
    """
    if df <= 0:
        raise ValueError("df must be positive")
    if x <= 0.0:
        return 0.0
    a = 0.5 * float(df)
    return _gammainc_lower_reg(a, 0.5 * float(x))


def _chi2_pdf(x: float, df: int) -> float:
    """PDF of chi-square distribution."""
    if x <= 0.0:
        return 0.0
    k = 0.5 * float(df)
    # log(pdf) = (k-1)log(x) - x/2 - k log(2) - lgamma(k)
    log_pdf = (k - 1.0) * math.log(x) - 0.5 * x - k * math.log(2.0) - math.lgamma(k)
    return math.exp(log_pdf)


def chi2_ppf(p: float, df: int) -> float:
    """Quantile (inverse CDF) of chi-square distribution.

    Uses Wilson-Hilferty for an initial guess then a safeguarded Newton method
    that maintains a bracket.

    Args:
        p: probability in (0,1)
        df: degrees of freedom (>0)

    Returns:
        x such that chi2_cdf(x, df) = p
    """
    if df <= 0:
        raise ValueError("df must be positive")
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")

    # Initial guess via Wilson-Hilferty
    k = float(df)
    z = normal_ppf(p)
    t = 1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))
    x = k * max(t, 1e-12) ** 3

    # Bracket
    lo = 0.0
    hi = max(x, 1e-12)
    # Expand hi until CDF(hi) >= p
    for _ in range(200):
        if chi2_cdf(hi, df) >= p:
            break
        hi *= 2.0
    else:
        # If we didn't break, something is off; fall back to hi
        return float(hi)

    # Ensure x within bracket
    x = min(max(x, lo + 1e-15), hi - 1e-15)

    tol = 1e-12
    for _ in range(100):
        cdf = chi2_cdf(x, df)
        if cdf < p:
            lo = x
        else:
            hi = x

        pdf = _chi2_pdf(x, df)
        if pdf > 0.0:
            step = (cdf - p) / pdf
            x_new = x - step
        else:
            x_new = float('nan')

        # Safeguard: keep inside bracket
        if (not math.isfinite(x_new)) or x_new <= lo or x_new >= hi:
            x_new = 0.5 * (lo + hi)

        if abs(x_new - x) <= tol * max(1.0, x):
            return float(x_new)
        x = x_new

    return float(x)


def chi2_critical_interval(df: int, alpha: float) -> Tuple[float, float]:
    """Two-sided chi-square critical interval.

    Returns (lower, upper) such that P(lower <= X <= upper) = 1 - alpha.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    lower = chi2_ppf(alpha / 2.0, df)
    upper = chi2_ppf(1.0 - alpha / 2.0, df)
    return float(lower), float(upper)
