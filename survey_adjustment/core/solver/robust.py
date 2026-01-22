"""Robust estimation module for least-squares adjustment.

Implements Iteratively Reweighted Least Squares (IRLS) with multiple
robust weight functions for outlier handling without data deletion.

Supported methods:
- Huber: soft downweighting, good for small to moderate outliers
- Danish: aggressive downweighting, good for larger outliers
- IGG-III: three-part function with hard rejection threshold

Reference: Ghilani & Wolf, "Adjustment Computations", 6th ed., Chapter 21
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np


class RobustMethod(Enum):
    """Robust estimation method enumeration."""
    NONE = "none"
    HUBER = "huber"
    DANISH = "danish"
    IGG3 = "igg3"

    @classmethod
    def from_string(cls, s: str) -> "RobustMethod":
        """Create RobustMethod from string (case-insensitive)."""
        s_lower = s.lower().strip()
        for method in cls:
            if method.value == s_lower:
                return method
        raise ValueError(f"Unknown robust method: {s}")


# ---------------------------------------------------------------------------
# Robust Weight Functions
# ---------------------------------------------------------------------------

def huber_weight(w: float, c: float = 1.5) -> float:
    """Huber weight function.

    Args:
        w: Standardized residual (absolute value will be used)
        c: Tuning constant (default 1.5)

    Returns:
        Weight factor u in (0, 1]

    Formula:
        u(|w|) = 1           if |w| <= c
        u(|w|) = c / |w|     if |w| > c
    """
    abs_w = abs(w)
    if abs_w <= c:
        return 1.0
    return c / abs_w


def danish_weight(w: float, c: float = 2.0) -> float:
    """Danish weight function.

    Args:
        w: Standardized residual (absolute value will be used)
        c: Tuning constant (default 2.0)

    Returns:
        Weight factor u in (0, 1]

    Formula:
        u(|w|) = 1                       if |w| <= c
        u(|w|) = exp(-((|w|-c)/c)^2)     if |w| > c

    Note: Danish method is more aggressive than Huber for large outliers.
    """
    abs_w = abs(w)
    if abs_w <= c:
        return 1.0
    return math.exp(-((abs_w - c) / c) ** 2)


def igg3_weight(w: float, k0: float = 1.5, k1: float = 3.0) -> float:
    """IGG-III (Institute of Geodesy and Geophysics) weight function.

    Args:
        w: Standardized residual (absolute value will be used)
        k0: Lower threshold (default 1.5)
        k1: Upper threshold (default 3.0)

    Returns:
        Weight factor u in [0, 1]

    Formula:
        u(|w|) = 1                                if |w| <= k0
        u(|w|) = (k0/|w|) * ((k1-|w|)/(k1-k0))^2 if k0 < |w| < k1
        u(|w|) = 0                                if |w| >= k1

    Note: IGG-III completely rejects observations with |w| >= k1.
    For numerical stability, we use a small epsilon instead of exactly 0.
    """
    abs_w = abs(w)
    if abs_w <= k0:
        return 1.0
    if abs_w >= k1:
        return 1e-10  # Near-zero but not exactly zero for numerical stability
    # Transition zone
    return (k0 / abs_w) * ((k1 - abs_w) / (k1 - k0)) ** 2


def get_weight_function(
    method: RobustMethod,
    huber_c: float = 1.5,
    danish_c: float = 2.0,
    igg3_k0: float = 1.5,
    igg3_k1: float = 3.0,
) -> Optional[Callable[[float], float]]:
    """Get the weight function for a robust method.

    Args:
        method: Robust estimation method
        huber_c: Huber tuning constant
        danish_c: Danish tuning constant
        igg3_k0: IGG-III lower threshold
        igg3_k1: IGG-III upper threshold

    Returns:
        Weight function or None if method is NONE
    """
    if method == RobustMethod.NONE:
        return None
    elif method == RobustMethod.HUBER:
        return lambda w: huber_weight(w, huber_c)
    elif method == RobustMethod.DANISH:
        return lambda w: danish_weight(w, danish_c)
    elif method == RobustMethod.IGG3:
        return lambda w: igg3_weight(w, igg3_k0, igg3_k1)
    else:
        return None


# ---------------------------------------------------------------------------
# IRLS Iteration Log
# ---------------------------------------------------------------------------

@dataclass
class IRLSIterationLog:
    """Log entry for a single IRLS iteration."""
    iteration: int
    max_weight_change: float
    num_downweighted: int
    min_weight: float
    variance_factor: float


@dataclass
class IRLSResult:
    """Result of IRLS convergence."""
    converged: bool
    iterations: int
    final_weights: np.ndarray
    iteration_log: List[IRLSIterationLog] = field(default_factory=list)
    message: str = ""


# ---------------------------------------------------------------------------
# IRLS Loop Helper
# ---------------------------------------------------------------------------

def compute_robust_weights(
    standardized_residuals: np.ndarray,
    weight_func: Callable[[float], float],
) -> np.ndarray:
    """Compute robust weight factors from standardized residuals.

    Args:
        standardized_residuals: Array of standardized residuals (w values)
        weight_func: Function that returns weight factor u for a given |w|

    Returns:
        Array of weight factors in (0, 1]
    """
    weights = np.ones(len(standardized_residuals))
    for i, w in enumerate(standardized_residuals):
        if w is not None and np.isfinite(w):
            weights[i] = weight_func(w)
        else:
            weights[i] = 1.0
    return weights


def irls_solve(
    solve_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, float]],
    num_observations: int,
    method: RobustMethod,
    max_iterations: int = 20,
    tolerance: float = 1e-3,
    huber_c: float = 1.5,
    danish_c: float = 2.0,
    igg3_k0: float = 1.5,
    igg3_k1: float = 3.0,
) -> IRLSResult:
    """Run IRLS loop for robust estimation.

    Args:
        solve_func: Function that takes weight factors and returns:
                   (standardized_residuals, residuals, variance_factor)
                   The function should apply the weights internally.
        num_observations: Number of observations (for initializing weights)
        method: Robust estimation method
        max_iterations: Maximum IRLS iterations
        tolerance: Convergence tolerance for max weight change
        huber_c: Huber tuning constant
        danish_c: Danish tuning constant
        igg3_k0: IGG-III lower threshold
        igg3_k1: IGG-III upper threshold

    Returns:
        IRLSResult with convergence info and final weights
    """
    if method == RobustMethod.NONE:
        # No robust estimation - return unit weights
        return IRLSResult(
            converged=True,
            iterations=0,
            final_weights=np.ones(num_observations),
            iteration_log=[],
            message="Robust estimation disabled (method=none)"
        )

    weight_func = get_weight_function(
        method, huber_c, danish_c, igg3_k0, igg3_k1
    )

    # Initialize weights to 1.0
    current_weights = np.ones(num_observations)
    iteration_log: List[IRLSIterationLog] = []

    for iteration in range(1, max_iterations + 1):
        # Solve with current weights
        std_residuals, residuals, variance_factor = solve_func(current_weights)

        # Compute new robust weights
        new_weights = compute_robust_weights(std_residuals, weight_func)

        # Check convergence
        weight_change = np.abs(new_weights - current_weights)
        max_change = np.max(weight_change)

        # Count downweighted observations
        num_downweighted = np.sum(new_weights < 0.999)
        min_weight = np.min(new_weights)

        # Log this iteration
        iteration_log.append(IRLSIterationLog(
            iteration=iteration,
            max_weight_change=float(max_change),
            num_downweighted=int(num_downweighted),
            min_weight=float(min_weight),
            variance_factor=float(variance_factor),
        ))

        # Update weights
        current_weights = new_weights

        # Check convergence
        if max_change < tolerance:
            return IRLSResult(
                converged=True,
                iterations=iteration,
                final_weights=current_weights,
                iteration_log=iteration_log,
                message=f"IRLS converged after {iteration} iterations (max weight change: {max_change:.6f})"
            )

    # Did not converge
    return IRLSResult(
        converged=False,
        iterations=max_iterations,
        final_weights=current_weights,
        iteration_log=iteration_log,
        message=f"IRLS did not converge after {max_iterations} iterations (max weight change: {max_change:.6f})"
    )


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def describe_method(method: RobustMethod, **params) -> str:
    """Get a human-readable description of the robust method and parameters.

    Args:
        method: Robust estimation method
        **params: Method parameters (huber_c, danish_c, igg3_k0, igg3_k1)

    Returns:
        Description string
    """
    if method == RobustMethod.NONE:
        return "None (standard least squares)"
    elif method == RobustMethod.HUBER:
        c = params.get("huber_c", 1.5)
        return f"Huber (c={c})"
    elif method == RobustMethod.DANISH:
        c = params.get("danish_c", 2.0)
        return f"Danish (c={c})"
    elif method == RobustMethod.IGG3:
        k0 = params.get("igg3_k0", 1.5)
        k1 = params.get("igg3_k1", 3.0)
        return f"IGG-III (k0={k0}, k1={k1})"
    return "Unknown"
