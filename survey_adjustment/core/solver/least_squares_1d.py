"""survey_adjustment.core.solver.least_squares_1d

1D least-squares adjustment for leveling networks (height differences).

The leveling adjustment is a linear problem (no iteration required):
- Observation equation: H_to - H_from = ΔH_obs
- Residual: v = ΔH_obs - (H_to - H_from)
- Unknown: heights of non-fixed points

This module intentionally contains **no QGIS imports** so it can be unit-
tested and re-used in non-QGIS contexts.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

from ..models.network import Network
from ..models.options import AdjustmentOptions
from ..models.point import Point
from ..models.observation import HeightDifferenceObservation
from ..results.adjustment_result import (
    AdjustmentResult,
    ResidualInfo,
)
from ..statistics import (
    chi_square_global_test,
    standardized_residuals,
    local_outlier_threshold,
    normal_ppf,
)
from ..statistics.reliability import (
    redundancy_numbers,
    mdb_values,
    external_reliability,
)


def adjust_leveling_1d(
    network: Network,
    options: AdjustmentOptions | None = None,
) -> AdjustmentResult:
    """Run a 1D least-squares adjustment for leveling networks.

    This is a linear adjustment (single iteration) for height differences.
    The observation equation is: H_to - H_from = ΔH_obs

    Args:
        network: Input network with points (having heights) and HeightDifferenceObservations
        options: Adjustment options (defaults if None)

    Returns:
        AdjustmentResult with adjusted heights, residuals, covariance, etc.
    """
    if np is None:
        return AdjustmentResult.failure("NumPy is required for the least-squares solver")

    options = options or AdjustmentOptions.default()

    # Validate network for 1D adjustment
    errors = network.validate_1d()
    if errors:
        return AdjustmentResult.failure("; ".join(errors))

    # Get enabled leveling observations
    leveling_obs: List[HeightDifferenceObservation] = network.get_leveling_observations()
    m = len(leveling_obs)

    if m == 0:
        return AdjustmentResult.failure("No enabled height difference observations")

    # Build parameter index: map point_id -> parameter index for unknown heights
    param_index: Dict[str, int] = {}
    height_order: List[str] = []

    # Get all points involved in leveling
    leveling_point_ids = set()
    for obs in leveling_obs:
        leveling_point_ids.add(obs.from_point_id)
        leveling_point_ids.add(obs.to_point_id)

    # Build index for unknown heights (non-fixed)
    for pid in sorted(leveling_point_ids):
        point = network.points[pid]
        if not point.fixed_height:
            param_index[pid] = len(height_order)
            height_order.append(pid)

    n = len(height_order)  # Number of unknowns

    if n == 0:
        return AdjustmentResult.failure("All points have fixed heights - nothing to adjust")

    # Build design matrix A, weight matrix P (diagonal), and misclosure vector l
    A = np.zeros((m, n), dtype=float)
    l = np.zeros(m, dtype=float)  # Misclosure (observed - approximate)
    sigmas = np.zeros(m, dtype=float)

    # Current heights (initial approximations)
    heights: Dict[str, float] = {}
    for pid in leveling_point_ids:
        point = network.points[pid]
        if point.height is None:
            return AdjustmentResult.failure(f"Point '{pid}' has no height value")
        heights[pid] = point.height

    for i, obs in enumerate(leveling_obs):
        from_pid = obs.from_point_id
        to_pid = obs.to_point_id

        # Design matrix row: H_to - H_from = ΔH
        # Coefficient for H_to is +1, for H_from is -1
        if to_pid in param_index:
            A[i, param_index[to_pid]] = 1.0
        if from_pid in param_index:
            A[i, param_index[from_pid]] = -1.0

        # Misclosure: l = ΔH_obs - (H_to - H_from)
        computed_dh = heights[to_pid] - heights[from_pid]
        l[i] = obs.value - computed_dh

        sigmas[i] = obs.sigma

    # Weight matrix (diagonal)
    Pdiag = 1.0 / (sigmas ** 2)

    # Normal equations: N = A^T P A, u = A^T P l
    Aw = A * Pdiag[:, None]
    N = A.T @ Aw
    u = A.T @ (Pdiag * l)

    # Solve for corrections
    try:
        dx = np.linalg.solve(N, u)
    except np.linalg.LinAlgError:
        return AdjustmentResult.failure("Normal matrix is singular (datum definition issue)")

    # Apply corrections to get adjusted heights
    adjusted_heights: Dict[str, float] = dict(heights)
    for pid, idx in param_index.items():
        adjusted_heights[pid] += dx[idx]

    # Compute residuals
    residuals = np.zeros(m, dtype=float)
    computed_values = np.zeros(m, dtype=float)

    for i, obs in enumerate(leveling_obs):
        from_pid = obs.from_point_id
        to_pid = obs.to_point_id
        computed_dh = adjusted_heights[to_pid] - adjusted_heights[from_pid]
        computed_values[i] = computed_dh
        residuals[i] = obs.value - computed_dh

    # Statistics
    dof = m - n
    vTPv = float((residuals * residuals * Pdiag).sum())

    if dof > 0:
        sigma0_sq = vTPv / dof
        variance_factor = sigma0_sq / options.a_priori_variance
    else:
        variance_factor = 1.0
        sigma0_sq = options.a_priori_variance

    sigma0_hat = math.sqrt(max(sigma0_sq, 1e-30))

    # Covariance matrix of unknowns
    Qxx: Optional[np.ndarray] = None
    cov_matrix: Optional[np.ndarray] = None
    sigma0_sq_post = variance_factor * options.a_priori_variance

    if options.compute_covariances:
        try:
            Qxx = np.linalg.solve(N, np.eye(n))
            cov_matrix = sigma0_sq_post * Qxx
        except np.linalg.LinAlgError:
            Qxx = None
            cov_matrix = None

    # Build adjusted points with posterior sigmas
    adjusted_points: Dict[str, Point] = {}

    for pid, p in network.points.items():
        # Keep original E, N coordinates
        new_height = adjusted_heights.get(pid, p.height)
        sigma_h = None

        if pid in param_index and cov_matrix is not None:
            idx = param_index[pid]
            sigma_h = math.sqrt(max(cov_matrix[idx, idx], 0.0))
        elif p.fixed_height:
            sigma_h = 0.0

        adjusted_points[pid] = Point(
            id=p.id,
            name=p.name,
            easting=p.easting,
            northing=p.northing,
            height=new_height,
            fixed_easting=p.fixed_easting,
            fixed_northing=p.fixed_northing,
            fixed_height=p.fixed_height,
            sigma_easting=p.sigma_easting,
            sigma_northing=p.sigma_northing,
            sigma_height=sigma_h,
        )

    # Reliability measures
    qvv_diag: Optional[np.ndarray] = None
    std_vals: np.ndarray

    sigma0_for_w = math.sqrt(max(options.a_priori_variance, 1e-30))

    if dof > 0 and Qxx is not None:
        # diag(Qvv) = diag(P^-1) - diag(A Qxx A^T)
        B = A @ Qxx
        diag_AQAt = (B * A).sum(axis=1)
        qvv_diag = (1.0 / Pdiag) - diag_AQAt
        qvv_diag = np.maximum(qvv_diag, 1e-30)
        std_vals = standardized_residuals(residuals, sigma0_for_w, qvv_diag)
    else:
        # Fallback: normalize by observation sigma only
        std_vals = residuals / (sigmas * max(sigma0_hat, 1e-12))

    # Local test threshold
    k_alpha = local_outlier_threshold(options.alpha_local)
    k_beta = normal_ppf(options.mdb_power)

    r_vals: Optional[np.ndarray] = None
    mdb: Optional[np.ndarray] = None
    ext_rel: Optional[np.ndarray] = None

    if options.compute_reliability and Qxx is not None and qvv_diag is not None and dof > 0:
        r_vals = redundancy_numbers(qvv_diag, Pdiag)
        mdb = mdb_values(k_alpha, k_beta, sigma0_hat, sigmas, r_vals)
        # External reliability for height parameters (all parameters are heights)
        coord_param_indices = list(range(n))
        ext_rel = external_reliability(Qxx, A, Pdiag, mdb, coord_param_indices)

    # Build residual details
    std_residuals: Dict[str, float] = {}
    residual_details: List[ResidualInfo] = []
    flagged: List[str] = []

    for j, obs in enumerate(leveling_obs):
        is_candidate = abs(float(std_vals[j])) > float(k_alpha)
        is_flagged = abs(float(std_vals[j])) > float(options.outlier_threshold)

        info = ResidualInfo(
            obs_id=obs.id,
            obs_type="height_diff",
            observed=float(obs.value),
            computed=float(computed_values[j]),
            residual=float(residuals[j]),
            standardized_residual=float(std_vals[j]),
            redundancy_number=float(r_vals[j]) if r_vals is not None else None,
            mdb=float(mdb[j]) if mdb is not None else None,
            external_reliability=float(ext_rel[j]) if ext_rel is not None else None,
            is_outlier_candidate=is_candidate,
            flagged=is_flagged,
            from_point=obs.from_point_id,
            to_point=obs.to_point_id,
        )

        std_residuals[obs.id] = float(std_vals[j])
        residual_details.append(info)
        if is_flagged:
            flagged.append(obs.id)

    # Chi-square global test
    chi_test = None
    if dof > 0:
        chi_test = chi_square_global_test(
            vTPv=vTPv,
            dof=dof,
            alpha=options.confidence_level,
            a_priori_variance=options.a_priori_variance,
        )

    # Build point covariances dict (1x1 for heights)
    point_covs: Dict[str, np.ndarray] = {}
    if cov_matrix is not None:
        for pid, idx in param_index.items():
            point_covs[pid] = np.array([[cov_matrix[idx, idx]]])

    return AdjustmentResult(
        success=True,
        iterations=1,  # Linear problem: single iteration
        converged=True,
        adjusted_points=adjusted_points,
        residuals={obs.id: float(residuals[j]) for j, obs in enumerate(leveling_obs)},
        standardized_residuals=std_residuals,
        residual_details=residual_details,
        degrees_of_freedom=dof,
        variance_factor=variance_factor,
        chi_square_test=chi_test,
        covariance_matrix=cov_matrix.tolist() if cov_matrix is not None else None,
        point_covariances=point_covs,
        error_ellipses={},  # Not applicable for 1D
        flagged_observations=flagged,
        network_name=network.name,
    )
