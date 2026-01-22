"""survey_adjustment.core.solver.least_squares_3d

3D least-squares adjustment for GNSS baseline networks.

The GNSS baseline adjustment is a linear problem (no iteration required):
- Observation equations (per baseline):
    dE_obs = E_to - E_from
    dN_obs = N_to - N_from
    dH_obs = H_to - H_from
- Residual: v = observed - computed (3x1 vector per baseline)
- Unknown: E, N, H of non-fixed points

Each baseline contributes 3 equations with a 3x3 covariance matrix.
Weight matrix P = C^-1 for each baseline block.

This module intentionally contains **no QGIS imports** so it can be unit-
tested and re-used in non-QGIS contexts.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Set

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

from ..models.network import Network
from ..models.options import AdjustmentOptions
from ..models.point import Point
from ..models.observation import GnssBaselineObservation
from ..results.adjustment_result import (
    AdjustmentResult,
    ResidualInfo,
    ErrorEllipse,
)
from ..statistics import (
    chi_square_global_test,
    local_outlier_threshold,
    normal_ppf,
)
from ..statistics.reliability import (
    redundancy_numbers,
    mdb_values,
)


def _cholesky_lower(matrix_3x3: np.ndarray) -> np.ndarray:
    """Compute lower Cholesky factorization of a 3x3 positive definite matrix.

    Returns L such that matrix = L @ L.T
    """
    return np.linalg.cholesky(matrix_3x3)


def _invert_3x3(matrix: np.ndarray) -> np.ndarray:
    """Invert a 3x3 matrix."""
    return np.linalg.inv(matrix)


def adjust_gnss_3d(
    network: Network,
    options: AdjustmentOptions | None = None,
) -> AdjustmentResult:
    """Run a 3D least-squares adjustment for GNSS baseline networks.

    This is a linear adjustment (single iteration) for GNSS baselines.
    Each baseline provides 3 observation equations for dE, dN, dH.

    Args:
        network: Input network with points (E, N, H) and GnssBaselineObservations
        options: Adjustment options (defaults if None)

    Returns:
        AdjustmentResult with adjusted coordinates, residuals, covariance, etc.
    """
    if np is None:
        return AdjustmentResult.failure("NumPy is required for the least-squares solver")

    options = options or AdjustmentOptions.default()

    # Validate network for 3D adjustment
    errors = network.validate_3d()
    if errors:
        return AdjustmentResult.failure("; ".join(errors))

    # Get enabled GNSS baseline observations
    gnss_obs: List[GnssBaselineObservation] = network.get_gnss_baseline_observations()
    num_baselines = len(gnss_obs)

    if num_baselines == 0:
        return AdjustmentResult.failure("No enabled GNSS baseline observations")

    m = 3 * num_baselines  # Total number of observation equations

    # Build parameter index: map (point_id, component) -> parameter index
    # component: 0=E, 1=N, 2=H
    param_index: Dict[Tuple[str, int], int] = {}
    param_order: List[Tuple[str, int]] = []  # (point_id, component)

    # Get all points involved in GNSS observations
    gnss_point_ids: Set[str] = set()
    for obs in gnss_obs:
        gnss_point_ids.add(obs.from_point_id)
        gnss_point_ids.add(obs.to_point_id)

    # Build index for unknown coordinates (non-fixed)
    for pid in sorted(gnss_point_ids):
        point = network.points[pid]
        if not point.fixed_easting:
            param_index[(pid, 0)] = len(param_order)
            param_order.append((pid, 0))
        if not point.fixed_northing:
            param_index[(pid, 1)] = len(param_order)
            param_order.append((pid, 1))
        if not point.fixed_height:
            param_index[(pid, 2)] = len(param_order)
            param_order.append((pid, 2))

    n = len(param_order)  # Number of unknowns

    if n == 0:
        return AdjustmentResult.failure("All coordinates are fixed - nothing to adjust")

    # Current coordinates (initial approximations)
    coords: Dict[str, Tuple[float, float, float]] = {}
    for pid in gnss_point_ids:
        point = network.points[pid]
        if point.height is None:
            return AdjustmentResult.failure(f"Point '{pid}' has no height value")
        coords[pid] = (point.easting, point.northing, point.height)

    # Build design matrix A, weight blocks, and misclosure vector l
    # Using whitening approach: pre-multiply by L^T where P = L^T @ L (Cholesky of weight)
    A = np.zeros((m, n), dtype=float)
    l = np.zeros(m, dtype=float)

    # Store covariance blocks for later use (for qvv computation)
    cov_blocks: List[np.ndarray] = []
    weight_blocks: List[np.ndarray] = []

    for i, obs in enumerate(gnss_obs):
        from_pid = obs.from_point_id
        to_pid = obs.to_point_id
        row_base = 3 * i

        # Covariance matrix for this baseline
        C = np.array(obs.covariance_matrix, dtype=float)
        cov_blocks.append(C)

        # Weight matrix P = C^-1
        try:
            P = _invert_3x3(C)
        except np.linalg.LinAlgError:
            return AdjustmentResult.failure(
                f"Baseline {obs.id}: covariance matrix is singular"
            )
        weight_blocks.append(P)

        # Design matrix rows for this baseline
        # dE = E_to - E_from => coefficient +1 for E_to, -1 for E_from
        # dN = N_to - N_from => coefficient +1 for N_to, -1 for N_from
        # dH = H_to - H_from => coefficient +1 for H_to, -1 for H_from

        # E component (row_base + 0)
        if (to_pid, 0) in param_index:
            A[row_base + 0, param_index[(to_pid, 0)]] = 1.0
        if (from_pid, 0) in param_index:
            A[row_base + 0, param_index[(from_pid, 0)]] = -1.0

        # N component (row_base + 1)
        if (to_pid, 1) in param_index:
            A[row_base + 1, param_index[(to_pid, 1)]] = 1.0
        if (from_pid, 1) in param_index:
            A[row_base + 1, param_index[(from_pid, 1)]] = -1.0

        # H component (row_base + 2)
        if (to_pid, 2) in param_index:
            A[row_base + 2, param_index[(to_pid, 2)]] = 1.0
        if (from_pid, 2) in param_index:
            A[row_base + 2, param_index[(from_pid, 2)]] = -1.0

        # Misclosure: l = observed - computed
        from_E, from_N, from_H = coords[from_pid]
        to_E, to_N, to_H = coords[to_pid]

        computed_dE = to_E - from_E
        computed_dN = to_N - from_N
        computed_dH = to_H - from_H

        l[row_base + 0] = obs.dE - computed_dE
        l[row_base + 1] = obs.dN - computed_dN
        l[row_base + 2] = obs.dH - computed_dH

    # Build full weight matrix P (block diagonal)
    P_full = np.zeros((m, m), dtype=float)
    for i, P_block in enumerate(weight_blocks):
        row_base = 3 * i
        P_full[row_base:row_base+3, row_base:row_base+3] = P_block

    # Normal equations: N = A^T P A, u = A^T P l
    AtP = A.T @ P_full
    N = AtP @ A
    u = AtP @ l

    # Solve for corrections
    try:
        dx = np.linalg.solve(N, u)
    except np.linalg.LinAlgError:
        return AdjustmentResult.failure("Normal matrix is singular (datum definition issue)")

    # Apply corrections to get adjusted coordinates
    adjusted_coords: Dict[str, List[float]] = {}
    for pid in gnss_point_ids:
        e, n_coord, h = coords[pid]
        adjusted_coords[pid] = [e, n_coord, h]

    for (pid, comp), idx in param_index.items():
        adjusted_coords[pid][comp] += dx[idx]

    # Compute residuals
    residuals = np.zeros(m, dtype=float)
    computed_values = np.zeros((num_baselines, 3), dtype=float)

    for i, obs in enumerate(gnss_obs):
        from_pid = obs.from_point_id
        to_pid = obs.to_point_id
        row_base = 3 * i

        from_coords = adjusted_coords[from_pid]
        to_coords = adjusted_coords[to_pid]

        computed_dE = to_coords[0] - from_coords[0]
        computed_dN = to_coords[1] - from_coords[1]
        computed_dH = to_coords[2] - from_coords[2]

        computed_values[i] = [computed_dE, computed_dN, computed_dH]

        residuals[row_base + 0] = obs.dE - computed_dE
        residuals[row_base + 1] = obs.dN - computed_dN
        residuals[row_base + 2] = obs.dH - computed_dH

    # Statistics
    dof = m - n
    vTPv = float(residuals @ P_full @ residuals)

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
        if pid in adjusted_coords:
            new_e, new_n, new_h = adjusted_coords[pid]
        else:
            new_e, new_n, new_h = p.easting, p.northing, p.height

        sigma_e = None
        sigma_n = None
        sigma_h = None

        if cov_matrix is not None:
            if (pid, 0) in param_index:
                idx = param_index[(pid, 0)]
                sigma_e = math.sqrt(max(cov_matrix[idx, idx], 0.0))
            elif p.fixed_easting:
                sigma_e = 0.0

            if (pid, 1) in param_index:
                idx = param_index[(pid, 1)]
                sigma_n = math.sqrt(max(cov_matrix[idx, idx], 0.0))
            elif p.fixed_northing:
                sigma_n = 0.0

            if (pid, 2) in param_index:
                idx = param_index[(pid, 2)]
                sigma_h = math.sqrt(max(cov_matrix[idx, idx], 0.0))
            elif p.fixed_height:
                sigma_h = 0.0

        adjusted_points[pid] = Point(
            id=p.id,
            name=p.name,
            easting=new_e,
            northing=new_n,
            height=new_h,
            fixed_easting=p.fixed_easting,
            fixed_northing=p.fixed_northing,
            fixed_height=p.fixed_height,
            sigma_easting=sigma_e,
            sigma_northing=sigma_n,
            sigma_height=sigma_h,
        )

    # Compute qvv diagonal for standardized residuals
    # diag(Qvv) = diag(P^-1) - diag(A Qxx A^T)
    qvv_diag: Optional[np.ndarray] = None
    std_vals = np.zeros(m, dtype=float)

    sigma0_for_w = math.sqrt(max(options.a_priori_variance, 1e-30))

    if dof > 0 and Qxx is not None:
        # Build Qll diagonal (covariance diagonal of observations)
        Qll_diag = np.zeros(m, dtype=float)
        for i, C in enumerate(cov_blocks):
            row_base = 3 * i
            Qll_diag[row_base + 0] = C[0, 0]
            Qll_diag[row_base + 1] = C[1, 1]
            Qll_diag[row_base + 2] = C[2, 2]

        # Compute diag(A Qxx A^T) efficiently
        B = A @ Qxx
        diag_AQAt = (B * A).sum(axis=1)

        qvv_diag = Qll_diag - diag_AQAt
        qvv_diag = np.maximum(qvv_diag, 1e-30)

        # Standardized residuals
        std_vals = residuals / np.sqrt(qvv_diag * sigma0_sq_post)
    else:
        # Fallback: normalize by observation sigma only
        for i, C in enumerate(cov_blocks):
            row_base = 3 * i
            std_vals[row_base + 0] = residuals[row_base + 0] / math.sqrt(C[0, 0])
            std_vals[row_base + 1] = residuals[row_base + 1] / math.sqrt(C[1, 1])
            std_vals[row_base + 2] = residuals[row_base + 2] / math.sqrt(C[2, 2])

    # Local test threshold
    k_alpha = local_outlier_threshold(options.alpha_local)
    k_beta = normal_ppf(options.mdb_power)

    # Build residual details - one per baseline with component info
    std_residuals: Dict[str, float] = {}
    residual_details: List[ResidualInfo] = []
    flagged: List[str] = []

    for i, obs in enumerate(gnss_obs):
        row_base = 3 * i

        # Residual components
        vE = residuals[row_base + 0]
        vN = residuals[row_base + 1]
        vH = residuals[row_base + 2]

        # Standardized residual components
        wE = std_vals[row_base + 0]
        wN = std_vals[row_base + 1]
        wH = std_vals[row_base + 2]

        # Maximum standardized residual (for outlier detection)
        w_max = max(abs(wE), abs(wN), abs(wH))

        # Redundancy numbers for this baseline
        r_E = r_N = r_H = None
        if qvv_diag is not None:
            P_block = weight_blocks[i]
            r_E = qvv_diag[row_base + 0] * P_block[0, 0]
            r_N = qvv_diag[row_base + 1] * P_block[1, 1]
            r_H = qvv_diag[row_base + 2] * P_block[2, 2]

        # Average redundancy for baseline
        r_avg = None
        if r_E is not None and r_N is not None and r_H is not None:
            r_avg = (r_E + r_N + r_H) / 3.0

        is_candidate = bool(w_max > k_alpha)
        is_flagged = bool(w_max > options.outlier_threshold)

        # Create residual info with baseline-level summary
        # Store component residuals in a format that can be parsed
        info = ResidualInfo(
            obs_id=obs.id,
            obs_type="gnss_baseline",
            observed=obs.baseline_length,  # Use length as summary
            computed=math.sqrt(computed_values[i, 0]**2 + computed_values[i, 1]**2 + computed_values[i, 2]**2),
            residual=math.sqrt(vE**2 + vN**2 + vH**2),  # 3D residual magnitude
            standardized_residual=w_max,  # Maximum standardized residual
            redundancy_number=r_avg,
            mdb=None,  # MDB computation is complex for correlated obs
            external_reliability=None,
            is_outlier_candidate=is_candidate,
            flagged=is_flagged,
            from_point=obs.from_point_id,
            to_point=obs.to_point_id,
        )

        # Store additional component info as custom attributes
        info._vE = vE
        info._vN = vN
        info._vH = vH
        info._wE = wE
        info._wN = wN
        info._wH = wH

        std_residuals[obs.id] = w_max
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

    # Build point covariances dict (3x3 for E,N,H if available)
    point_covs: Dict[str, np.ndarray] = {}
    if cov_matrix is not None:
        for pid in gnss_point_ids:
            # Build 3x3 covariance matrix for this point
            cov_3x3 = np.zeros((3, 3), dtype=float)
            indices = []
            for comp in range(3):
                if (pid, comp) in param_index:
                    indices.append((comp, param_index[(pid, comp)]))
                else:
                    indices.append((comp, None))

            for ci, idx_i in indices:
                for cj, idx_j in indices:
                    if idx_i is not None and idx_j is not None:
                        cov_3x3[ci, cj] = cov_matrix[idx_i, idx_j]

            if np.any(cov_3x3 != 0):
                point_covs[pid] = cov_3x3

    # Compute error ellipses for horizontal coordinates
    error_ellipses: Dict[str, ErrorEllipse] = {}
    if cov_matrix is not None:
        for pid in gnss_point_ids:
            # Get 2x2 horizontal covariance (E, N)
            idx_E = param_index.get((pid, 0))
            idx_N = param_index.get((pid, 1))

            if idx_E is not None and idx_N is not None:
                cov_2x2 = np.array([
                    [cov_matrix[idx_E, idx_E], cov_matrix[idx_E, idx_N]],
                    [cov_matrix[idx_N, idx_E], cov_matrix[idx_N, idx_N]],
                ])

                # Compute eigenvalues and eigenvectors for error ellipse
                try:
                    eigvals, eigvecs = np.linalg.eigh(cov_2x2)
                    # Scale by chi-square factor for confidence level
                    # For 2 DOF: chi2(0.95, 2) ≈ 5.991
                    from ..statistics.distributions import chi2_ppf
                    scale = math.sqrt(chi2_ppf(options.confidence_level, 2))

                    semi_major = scale * math.sqrt(max(eigvals[1], 0))
                    semi_minor = scale * math.sqrt(max(eigvals[0], 0))

                    # Orientation: angle of major axis from East (x) axis
                    orientation = math.atan2(eigvecs[1, 1], eigvecs[0, 1])

                    error_ellipses[pid] = ErrorEllipse(
                        point_id=pid,
                        semi_major=semi_major,
                        semi_minor=semi_minor,
                        orientation=orientation,
                        confidence_level=options.confidence_level,
                    )
                except np.linalg.LinAlgError:
                    pass  # Skip if eigenvalue computation fails

    return AdjustmentResult(
        success=True,
        iterations=1,  # Linear problem: single iteration
        converged=True,
        adjusted_points=adjusted_points,
        residuals={obs.id: math.sqrt(sum(residuals[3*i + j]**2 for j in range(3)))
                   for i, obs in enumerate(gnss_obs)},
        standardized_residuals=std_residuals,
        residual_details=residual_details,
        degrees_of_freedom=dof,
        variance_factor=variance_factor,
        chi_square_test=chi_test,
        covariance_matrix=cov_matrix.tolist() if cov_matrix is not None else None,
        point_covariances=point_covs,
        error_ellipses=error_ellipses,
        flagged_observations=flagged,
        network_name=network.name,
    )
